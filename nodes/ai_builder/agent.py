# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
The agent loop: reference analysis -> plan -> for each step: generate -> safety scan -> execute
-> validate -> (retry with feedback) -> record in session.

Nothing here is hidden: every prompt is in prompts.py, every script and result is written to the
session folder, and `dry_run` stops right after generation so you can read the code first.
"""

import json
import os
import re
import time

from . import config, prompts
from .blender_runner import BlenderRunner
from .llm import LLMClient, extract_code, extract_json
from .rag import ReferenceIndex
from .safety import format_scan, scan_code
from .validation import feedback_for_retry, format_step_report, summarize_validation


def _slug(text, n=28):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return (s[:n] or "Step").rstrip("_")


class AgentOptions(dict):
    DEFAULTS = {
        "execution_mode": "headless",   # headless | live
        "dry_run": False,
        "max_retries": config.DEFAULT_MAX_RETRIES,
        "validate": True,
        "auto_fix_normals": True,
        "auto_fix_doubles": True,
        "safety_scan": True,
        "render_preview": True,
        "render_engine": config.DEFAULT_RENDER_ENGINE,
        "render_width": config.DEFAULT_RENDER_WIDTH,
        "render_height": config.DEFAULT_RENDER_HEIGHT,
        "render_samples": config.DEFAULT_RENDER_SAMPLES,
        "blender_path": "",
        "blender_timeout": config.DEFAULT_BLENDER_TIMEOUT,
        "live_host": "127.0.0.1",
        "live_port": 8119,
        "live_timeout": config.DEFAULT_LIVE_TIMEOUT,
        "live_save": False,
        "rag_k": 6,
        "extra_reference_text": "",
        "log": None,                    # callable(str) for progress
    }

    def __init__(self, **kw):
        super().__init__(self.DEFAULTS)
        self.update({k: v for k, v in kw.items() if v is not None})


class SceneBuilderAgent:
    def __init__(self, session, llm_cfg, options=None):
        self.session = session
        self.client = LLMClient(llm_cfg)
        self.opt = options if isinstance(options, AgentOptions) else AgentOptions(**(options or {}))
        self.index = ReferenceIndex(extra_text=self.opt["extra_reference_text"])
        self.runner = BlenderRunner(
            session, blender_path=self.opt["blender_path"], timeout=self.opt["blender_timeout"],
            live_host=self.opt["live_host"], live_port=self.opt["live_port"], live_timeout=self.opt["live_timeout"],
        )
        self.blender_version = self._detect_blender_version()

    # ------------------------------------------------------------------ utils
    def log(self, msg):
        line = f"[AI Scene Builder] {msg}"
        print(line, flush=True)
        if callable(self.opt.get("log")):
            try:
                self.opt["log"](msg)
            except Exception:
                pass

    def _detect_blender_version(self):
        path = self.runner.blender_path or ""
        m = re.search(r"Blender\s*(\d+\.\d+)", path)
        if m:
            return m.group(1)
        return (self.session.state.get("last_scene") or {}).get("blender_version", "4.x/5.x")

    def _render_cfg(self, script_path):
        if not self.opt["render_preview"]:
            return None
        return {
            "path": self.session.render_path_for(script_path),
            "engine": self.opt["render_engine"],
            "width": int(self.opt["render_width"]),
            "height": int(self.opt["render_height"]),
            "samples": int(self.opt["render_samples"]),
            "fit_camera": True,
        }

    # ------------------------------------------------------------------ references
    def analyze_references(self, images_b64, user_notes="", prompt=""):
        """images_b64: list of base64 PNG strings. Returns (brief_text, analyses_list)."""
        analyses = []
        model = self.client.cfg.get("vision_model") or self.client.cfg["model"]
        total = len(images_b64)
        for i, b64 in enumerate(images_b64, start=1):
            self.log(f"Analyzing reference image {i}/{total} with {model}")
            msgs = prompts.build_reference_analysis_messages(i, total, user_notes)
            text = self.client.chat(msgs, images=[b64], json_mode=True, model=model, temperature=0.1)
            data = extract_json(text)
            analyses.append(data if isinstance(data, dict) else {"raw": text[:2000]})
        if not analyses:
            return "", []
        if total == 1 and not prompt.strip():
            brief = self._brief_from_single(analyses[0])
        else:
            self.log("Merging reference analyses into one brief")
            msgs = prompts.build_reference_merge_messages(prompt, json.dumps(analyses, indent=1)[:12000])
            brief = self.client.chat(msgs, temperature=0.2)
        self.session.set_reference_brief(brief)
        return brief, analyses

    @staticmethod
    def _brief_from_single(a):
        if "raw" in a:
            return a["raw"]
        lines = [f"SUBJECT: {a.get('subject', '')}", f"TYPE: {a.get('type', '')}", f"STYLE: {a.get('style', '')}",
                 f"LAYOUT: {a.get('layout', '')}", "ELEMENTS:"]
        for e in a.get("elements", []) or []:
            if isinstance(e, dict):
                lines.append(f"  - {e.get('name', '?')}: {e.get('description', '')} | size {e.get('approx_size_m', '?')} | "
                             f"at {e.get('position', '?')} | {e.get('materials', '')}")
        lines += [f"MATERIALS: {', '.join(map(str, a.get('materials', []) or []))}",
                  f"COLORS: {', '.join(map(str, a.get('colors', []) or []))}",
                  f"LIGHTING: {a.get('lighting', '')}", f"CAMERA: {a.get('camera', '')}", f"MOOD: {a.get('mood', '')}",
                  f"MUST-NOT-MISS: {a.get('notes_for_3d', '')}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ planning
    def plan(self, prompt, reference_brief="", max_steps=8):
        self.log(f"Planning (max {max_steps} steps) with {self.client.cfg['model']}")
        msgs = prompts.build_planner_messages(prompt, reference_brief or self.session.state.get("reference_brief", ""),
                                              self.session.scene_summary(), max_steps)
        text = self.client.chat(msgs, json_mode=True, temperature=0.2)
        data = extract_json(text)
        steps = []
        if isinstance(data, dict):
            steps = data.get("steps") or data.get("plan") or []
        elif isinstance(data, list):
            steps = data
        clean = []
        for i, s in enumerate(steps[:max_steps], start=1):
            if isinstance(s, str):
                clean.append({"title": f"Step {i}", "category": "edit", "instruction": s})
            elif isinstance(s, dict) and (s.get("instruction") or s.get("description")):
                clean.append({"title": str(s.get("title") or f"Step {i}")[:80],
                              "category": str(s.get("category") or "edit"),
                              "instruction": str(s.get("instruction") or s.get("description"))})
        if not clean:
            clean = [{"title": "Build", "category": "edit", "instruction": prompt.strip()}]
        self.session.set_plan(clean)
        return clean

    @staticmethod
    def parse_manual_plan(text):
        """One step per non-empty line; 'Title: instruction' or just 'instruction'. Also accepts JSON."""
        text = (text or "").strip()
        if not text:
            return []
        data = extract_json(text) if text.startswith(("{", "[")) else None
        if data:
            steps = data.get("steps") if isinstance(data, dict) else data
            return [{"title": str(s.get("title", f"Step {i}"))[:80], "category": str(s.get("category", "edit")),
                     "instruction": str(s.get("instruction", ""))} for i, s in enumerate(steps, start=1)
                    if isinstance(s, dict) and s.get("instruction")]
        steps = []
        for i, line in enumerate([l.strip() for l in text.splitlines() if l.strip()], start=1):
            line = re.sub(r"^\s*(\d+[\).:-]|[-*])\s*", "", line)
            if ":" in line and len(line.split(":", 1)[0]) <= 60:
                title, instr = line.split(":", 1)
                steps.append({"title": title.strip()[:80], "category": "edit", "instruction": instr.strip() or title.strip()})
            else:
                steps.append({"title": f"Step {i}", "category": "edit", "instruction": line})
        return steps

    @staticmethod
    def plan_to_text(steps):
        return "\n".join(f"{i}. [{s.get('category', 'edit')}] {s['title']}: {s['instruction']}" for i, s in enumerate(steps, start=1))

    # ------------------------------------------------------------------ one step
    def run_step(self, instruction, title="Step", reference_brief="", category="edit"):
        idx = self.session.turn_count + 1
        collection_name = f"Step_{idx:02d}_{_slug(title)}"
        reference_brief = reference_brief or self.session.state.get("reference_brief", "")
        scene_is_new = not self.session.blend_exists()
        rag_query = f"{title} {instruction} {category} blender python bpy"
        chunks = self.index.retrieve(rag_query, k=int(self.opt["rag_k"]), client=self.client)
        rag_context = ReferenceIndex.format_context(chunks)
        feedback, previous_code, code, script_path, result = None, None, "", "", None
        attempts = 0
        scan = {"blocked": [], "warnings": []}
        while attempts <= int(self.opt["max_retries"]):
            attempts += 1
            self.log(f"Step {idx} '{title}' - generating script (attempt {attempts}) with {self.client.cfg['model']}")
            msgs = prompts.build_codegen_messages(
                instruction, self.session.scene_summary(), self.session.history_summary(), reference_brief,
                rag_context, self.blender_version, collection_name, scene_is_new, feedback, previous_code)
            text = self.client.chat(msgs)
            code = extract_code(text)
            if not code.strip():
                feedback = "Your reply contained no ```python code block. Return the complete script in one code block."
                previous_code = text[:4000]
                continue
            code = self._with_header(code, idx, title, attempts)
            script_path = self.session.next_script_path(attempts)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            scan = scan_code(code) if self.opt["safety_scan"] else {"blocked": [], "warnings": []}
            if not scene_is_new and self.opt["safety_scan"]:
                # Protect the existing scene: wiping it is only allowed when the instruction asks for it.
                asks_to_clear = re.search(r"\b(clear|wipe|empty|start over|from scratch|remove (all|everything)|delete (all|everything)|replace the scene|reset)\b",
                                          instruction, flags=re.IGNORECASE)
                destructive = [w for w in scan["warnings"] if "clears the whole scene" in w]
                if destructive and not asks_to_clear:
                    scan["blocked"] = scan["blocked"] + [w + "  (existing scene must not be cleared unless the instruction asks)" for w in destructive]
            if scan["blocked"]:
                self.log("Safety scan blocked the script:\n" + format_scan(scan))
                feedback = ("The script was REJECTED by the safety scan (not executed). Remove these constructs:\n"
                            + "\n".join(scan["blocked"]) + "\nUse only bpy/bmesh/mathutils/math/random/gap_helpers.")
                previous_code = code
                result = {"ok": False, "error": "blocked by safety scan", "safety": scan}
                continue
            if self.opt["dry_run"]:
                result = {"ok": True, "dry_run": True, "error": None, "safety": scan, "stdout": "", "built": [],
                          "validation": None, "scene": None, "render_path": None, "timings": {}}
                self.log(f"Dry run - script saved to {script_path} (not executed)")
                break
            self.log(f"Executing {os.path.basename(script_path)} in Blender ({self.opt['execution_mode']})")
            result = self.runner.run(
                script_path, mode=self.opt["execution_mode"], validate=self.opt["validate"],
                auto_fix_normals=self.opt["auto_fix_normals"], auto_fix_doubles=self.opt["auto_fix_doubles"],
                render=self._render_cfg(script_path),
                save=(self.opt["live_save"] if self.opt["execution_mode"] == "live" else True),
            )
            result["safety"] = scan
            if result.get("ok"):
                self.log(f"Step {idx} OK in {result.get('timings', {}).get('total_s', '?')}s")
                break
            feedback = feedback_for_retry(result)
            previous_code = code
            self.log(f"Step {idx} failed: {str(result.get('error'))[:300]}")
        ok = bool(result and result.get("ok"))
        summary = self._turn_summary(title, instruction, result)
        turn = self.session.add_turn({
            "title": title, "category": category, "instruction": instruction, "collection": collection_name,
            "ok": ok, "dry_run": bool(self.opt["dry_run"]), "attempts": attempts, "script": script_path,
            "result": self.session.result_path_for(script_path) if script_path else "",
            "render": (result or {}).get("render_path"), "summary": summary,
            "scene": (result or {}).get("scene") or None, "validation": (result or {}).get("validation") or None,
            "error": (result or {}).get("error"), "model": self.client.cfg["model"],
        })
        report = format_step_report(turn["index"], title, result or {}, attempts, script_path)
        if scan["warnings"]:
            report += "\nsafety warnings: " + "; ".join(scan["warnings"][:6])
        return {"ok": ok, "result": result or {}, "code": code, "script_path": script_path, "attempts": attempts,
                "report": report, "turn": turn, "render_path": (result or {}).get("render_path")}

    def _with_header(self, code, idx, title, attempt):
        header = (f"# Generated by ComfyUI-Blender-Toolbox AI Scene Builder (Geekatplay Studio - Vladimir Chopine)\n"
                  f"# session: {self.session.name} | step {idx}: {title} | attempt {attempt}\n"
                  f"# model: {self.client.cfg['provider']}/{self.client.cfg['model']} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# Review this file before trusting it. It runs with full Blender Python access.\n")
        if "from gap_helpers import" not in code and "import gap_helpers" not in code:
            code = "from gap_helpers import *\n" + code
        if "import bpy" not in code:
            code = "import bpy\n" + code
        return header + code

    @staticmethod
    def _turn_summary(title, instruction, result):
        if not result:
            return f"{title}: no result"
        if result.get("dry_run"):
            return f"{title}: script generated (dry run) - {instruction[:100]}"
        built = result.get("built") or []
        names = ", ".join(b["name"] for b in built[:8])
        v = result.get("validation") or {}
        t = v.get("totals", {})
        status = "OK" if result.get("ok") else f"FAILED ({str(result.get('error'))[:80]})"
        return f"{title} [{status}] built {len(built)} objects ({names}) | {t.get('vertices', 0)} verts, {t.get('errors', 0)} errors"

    # ------------------------------------------------------------------ plan execution
    def run_plan(self, steps, reference_brief="", stop_on_failure=True):
        reports, results = [], []
        for i, step in enumerate(steps, start=1):
            self.log(f"=== Plan step {i}/{len(steps)}: {step['title']} ===")
            r = self.run_step(step["instruction"], step["title"], reference_brief, step.get("category", "edit"))
            reports.append(r["report"])
            results.append(r)
            if not r["ok"] and stop_on_failure and not self.opt["dry_run"]:
                reports.append(f"Stopped after step {i} failed (stop_on_failure=True).")
                break
        return results, "\n\n".join(reports)

    # ------------------------------------------------------------------ user-provided script
    def run_script(self, code, title="Manual script"):
        idx = self.session.turn_count + 1
        script_path = self.session.next_script_path(1, label="manual")
        code = self._with_header(code, idx, title, 1)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        scan = scan_code(code) if self.opt["safety_scan"] else {"blocked": [], "warnings": []}
        if scan["blocked"]:
            result = {"ok": False, "error": "blocked by safety scan:\n" + "\n".join(scan["blocked"]), "safety": scan,
                      "stdout": "", "built": [], "validation": None, "scene": None, "render_path": None, "timings": {}}
        else:
            result = self.runner.run(
                script_path, mode=self.opt["execution_mode"], validate=self.opt["validate"],
                auto_fix_normals=self.opt["auto_fix_normals"], auto_fix_doubles=self.opt["auto_fix_doubles"],
                render=self._render_cfg(script_path),
                save=(self.opt["live_save"] if self.opt["execution_mode"] == "live" else True),
            )
            result["safety"] = scan
        turn = self.session.add_turn({
            "title": title, "category": "manual", "instruction": "(user-provided script)", "ok": bool(result.get("ok")),
            "attempts": 1, "script": script_path, "result": self.session.result_path_for(script_path),
            "render": result.get("render_path"), "summary": self._turn_summary(title, "", result),
            "scene": result.get("scene"), "validation": result.get("validation"), "error": result.get("error"),
        })
        report = format_step_report(turn["index"], title, result, 1, script_path)
        return {"ok": bool(result.get("ok")), "result": result, "code": code, "script_path": script_path,
                "attempts": 1, "report": report, "turn": turn, "render_path": result.get("render_path")}

    # ------------------------------------------------------------------ validate only
    def validate_only(self, auto_fix_normals=False, auto_fix_doubles=False, render=True):
        code = ("from gap_helpers import *\nimport bpy\nprint('validation pass over', len(bpy.data.objects), 'objects')\n")
        script_path = self.session.next_script_path(1, label="validate")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        result = self.runner.run(
            script_path, mode=self.opt["execution_mode"], validate=True, auto_fix_normals=auto_fix_normals,
            auto_fix_doubles=auto_fix_doubles, render=self._render_cfg(script_path) if render else None,
            save=bool(auto_fix_normals or auto_fix_doubles) if self.opt["execution_mode"] == "headless" else self.opt["live_save"],
        )
        v = result.get("validation")
        if v:
            self.session.state["last_validation"] = v
            if result.get("scene"):
                self.session.state["last_scene"] = result["scene"]
            self.session.save()
        return result, summarize_validation(v) if v else str(result.get("error"))
