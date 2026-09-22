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
import sys
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
        "preview_views": config.DEFAULT_PREVIEW_VIEWS,
        "preview_camera": "preview",
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
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            # Models emit characters a Windows cp1252 console cannot encode. Losing a log line is
            # acceptable; crashing the build because of one is not.
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)
        if callable(self.opt.get("log")):
            try:
                self.opt["log"](msg)
            except Exception:
                pass
        try:
            self.session.debug(msg)
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
            "views": self.opt.get("preview_views", "single"),
            "camera": self.opt.get("preview_camera", "preview"),
            "fit_camera": True,
        }

    # ------------------------------------------------------------------ references
    def analyze_references(self, images_b64, user_notes="", prompt=""):
        """Deep multi-pass analysis. Returns (brief_text, analyses_list).

        Each image gets three focused passes (structure / detail / materials) instead of one
        "describe everything" question. A single broad prompt makes vision models answer vaguely -
        every part the same size, three generic components - which is what produced featureless
        results before. Sizes come back as fractions of one overall height and are converted to
        meters during the merge.
        """
        analyses = []
        model = self.client.cfg.get("vision_model") or self.client.cfg["model"]
        total = len(images_b64)
        for i, b64 in enumerate(images_b64, start=1):
            entry = {"image": i}
            for p in range(len(prompts.REF_PASSES)):
                key, msgs = prompts.build_reference_pass_messages(p, i, total, user_notes)
                self.log(f"Reference image {i}/{total}: {key} pass ({model})")
                text = self.client.chat(msgs, images=[b64], json_mode=True, model=model,
                                        temperature=0.1, max_tokens=4096)
                self.session.debug(f"REFERENCE IMAGE {i}/{total} - {key.upper()} PASS RAW REPLY",
                                   text, divider=True)
                data = extract_json(text)
                if not isinstance(data, (dict, list)):
                    self.log(f"  WARNING: {key} pass did not return usable JSON - "
                             f"the vision model may be too small or the reply was truncated")
                entry[key] = data if isinstance(data, (dict, list)) else {"raw": text[:2000]}
            entry = self._resolve_sizes(entry)
            self.session.debug(f"REFERENCE IMAGE {i}/{total} - RESOLVED TO METERS", entry)
            analyses.append(entry)
            self.log(f"  -> {len(entry.get('structure', {}).get('parts', []))} parts, "
                     f"{len(entry.get('detail', {}).get('details', []))} detail features, "
                     f"{len(entry.get('materials', {}).get('materials', []))} materials")
        if not analyses:
            return "", []
        self.log("Merging analyses into one build specification")
        msgs = prompts.build_reference_merge_messages(prompt, json.dumps(analyses, indent=1)[:20000])
        brief = self.client.chat(msgs, temperature=0.2, max_tokens=4096)
        if len(brief.strip()) < 200:  # merge produced nothing usable - fall back to the raw numbers
            self.log("  merge returned too little text; rebuilding the brief from the raw numbers")
            brief = self._brief_from_analysis(analyses[0], prompt)
        self.session.set_reference_brief(brief)
        self.session.debug("FINAL BUILD SPECIFICATION", brief, divider=True)
        return brief, analyses

    @staticmethod
    def _resolve_sizes(entry):
        """Convert every *_frac field into meters using the overall height from the structure pass."""
        s = entry.get("structure") or {}
        try:
            h = float(s.get("overall_height_m") or 0)
        except (TypeError, ValueError):
            h = 0.0
        if h <= 0:
            h = 0.3  # sane default for an unidentified object rather than a nonsense 0.1 everywhere
            s["overall_height_m"] = h
            s["overall_height_assumed"] = True
        try:
            w = float(s.get("max_width_m") or 0) or h * 0.6
        except (TypeError, ValueError):
            w = h * 0.6
        s["max_width_m"] = w

        def m(value, base):
            try:
                return round(float(value) * base, 4)
            except (TypeError, ValueError):
                return None

        for part in s.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            part["height_m"] = m(part.get("height_frac"), h)
            part["diameter_m"] = m(part.get("diameter_frac"), w)
            part["z_bottom_m"] = m(part.get("z_bottom_frac"), h)
            if part["height_m"] is not None and part["z_bottom_m"] is not None:
                part["z_top_m"] = round(part["z_bottom_m"] + part["height_m"], 4)
        for det in (entry.get("detail") or {}).get("details", []) or []:
            if isinstance(det, dict):
                det["size_m"] = m(det.get("size_frac"), h)
        entry["structure"] = s
        return entry

    @staticmethod
    def _brief_from_analysis(a, prompt=""):
        """Deterministic text brief straight from the JSON, used when the merge call is unusable."""
        s = a.get("structure") or {}
        lines = [f"OBJECT: {s.get('object', 'object')}, overall height {s.get('overall_height_m')} m, "
                 f"max width {s.get('max_width_m')} m",
                 f"STYLE: {s.get('style', '')}",
                 f"CONSTRUCTION: {s.get('construction', '')}", "PARTS (bottom to top):"]
        parts = [p for p in (s.get("parts") or []) if isinstance(p, dict)]
        parts.sort(key=lambda p: p.get("z_bottom_m") if isinstance(p.get("z_bottom_m"), (int, float)) else 0)
        for p in parts:
            lines.append(f"  - {p.get('name')} | {p.get('primitive')} | count {p.get('count', 1)} | "
                         f"height {p.get('height_m')} m | diameter {p.get('diameter_m')} m | "
                         f"z from {p.get('z_bottom_m')} m to {p.get('z_top_m')} m | "
                         f"{p.get('arrangement', '')} | {p.get('shape_notes', '')} | material {p.get('material', '')}")
        lines.append("DETAILS:")
        for d in ((a.get("detail") or {}).get("details") or []):
            if isinstance(d, dict):
                lines.append(f"  - {d.get('kind')} | count {d.get('count')} | size {d.get('size_m')} m | "
                             f"{d.get('where', '')} | {d.get('arrangement', '')} | {d.get('color', '')}")
        lines.append("MATERIALS:")
        for mt in ((a.get("materials") or {}).get("materials") or []):
            if isinstance(mt, dict):
                rgb = mt.get("rgb") or []
                lines.append(f"  - {mt.get('name')} | rgb {rgb} | metallic {mt.get('metallic')} | "
                             f"roughness {mt.get('roughness')} | alpha {mt.get('alpha', 1)} | used for {mt.get('used_for', '')}")
        mats = a.get("materials") or {}
        lines += [f"LIGHTING: {mats.get('lighting', '')}", f"CAMERA: {mats.get('camera', '')}"]
        if prompt.strip():
            lines.append(f"USER REQUEST: {prompt.strip()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ planning
    def plan(self, prompt, reference_brief="", max_steps=8):
        self.log(f"Planning (max {max_steps} steps) with {self.client.cfg['model']}")
        msgs = prompts.build_planner_messages(prompt, reference_brief or self.session.state.get("reference_brief", ""),
                                              self.session.scene_summary(), max_steps)
        self.session.debug("PLANNER - PROMPT SENT", divider=True)
        for m in msgs:
            self.session.debug(f"  [{m['role']}]", m["content"])
        text = self.client.chat(msgs, json_mode=True, temperature=0.2)
        self.session.debug("PLANNER - RAW MODEL REPLY", text)
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
            self.log("  planner returned no usable steps; building the whole request as one step")
            clean = [{"title": "Build", "category": "edit", "instruction": prompt.strip()}]
        self.session.set_plan(clean)
        self.session.debug("PLAN", self.plan_to_text(clean))
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
            self.session.debug(f"STEP {idx} '{title}' ATTEMPT {attempts} - PROMPT SENT", divider=True)
            for m in msgs:
                self.session.debug(f"  [{m['role']}]", m["content"])
            self.session.debug(f"STEP {idx} ATTEMPT {attempts} - RETRIEVED REFERENCE CHUNKS",
                               "\n".join(f"{c['source']} :: {c['title']} (score {c['score']})" for c in chunks))
            text = self.client.chat(msgs)
            self.session.debug(f"STEP {idx} ATTEMPT {attempts} - RAW MODEL REPLY", text)
            code = extract_code(text)
            if not code.strip():
                feedback = "Your reply contained no ```python code block. Return the complete script in one code block."
                previous_code = text[:4000]
                continue
            code = self._with_header(code, idx, title, attempts)
            script_path = self.session.next_script_path(attempts)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            self.session.debug(f"STEP {idx} ATTEMPT {attempts} - GENERATED SCRIPT ({script_path})", code)
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
            self.session.debug(f"STEP {idx} ATTEMPT {attempts} - BLENDER RESULT", {
                "ok": result.get("ok"), "error": result.get("error"),
                "built": result.get("built"), "timings": result.get("timings"),
                "render_path": result.get("render_path"), "saved_blend": result.get("saved_blend"),
                "validation_totals": (result.get("validation") or {}).get("totals"),
                "validation_issues": [i for i in (result.get("validation") or {}).get("issues", [])
                                      if i.get("severity") == "error"][:20],
                "api_hint": result.get("api_hint"),
            })
            if result.get("traceback"):
                self.session.debug(f"STEP {idx} ATTEMPT {attempts} - TRACEBACK", result["traceback"])
            if result.get("stdout"):
                self.session.debug(f"STEP {idx} ATTEMPT {attempts} - BLENDER STDOUT", result["stdout"][-6000:])
            if result.get("ok"):
                self.log(f"Step {idx} OK in {result.get('timings', {}).get('total_s', '?')}s")
                break
            feedback = feedback_for_retry(result)
            previous_code = code
            self.session.debug(f"STEP {idx} ATTEMPT {attempts} - FEEDBACK FOR RETRY", feedback)
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
                "report": report, "turn": turn, "render_path": (result or {}).get("render_path"),
                "render_paths": (result or {}).get("render_paths") or []}

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

    # ------------------------------------------------------------------ visual critic / refine
    def render_current(self, camera_direction=None, width=None, height=None, samples=None):
        """Render the session scene as it stands, without changing it. Returns the image path."""
        script_path = self.session.next_script_path(1, label="view")
        code = "from gap_helpers import *\nimport bpy\nprint('view render of', len(bpy.data.objects), 'objects')\n"
        if camera_direction:
            code += (f"cam = bpy.context.scene.camera or add_camera(name='AI_Preview_Camera')\n"
                     f"frame_camera_to_scene(cam, margin=1.05, direction={tuple(camera_direction)})\n")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        render = self._render_cfg(script_path) or {
            "path": self.session.render_path_for(script_path), "engine": self.opt["render_engine"],
            "width": int(self.opt["render_width"]), "height": int(self.opt["render_height"]),
            "samples": int(self.opt["render_samples"]), "fit_camera": True,
        }
        if width:
            render["width"] = int(width)
        if height:
            render["height"] = int(height)
        if samples:
            render["samples"] = int(samples)
        result = self.runner.run(script_path, mode=self.opt["execution_mode"], validate=False,
                                 render=render, save=False, probe=True)
        if result.get("scene"):
            self.session.state["last_scene"] = result["scene"]
            self.session.save()
        return result.get("render_path"), result   # critic compares one view; all views are in render_paths

    def critique(self, reference_images_b64, render_path, reference_brief=""):
        """Compare the current render against the reference images. Returns the critique dict."""
        if not render_path or not os.path.exists(render_path):
            return {"score": 0, "verdict": "no render available to critique", "differences": [], "keep": []}
        import base64
        with open(render_path, "rb") as f:
            render_b64 = base64.b64encode(f.read()).decode("utf-8")
        model = self.client.cfg.get("vision_model") or self.client.cfg["model"]
        self.log(f"Critiquing the render against the reference with {model}")
        msgs = prompts.build_critic_messages(reference_brief or self.session.state.get("reference_brief", ""),
                                             self.session.scene_summary())
        images = list(reference_images_b64[:1]) + [render_b64]
        text = self.client.chat(msgs, images=images, json_mode=True, model=model,
                                temperature=0.1, max_tokens=3072)
        data = extract_json(text)
        if not isinstance(data, dict):
            return {"score": 0, "verdict": "critique could not be parsed", "differences": [],
                    "keep": [], "raw": text[:2000]}
        data.setdefault("differences", [])
        data.setdefault("keep", [])
        order = {"critical": 0, "major": 1, "minor": 2}
        data["differences"].sort(key=lambda d: order.get(str(d.get("importance", "major")).lower(), 1))
        self.log(f"  score {data.get('score')}/100 - {data.get('verdict', '')[:120]}")
        for d in data["differences"][:6]:
            self.log(f"    [{d.get('importance', '?')}] {str(d.get('issue'))[:110]}")
        return data

    def correction_steps(self, critique, reference_brief="", max_steps=3):
        """Turn a critique into ordered build steps that repair the model."""
        diffs = [d for d in critique.get("differences", [])
                 if str(d.get("importance", "")).lower() != "minor"]
        if not diffs:
            return []
        self.log(f"Planning corrections for {len(diffs)} difference(s)")
        msgs = prompts.build_correction_messages(
            json.dumps({"verdict": critique.get("verdict"), "differences": diffs,
                        "keep": critique.get("keep", [])}, indent=1)[:8000],
            self.session.scene_summary(),
            reference_brief or self.session.state.get("reference_brief", ""),
            max_steps)
        text = self.client.chat(msgs, json_mode=True, temperature=0.2, max_tokens=3072)
        data = extract_json(text)
        steps = (data.get("steps") if isinstance(data, dict) else data) or []
        clean = []
        for i, s in enumerate(steps[:max_steps], start=1):
            if isinstance(s, dict) and s.get("instruction"):
                clean.append({"title": str(s.get("title") or f"Fix {i}")[:80],
                              "category": str(s.get("category") or "edit"),
                              "instruction": str(s["instruction"])})
        return clean

    def refine(self, reference_images_b64, reference_brief="", rounds=2, target_score=85,
               max_fix_steps=3, camera_direction=None):
        """Render -> critique -> fix -> repeat. The loop that actually converges on the reference.

        Returns (history, last_render_path). history is one entry per round with the score,
        the critique and the steps that were executed.
        """
        history = []
        render_path = None
        for round_index in range(1, int(rounds) + 1):
            self.log(f"=== Refine round {round_index}/{rounds}: rendering current scene ===")
            render_path, _ = self.render_current(camera_direction=camera_direction)
            critique = self.critique(reference_images_b64, render_path, reference_brief)
            score = critique.get("score") or 0
            entry = {"round": round_index, "score": score, "render": render_path,
                     "critique": critique, "steps": [], "results": []}
            history.append(entry)
            if score >= target_score:
                self.log(f"Score {score} reached the target ({target_score}); stopping refinement.")
                break
            steps = self.correction_steps(critique, reference_brief, max_fix_steps)
            if not steps:
                self.log("No actionable corrections returned; stopping refinement.")
                break
            entry["steps"] = steps
            results, report = self.run_plan(steps, reference_brief, stop_on_failure=False)
            entry["results"] = results
            entry["report"] = report
            self.log(f"Round {round_index}: applied {sum(1 for r in results if r['ok'])}/{len(results)} fixes")
        if history:
            self.session.state["refine_history"] = [
                {"round": h["round"], "score": h["score"], "verdict": h["critique"].get("verdict"),
                 "render": h["render"], "steps": [s["title"] for s in h["steps"]]} for h in history]
            self.session.save()
        return history, render_path

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
