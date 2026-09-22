# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder nodes
"""
ComfyUI nodes for the AI Scene Builder.

Pipeline (see docs/ai_builder/AI_SCENE_BUILDER.md):
  AI Scene Session -> AI Model Config -> [AI Reference Analyzer] -> AI Scene Planner -> AI Scene Builder
  or, for conversational passes:  AI Scene Session + AI Model Config -> AI Step Builder (one instruction per run)
  Review-first flow:  builder with dry_run=True -> read/edit the script -> AI Script Runner
  QA only:            AI Scene Validator

Every node prints what it does; nothing runs in Blender without being saved to the session folder first.
"""

import base64
import io
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image

from .ai_builder import bridge, config
from .ai_builder.agent import AgentOptions, SceneBuilderAgent
from .ai_builder.llm import LLMConfig, LLMClient
from .ai_builder.session import SceneSession
from .ai_builder.validation import summarize_validation

CATEGORY = "Geekatplay Studio/AI Scene Builder"
SESSION_TYPE = "GAP_AI_SESSION"
LLM_TYPE = "GAP_AI_LLM"

print("[AI Scene Builder] " + config.SANDBOX_WARNING.splitlines()[0] + " - see docs/ai_builder/AI_SCENE_BUILDER.md")


# --------------------------------------------------------------------------- image helpers
def _tensor_to_b64_list(images, max_side=1024):
    """Each image of a ComfyUI batch -> base64 PNG (downscaled so vision prompts stay small)."""
    out = []
    if images is None:
        return out
    for i in range(images.shape[0]):
        arr = (255.0 * images[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return out


def _load_image_tensor(path, fallback_size=(768, 512), text=""):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", fallback_size, (40, 40, 44))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _load_image_batch(paths, fallback_size=(768, 512)):
    """Several renders of the same scene as one IMAGE batch, so Preview Image shows every view.

    Views are rendered at the same resolution; any that differ are skipped rather than crashing
    torch.cat, and a missing set falls back to a single placeholder.
    """
    paths = [p for p in (paths or []) if p and os.path.exists(p)]
    if not paths:
        return _blank_image(*fallback_size)
    tensors, shape = [], None
    for p in paths:
        t = _load_image_tensor(p, fallback_size)
        if shape is None:
            shape = t.shape[1:3]
        if tuple(t.shape[1:3]) == tuple(shape):
            tensors.append(t)
    return torch.cat(tensors, dim=0) if tensors else _blank_image(*fallback_size)


def _blank_image(w=768, h=512):
    return torch.zeros((1, h, w, 3), dtype=torch.float32) + 0.16


def _collect_render_paths(result):
    """Every view a step rendered, newest API first, falling back to the single-path field."""
    paths = result.get("render_paths") or []
    if not paths and result.get("render_path"):
        paths = [result["render_path"]]
    return [p for p in paths if p]


def _save_refs(session, images_b64):
    paths = []
    for i, b64 in enumerate(images_b64, start=1):
        p = os.path.join(session.refs_dir, f"ref_{time.strftime('%Y%m%d_%H%M%S')}_{i}.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(b64))
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- shared inputs
def _exec_inputs():
    """Execution settings shared by the builder / step / script-runner nodes."""
    return {
        "execution_mode": (["headless", "live"], {"default": "headless",
                           "tooltip": "headless: run Blender in the background on the session .blend (safe default; "
                                      "you see the result as a preview render, NOT in your open Blender). "
                                      "live: build inside the Blender you have open so you watch it appear - requires "
                                      "the toolbox addon running with 'Start Listener' AND 'Allow AI code execution' ON."}),
        "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Generate and SAVE the script but do not execute it. Review it, then use AI Script Runner."}),
        "max_retries": ("INT", {"default": config.DEFAULT_MAX_RETRIES, "min": 0, "max": 6,
                        "tooltip": "How many times the model may fix its own script after an error or failed validation."}),
        "validate": ("BOOLEAN", {"default": True, "tooltip": "Check polygons, normals, textures and names after every step."}),
        "auto_fix_normals": ("BOOLEAN", {"default": True, "tooltip": "Recalculate flipped normals outward automatically."}),
        "auto_fix_doubles": ("BOOLEAN", {"default": True, "tooltip": "Merge duplicate vertices and delete loose geometry automatically."}),
        "safety_scan": ("BOOLEAN", {"default": True, "tooltip": "Block scripts that touch files, network or processes. Keyword scan, not a sandbox."}),
        "render_preview": ("BOOLEAN", {"default": True}),
        "preview_views": (["quad", "single", "six"], {"default": config.DEFAULT_PREVIEW_VIEWS,
                          "tooltip": "quad: three-quarter + front + right + top, as one image batch - "
                                     "a single angle hides everything behind the object, which is when "
                                     "parts look piled together. single: just the three-quarter view "
                                     "(fastest). six: adds back and left."}),
        "preview_camera": (["preview", "scene"], {"default": "preview",
                           "tooltip": "preview: the builder frames its own camera on the scene (reliable). "
                                      "scene: use the camera the generated script created - only useful "
                                      "once you trust it to aim the camera well."}),
        "render_engine": (["CYCLES", "BLENDER_EEVEE"], {"default": config.DEFAULT_RENDER_ENGINE}),
        "render_width": ("INT", {"default": config.DEFAULT_RENDER_WIDTH, "min": 64, "max": 4096, "step": 8}),
        "render_height": ("INT", {"default": config.DEFAULT_RENDER_HEIGHT, "min": 64, "max": 4096, "step": 8}),
        "render_samples": ("INT", {"default": config.DEFAULT_RENDER_SAMPLES, "min": 1, "max": 1024}),
        "blender_timeout": ("INT", {"default": config.DEFAULT_BLENDER_TIMEOUT, "min": 30, "max": 7200,
                            "tooltip": "Seconds allowed per step in headless mode."}),
    }


def _exec_optional_inputs():
    return {
        "blender_path": ("STRING", {"default": "", "tooltip": "Override the auto-detected blender executable."}),
        "live_host": ("STRING", {"default": "127.0.0.1"}),
        "live_port": ("INT", {"default": 8119, "min": 1024, "max": 65535}),
        "live_save": ("BOOLEAN", {"default": False, "tooltip": "In live mode also save the running Blender file to the session .blend path."}),
        "extra_reference_notes": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Your own notes/API snippets. Added to the retrieval index for this run."}),
    }


def _options_from(kwargs, log=None):
    keys = ["execution_mode", "dry_run", "max_retries", "validate", "auto_fix_normals", "auto_fix_doubles", "safety_scan",
            "render_preview", "preview_views", "preview_camera", "render_engine", "render_width", "render_height",
            "render_samples", "blender_timeout", "blender_path", "live_host", "live_port", "live_save"]
    opts = {k: kwargs[k] for k in keys if k in kwargs}
    opts["extra_reference_text"] = kwargs.get("extra_reference_notes", "")
    opts["log"] = log
    return AgentOptions(**opts)


def _collect_outputs(session, results, report):
    renders = []
    scripts, validations = [], []
    for r in results:
        paths = _collect_render_paths(r.get("result") or {}) or _collect_render_paths(r)
        if paths:
            renders = paths          # keep the last step's views, which show the scene as it now is
        if r.get("code"):
            scripts.append(f"# ===== {os.path.basename(r.get('script_path', 'script'))} =====\n{r['code']}")
        v = (r.get("result") or {}).get("validation")
        if v:
            validations.append({"script": os.path.basename(r.get("script_path", "")), "validation": v})
    preview = _load_image_batch(renders)
    header = f"AI Scene Builder report - session '{session.name}'\nfolder: {session.root}\nblend: {session.blend_path}\n" \
             f"NOTE: scripts ran with full Blender Python access (see docs/ai_builder/AI_SCENE_BUILDER.md#security).\n\n"
    return preview, session.blend_path, header + report, "\n\n".join(scripts), json.dumps(validations, indent=1)


# =========================================================================== nodes
class GapAISceneSession:
    """Create or continue a scene session. The session folder holds the .blend, scripts, results, renders."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_name": ("STRING", {"default": "my_scene", "tooltip": "Folder name under ComfyUI/output/ai_scene_builder/"}),
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Archive the existing session (scene + scripts) and start empty. Turn OFF again after one run."}),
            },
            "optional": {
                "blend_file": ("STRING", {"default": "", "tooltip": "Optional: continue from an existing .blend file instead of the session's scene.blend"}),
                "notes": ("STRING", {"multiline": True, "default": "", "tooltip": "Free notes stored in session.json"}),
            },
        }

    RETURN_TYPES = (SESSION_TYPE, "STRING")
    RETURN_NAMES = ("session", "session_info")
    FUNCTION = "open_session"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def open_session(self, session_name, reset, blend_file="", notes=""):
        session = SceneSession(session_name).open(reset=reset and True, blend_path=blend_file.strip())
        if notes.strip():
            session.state["notes"] = notes.strip()
            session.save()
        info = session.info_text()
        if reset:
            info += "\n  WARNING: 'reset' is ON - every run archives the session. Turn it off after this run."
        print("[AI Scene Builder] " + info.replace("\n", "\n  "))
        return (session.to_payload(), info)


class GapAILLMConfig:
    """Which model does the thinking. Ollama (local) by default; Anthropic and OpenAI-compatible APIs supported."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["ollama", "anthropic", "openai_compatible"], {"default": "ollama"}),
                "model": ("STRING", {"default": config.DEFAULT_CODE_MODEL, "tooltip": "Code/planning model. 'auto' picks the strongest coder installed in Ollama (recommended). Or pin one: qwen2.5-coder:32b, qwen3-coder:30b. Anthropic: claude-sonnet-5 / claude-opus-5."}),
                "vision_model": ("STRING", {"default": config.DEFAULT_VISION_MODEL, "tooltip": "Model that reads the reference images. 'auto' picks the largest vision model installed (recommended - size matters more here than anywhere else). For anthropic/openai leave as 'auto' to reuse the main model."}),
                "url": ("STRING", {"default": config.DEFAULT_OLLAMA_URL, "tooltip": "Ollama: http://127.0.0.1:11434 (or https://ollama.com). Anthropic: https://api.anthropic.com. OpenAI-compatible: base URL without /v1."}),
                "temperature": ("FLOAT", {"default": config.DEFAULT_TEMPERATURE, "min": 0.0, "max": 1.5, "step": 0.05}),
                "num_ctx": ("INT", {"default": config.DEFAULT_NUM_CTX, "min": 2048, "max": 262144, "step": 1024, "tooltip": "Ollama context window. Bigger = more scene state fits, more VRAM."}),
                "max_tokens": ("INT", {"default": config.DEFAULT_MAX_TOKENS, "min": 512, "max": 65536, "step": 256}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "password": True, "tooltip": "Leave empty to use the API Key Manager (names: Anthropic, OpenAI, Ollama Cloud) or env vars."}),
                "embed_model": ("STRING", {"default": config.DEFAULT_EMBED_MODEL, "tooltip": "Ollama embedding model for doc retrieval. 'auto' picks one if installed; empty = keyword search only."}),
                "keep_alive": ("STRING", {"default": config.DEFAULT_KEEP_ALIVE}),
                "timeout_s": ("INT", {"default": config.DEFAULT_TIMEOUT[1], "min": 30, "max": 7200}),
            },
        }

    RETURN_TYPES = (LLM_TYPE, "STRING")
    RETURN_NAMES = ("llm", "llm_info")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, provider, model, vision_model, url, temperature, num_ctx, max_tokens,
              api_key="", embed_model=config.DEFAULT_EMBED_MODEL, keep_alive=config.DEFAULT_KEEP_ALIVE, timeout_s=900):
        url = url.strip()
        if provider == "anthropic" and ("11434" in url or not url):
            url = config.DEFAULT_ANTHROPIC_URL
        if provider == "openai_compatible" and ("11434" in url or not url):
            url = config.DEFAULT_OPENAI_URL
        model, vision_model, embed_model = model.strip(), vision_model.strip(), embed_model.strip()
        auto = config.AUTO_MODEL
        if provider != "ollama":
            # Cloud providers have no model list to query; 'auto' just reuses the main model.
            if model.lower() == auto:
                model = config.DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else config.DEFAULT_OPENAI_MODEL
            if vision_model.lower() == auto or not vision_model:
                vision_model = model
            if embed_model.lower() == auto:
                embed_model = ""

        cfg = LLMConfig(provider=provider, model=model, vision_model=vision_model, url=url,
                        api_key=api_key, temperature=temperature, num_ctx=num_ctx, max_tokens=max_tokens,
                        embed_model=embed_model, keep_alive=keep_alive, timeout=(10, int(timeout_s)))
        info = ""
        if provider == "ollama":
            client = LLMClient(cfg)
            details = client.ollama_model_details()
            available = list(details) or client.list_ollama_models()
            if not available:
                info = cfg.describe() + f"\nWARNING: could not list models at {url} - is Ollama running?"
                return (dict(cfg), info)

            picks = client.pick_best_models(details)
            chosen = []
            for field, key, fallback in (("model", "code", config.FALLBACK_CODE_MODEL),
                                         ("vision_model", "vision", config.FALLBACK_VISION_MODEL),
                                         ("embed_model", "embed", config.FALLBACK_EMBED_MODEL)):
                if str(cfg[field]).lower() != auto:
                    continue
                best = picks.get(key)
                if best:
                    cfg[field] = best
                    chosen.append(f"  {field:<12} auto -> {picks['reasons'].get(key, best)}")
                elif key == "embed":
                    cfg[field] = ""      # retrieval falls back to keyword search, which is fine
                    chosen.append("  embed_model  auto -> none installed, using keyword retrieval")
                else:
                    cfg[field] = fallback
                    chosen.append(f"  {field:<12} auto -> nothing suitable installed; "
                                  f"falling back to {fallback} (run installer/install_ai_builder.py)")

            info = cfg.describe()
            if chosen:
                info += "\nAUTO-SELECTED (size is the biggest quality factor, so the largest capable model wins):\n" \
                        + "\n".join(chosen)
            base = {m.split(":")[0] for m in available}
            missing = [m for m in (cfg["model"], cfg["vision_model"])
                       if m and m not in available and m.split(":")[0] not in base]
            info += "\navailable: " + ", ".join(available[:30])
            if missing:
                info += ("\nMISSING (run 'ollama pull <name>' or installer/install_ai_builder.py): "
                         + ", ".join(missing))
            info += "\n" + self._quality_advice(cfg, details)
        else:
            info = cfg.describe()
        return (dict(cfg), info)

    @staticmethod
    def _quality_advice(cfg, details):
        """Model size drives output quality more than any other setting, so say so plainly.

        A small vision model reports every part as the same size and misses most detail, which turns
        a richly detailed reference into a couple of featureless shapes. Parameter counts come from
        Ollama itself because tags like "qwen3.8:latest" hide a 27B model behind a plain name.
        """
        def size_of(name):
            d = details.get(name)
            if d is None:
                d = next((v for k, v in details.items() if k.split(":")[0] == str(name).split(":")[0]), None)
            return (d or {}).get("params_b")

        notes = []
        v_size, c_size = size_of(cfg.get("vision_model")), size_of(cfg.get("model"))
        vision_models = sorted(((d["params_b"], n) for n, d in details.items()
                                if d.get("params_b") and "vision" in (d.get("capabilities") or [])), reverse=True)
        code_models = sorted(((d["params_b"], n) for n, d in details.items()
                              if d.get("params_b") and ("coder" in n.lower() or "code" in n.lower()
                                                        or "tools" in (d.get("capabilities") or []))), reverse=True)
        if vision_models and (v_size is None or (v_size < 10 and vision_models[0][0] > v_size)):
            best, name = vision_models[0]
            if v_size is None or best > v_size:
                notes.append(f"TIP: vision model '{cfg['vision_model']}'"
                             + (f" is small ({v_size:g}B)." if v_size else " size unknown.")
                             + f" Small vision models report every part as the same size and miss fine detail,"
                               f" which yields featureless geometry. You have '{name}' ({best:g}B) -"
                               f" it produces a far richer build specification.")
        if code_models and c_size is not None and c_size < 14 and code_models[0][0] > c_size:
            best, name = code_models[0]
            notes.append(f"TIP: code model '{cfg['model']}' is small ({c_size:g}B); '{name}' ({best:g}B) "
                         f"needs fewer retries and writes more detailed geometry.")
        if not notes:
            notes.append("Model sizes look good for detailed work.")
        return "\n".join(notes)


class GapAIReferenceAnalyzer:
    """Turn 1-3 reference image batches (+ your text) into one structured build brief."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": (LLM_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What you want built. The brief merges this with what the images show."}),
            },
            "optional": {
                "images": ("IMAGE",),
                "images_2": ("IMAGE",),
                "images_3": ("IMAGE",),
                "notes": ("STRING", {"multiline": True, "default": "", "tooltip": "Hints for the vision model, e.g. 'image 2 is the rear view'."}),
                "session": (SESSION_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("reference_brief", "analyses_json")
    FUNCTION = "analyze"
    CATEGORY = CATEGORY

    def analyze(self, llm, prompt, images=None, images_2=None, images_3=None, notes="", session=None):
        b64s = []
        for batch in (images, images_2, images_3):
            b64s.extend(_tensor_to_b64_list(batch))
        if not b64s:
            brief = prompt.strip()
            return (brief, "[]")
        sess = SceneSession.from_payload(session) if session else SceneSession("_reference_scratch").open()
        _save_refs(sess, b64s)
        agent = SceneBuilderAgent(sess, llm, AgentOptions(render_preview=False))
        brief, analyses = agent.analyze_references(b64s, notes, prompt)
        return (brief, json.dumps(analyses, indent=1))


class GapAIScenePlanner:
    """Decompose a request into ordered build steps (or pass your own plan through)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "llm": (LLM_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": "A small medieval watchtower on a grassy hill surrounded by pine trees, late afternoon sun."}),
                "max_steps": ("INT", {"default": 6, "min": 1, "max": 30}),
            },
            "optional": {
                "reference_brief": ("STRING", {"forceInput": True}),
                "manual_plan": ("STRING", {"multiline": True, "default": "", "tooltip": "Write your own steps, one per line ('Title: instruction'). When filled, the model planner is skipped."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan_json", "plan_text")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def plan(self, session, llm, prompt, max_steps, reference_brief="", manual_plan=""):
        sess = SceneSession.from_payload(session)
        agent = SceneBuilderAgent(sess, llm, AgentOptions(render_preview=False))
        steps = agent.parse_manual_plan(manual_plan)
        if steps:
            sess.set_plan(steps)
        else:
            steps = agent.plan(prompt, reference_brief, max_steps)
        text = agent.plan_to_text(steps)
        print("[AI Scene Builder] Plan:\n" + text)
        return (json.dumps({"steps": steps}, indent=1), text)


class GapAISceneBuilder:
    """Complete-scene builder: executes every step of a plan with validation and retries."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "session": (SESSION_TYPE,),
            "llm": (LLM_TYPE,),
            "plan_json": ("STRING", {"multiline": True, "default": "", "tooltip": "Connect the planner's plan_json, or paste steps (one per line). Empty = build 'prompt' as one step."}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Used only when plan_json is empty."}),
            "start_fresh": ("BOOLEAN", {"default": True,
                            "tooltip": "ON: archive whatever this session already contains and build the plan into an "
                                       "empty scene. A complete build re-run on top of an old one gives two objects "
                                       "superimposed - the 'pile of parts' problem. OFF: add the plan to the existing scene "
                                       "(use the Step Builder for that instead)."}),
            "stop_on_failure": ("BOOLEAN", {"default": True}),
        }
        req.update(_exec_inputs())
        opt = {"reference_brief": ("STRING", {"forceInput": True})}
        opt.update(_exec_optional_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", SESSION_TYPE)
    RETURN_NAMES = ("preview", "blend_path", "report", "scripts", "validation_json", "session")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(self, session, llm, plan_json, prompt, stop_on_failure, reference_brief="", **kwargs):
        sess = SceneSession.from_payload(session)
        start_fresh = bool(kwargs.pop("start_fresh", True))
        prefix = ""
        if start_fresh and (sess.turn_count or sess.blend_exists()):
            previous = sess.state.get("object_identity") or "previous build"
            sess = SceneSession(sess.name, sess.root).open(reset=True)
            prefix = (f"start_fresh: archived the previous scene ('{previous}', {sess.turn_count} steps before "
                      f"reset) to archive_*/ inside the session folder and built into an empty scene.\n\n")
            print("[AI Scene Builder] " + prefix.strip())
        agent = SceneBuilderAgent(sess, llm, _options_from(kwargs))
        steps = agent.parse_manual_plan(plan_json)
        if not steps:
            if not prompt.strip():
                raise ValueError("Give the builder a plan_json (from AI Scene Planner) or a prompt.")
            steps = [{"title": "Build", "category": "edit", "instruction": prompt.strip()}]
        results, report = agent.run_plan(steps, reference_brief, stop_on_failure)
        preview, blend, report, scripts, vjson = _collect_outputs(sess, results, prefix + report)
        return {"ui": {"text": [report[-2000:]]}, "result": (preview, blend, report, scripts, vjson, sess.to_payload())}


class GapAIWholeObjectBuilder:
    """One script for the whole object, rendered, critiqued against the reference, revised, rebuilt.

    This is how a frontier model reproduces a reference: one author writes the complete build file,
    looks at the render next to the picture, edits the file and rebuilds from scratch. Parts relate to
    each other because they were written together; a revision fixes the script instead of piling
    corrections onto a scene. Best with Claude/GPT or a large vision-capable local model; a small
    blind coder still runs but should use the step-by-step Complete Scene builder instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "session": (SESSION_TYPE,),
            "llm": (LLM_TYPE,),
            "prompt": ("STRING", {"multiline": True,
                       "default": "Reproduce the object in the reference images as a detailed 3D model: match its "
                                  "shapes, proportions, materials and all of its decoration.",
                       "tooltip": "What to build. With reference images connected, the brief from the Reference "
                                  "Analyzer carries the measurements; this text says what matters most."}),
            "rounds": ("INT", {"default": 3, "min": 1, "max": 8,
                       "tooltip": "write script -> build -> render -> critique -> revise script -> rebuild. Each round "
                                  "is a full rebuild; the best-scoring round is kept."}),
            "target_score": ("INT", {"default": 85, "min": 0, "max": 100,
                             "tooltip": "Stop once the critic scores the resemblance this high."}),
            "start_fresh": ("BOOLEAN", {"default": True,
                            "tooltip": "Archive whatever the session holds first. Whole-object rounds always rebuild "
                                       "into an empty scene; this only decides whether the OLD session content is "
                                       "archived (ON) or overwritten (OFF)."}),
        }
        req.update(_exec_inputs())
        opt = {
            "images": ("IMAGE",),
            "images_2": ("IMAGE",),
            "reference_brief": ("STRING", {"forceInput": True}),
        }
        opt.update(_exec_optional_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "INT", SESSION_TYPE)
    RETURN_NAMES = ("preview", "blend_path", "report", "script", "critique_json", "final_score", "session")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(self, session, llm, prompt, rounds, target_score, images=None, images_2=None,
              reference_brief="", **kwargs):
        sess = SceneSession.from_payload(session)
        start_fresh = bool(kwargs.pop("start_fresh", True))
        prefix = ""
        if start_fresh and (sess.turn_count or sess.blend_exists()):
            previous = sess.state.get("object_identity") or "previous build"
            sess = SceneSession(sess.name, sess.root).open(reset=True)
            prefix = f"start_fresh: archived the previous session content ('{previous}') to archive_*/.\n\n"
        refs = _tensor_to_b64_list(images) + _tensor_to_b64_list(images_2)
        if refs:
            _save_refs(sess, refs)
        agent = SceneBuilderAgent(sess, llm, _options_from(kwargs))
        r = agent.build_whole(prompt.strip() or "Build the object described in the specification.",
                              reference_brief, rounds=rounds, target_score=target_score,
                              reference_images_b64=refs or None)
        if not refs and not agent.reference_images_b64():
            r["report"] += ("\n\nNOTE: no reference image was available, so there was nothing to critique against - "
                            "one round was built and kept. Connect the reference images to score and revise.")
        preview, blend, report, scripts, _ = _collect_outputs(sess, [r], prefix + r["report"])
        critique_json = json.dumps([{"round": h["round"], "score": h["score"], "critique": h["critique"]}
                                    for h in r.get("history", [])], indent=1)
        return {"ui": {"text": [report[-2000:]]},
                "result": (preview, blend, report, r.get("code", ""), critique_json, int(r.get("score") or 0),
                           sess.to_payload())}


def _identity_warning(sess, reference_brief):
    """Warn when a conversational step's reference describes a different object than the scene holds.

    Building robot legs into a session that contains a rocket is how two objects end up piled
    together. The Step Builder deliberately keeps adding (that is its job), so it warns instead of
    wiping - the user may be composing a scene on purpose.
    """
    stored = (sess.state.get("object_identity") or "").strip().lower()
    incoming = SceneBuilderAgent.spec_object(reference_brief)
    if not stored or not incoming or stored == incoming or not (sess.turn_count or sess.blend_exists()):
        return ""
    common = set(stored.replace("_", " ").split()) & set(incoming.replace("_", " ").split())
    if common - {"steampunk", "vintage", "retro", "ornament", "model", "object", "the", "a"}:
        return ""  # same thing described slightly differently
    return (f"WARNING: this session already contains '{stored}', but the reference brief describes "
            f"'{incoming}'. Building it here will pile the two objects together. Use a new session_name "
            f"or tick 'reset' on the Session node if that is not what you want.\n\n")


class GapAIStepBuilder:
    """Conversational single step: 'now add a stone bridge over the river'. Builds on the session scene."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "session": (SESSION_TYPE,),
            "llm": (LLM_TYPE,),
            "instruction": ("STRING", {"multiline": True, "default": "Add a wooden fence around the tower base with a gate on the south side."}),
            "step_title": ("STRING", {"default": "Edit"}),
        }
        req.update(_exec_inputs())
        opt = {"reference_brief": ("STRING", {"forceInput": True})}
        opt.update(_exec_optional_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", SESSION_TYPE)
    RETURN_NAMES = ("preview", "blend_path", "report", "script", "validation_json", "session")
    FUNCTION = "build_step"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build_step(self, session, llm, instruction, step_title, reference_brief="", **kwargs):
        sess = SceneSession.from_payload(session)
        warning = _identity_warning(sess, reference_brief)
        if warning:
            print("[AI Scene Builder] " + warning.strip())
        agent = SceneBuilderAgent(sess, llm, _options_from(kwargs))
        r = agent.run_step(instruction, step_title or "Edit", reference_brief)
        preview, blend, report, scripts, vjson = _collect_outputs(sess, [r], warning + r["report"])
        return {"ui": {"text": [report[-2000:]]}, "result": (preview, blend, report, scripts, vjson, sess.to_payload())}


class GapAIScriptRunner:
    """Run a script you reviewed/edited (e.g. from a dry run). Same validation, same logging."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "session": (SESSION_TYPE,),
            "script": ("STRING", {"multiline": True, "default": "from gap_helpers import *\n\ncoll = get_or_create_collection('Manual')\nB = Builder()\nB.box((0, 0, 1), (2, 2, 2))\nlog_built(B.build('Test_Cube', coll, make_material('Mat_Test', (0.8, 0.3, 0.2))))\n"}),
            "title": ("STRING", {"default": "Manual script"}),
        }
        req.update(_exec_inputs())
        req.pop("dry_run")
        req.pop("max_retries")
        opt = {"llm": (LLM_TYPE,)}
        opt.update(_exec_optional_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", SESSION_TYPE)
    RETURN_NAMES = ("preview", "blend_path", "report", "validation_json", "session")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, session, script, title, llm=None, **kwargs):
        sess = SceneSession.from_payload(session)
        agent = SceneBuilderAgent(sess, llm or LLMConfig(), _options_from(kwargs))
        r = agent.run_script(script, title or "Manual script")
        preview, blend, report, _, vjson = _collect_outputs(sess, [r], r["report"])
        return {"ui": {"text": [report[-2000:]]}, "result": (preview, blend, report, vjson, sess.to_payload())}


class GapAIVisualRefiner:
    """Render the scene, compare it to the reference images, and fix the differences. Repeat.

    This is what closes the gap to a reference. A single generate-and-hope pass never sees its own
    output; this loop looks at the render next to the reference, lists what is wrong (silhouette,
    missing parts, wrong shapes, placement, materials, detail density) and executes the fixes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "session": (SESSION_TYPE,),
            "llm": (LLM_TYPE,),
            "rounds": ("INT", {"default": 2, "min": 1, "max": 8,
                       "tooltip": "How many render -> critique -> fix cycles to run."}),
            "target_score": ("INT", {"default": 85, "min": 0, "max": 100,
                             "tooltip": "Stop early once the critic scores the resemblance this high."}),
            "max_fix_steps": ("INT", {"default": 3, "min": 1, "max": 8,
                              "tooltip": "Maximum correction steps per round."}),
        }
        req.update(_exec_inputs())
        req.pop("dry_run")
        opt = {
            "images": ("IMAGE",),
            "images_2": ("IMAGE",),
            "reference_brief": ("STRING", {"forceInput": True}),
        }
        opt.update(_exec_optional_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", SESSION_TYPE)
    RETURN_NAMES = ("preview", "report", "critique_json", "blend_path", "final_score", "session")
    FUNCTION = "refine"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def refine(self, session, llm, rounds, target_score, max_fix_steps,
               images=None, images_2=None, reference_brief="", **kwargs):
        sess = SceneSession.from_payload(session)
        refs = _tensor_to_b64_list(images) + _tensor_to_b64_list(images_2)
        if not refs:
            report = ("AI Visual Refiner needs at least one reference image to compare against.\n"
                      "Connect the same LoadImage you gave the Reference Analyzer.")
            return {"ui": {"text": [report]}, "result": (_blank_image(), report, "{}", sess.blend_path, 0, sess.to_payload())}
        if not sess.blend_exists():
            report = f"Session '{sess.name}' has no scene yet - build something before refining it."
            return {"ui": {"text": [report]}, "result": (_blank_image(), report, "{}", sess.blend_path, 0, sess.to_payload())}
        agent = SceneBuilderAgent(sess, llm, _options_from(kwargs))
        history, render_path = agent.refine(refs, reference_brief, rounds=rounds,
                                            target_score=target_score, max_fix_steps=max_fix_steps)
        lines = [f"AI Visual Refiner - session '{sess.name}'", f"blend: {sess.blend_path}", ""]
        for h in history:
            c = h["critique"]
            lines.append(f"## Round {h['round']}: score {h['score']}/100 - {c.get('verdict', '')}")
            for d in c.get("differences", [])[:8]:
                lines.append(f"  [{d.get('importance', '?'):<8}] {d.get('category', '')}: {d.get('issue', '')}")
                lines.append(f"             fix: {d.get('fix', '')}")
            if h["steps"]:
                lines.append("  applied: " + "; ".join(s["title"] for s in h["steps"]))
            if h.get("report"):
                lines.append("  " + h["report"].replace("\n", "\n  ")[:1500])
            lines.append("")
        scores = [h["score"] for h in history]
        final = scores[-1] if scores else 0
        if len(scores) > 1:
            lines.append(f"score progression: {' -> '.join(str(s) for s in scores)}")
        preview = _load_image_batch([render_path] if render_path else [])
        report = "\n".join(lines)
        critique_json = json.dumps([h["critique"] for h in history], indent=1)
        return {"ui": {"text": [report[-2000:]]},
                "result": (preview, report, critique_json, sess.blend_path, int(final), sess.to_payload())}


class GapAIBlenderBridgeCheck:
    """Verify a running Blender is reachable and the toolbox bridge actually works.

    A socket connect only proves something holds the port. This does a full round trip: the addon
    writes back its version, the open file, the scene contents and whether live AI execution is
    allowed - so you know what will and will not work before you queue a long build.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8119, "min": 1024, "max": 65535}),
                "timeout_s": ("INT", {"default": 10, "min": 2, "max": 120,
                              "tooltip": "How long to wait for Blender's reply. Raise it if Blender is busy."}),
                "raise_on_failure": ("BOOLEAN", {"default": False,
                                     "tooltip": "ON: stop the workflow when the bridge is down. "
                                                "OFF: report the problem and carry on."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("connected", "live_exec_allowed", "status", "status_json")
    FUNCTION = "check"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check(self, host, port, timeout_s, raise_on_failure):
        status = bridge.ping(host.strip() or "127.0.0.1", port, timeout=timeout_s)
        text = bridge.describe(status)
        print("[AI Scene Builder] " + text.replace("\n", "\n[AI Scene Builder] "))
        if raise_on_failure and not status.get("ok"):
            raise RuntimeError(text)
        return {"ui": {"text": [text]},
                "result": (bool(status.get("ok")), bool(status.get("ai_exec_allowed")),
                           text, json.dumps(status, indent=1))}


class GapAISendSceneToBlender:
    """Load a finished scene into the Blender you have open, with materials and collections intact.

    Appending the .blend keeps everything the headless build made: collections, material node
    trees, lights and cameras. Exporting through GLB would flatten most of that, so this is the
    way to get a completed build onto your desk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["append", "link", "open"], {"default": "append",
                         "tooltip": "append: copy the objects into your current scene (default, safe). "
                                    "link: reference them read-only from the .blend. "
                                    "open: REPLACE your open file with the built scene - unsaved work is lost."}),
                "host": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8119, "min": 1024, "max": 65535}),
                "frame_viewport": ("BOOLEAN", {"default": True,
                                   "tooltip": "Select and zoom the viewport onto what was just imported."}),
                "clear_scene_first": ("BOOLEAN", {"default": False,
                                      "tooltip": "Delete every object in the open Blender scene before importing."}),
                "timeout_s": ("INT", {"default": 180, "min": 5, "max": 3600}),
            },
            "optional": {
                "session": (SESSION_TYPE,),
                "blend_path": ("STRING", {"default": "",
                               "tooltip": "Explicit .blend to send. Leave empty to use the session's scene.blend."}),
                "collections": ("STRING", {"default": "",
                                "tooltip": "Comma-separated collection names to import. Empty = all of them."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("sent", "report", "blend_path")
    FUNCTION = "send"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def send(self, mode, host, port, frame_viewport, clear_scene_first, timeout_s,
             session=None, blend_path="", collections=""):
        path = (blend_path or "").strip()
        if not path and session is not None:
            path = SceneSession.from_payload(session).blend_path
        if not path:
            report = "Give this node a session or an explicit blend_path."
            return {"ui": {"text": [report]}, "result": (False, report, "")}
        if not os.path.isfile(path):
            report = (f"No .blend at {path}\nBuild the scene first - a headless build writes "
                      f"scene.blend into the session folder.")
            return {"ui": {"text": [report]}, "result": (False, report, path)}

        status = bridge.ping(host.strip() or "127.0.0.1", port, timeout=min(timeout_s, 15))
        if not status.get("ok"):
            report = bridge.describe(status)
            print("[AI Scene Builder] " + report.replace("\n", "\n[AI Scene Builder] "))
            return {"ui": {"text": [report]}, "result": (False, report, path)}

        names = [c.strip() for c in collections.split(",") if c.strip()]
        if mode == "open":
            print("[AI Scene Builder] mode 'open' REPLACES the file open in Blender; unsaved work is lost.")
        result = bridge.append_blend(path, host.strip() or "127.0.0.1", port, mode=mode,
                                     collections=names, clear_scene=clear_scene_first,
                                     frame=frame_viewport, timeout=timeout_s)
        if result.get("ok"):
            report = (f"Sent to Blender {status.get('blender_version')} ({mode})\n"
                      f"  from : {path}\n"
                      f"  objects    : {len(result.get('appended_objects', []))}"
                      f" ({', '.join(result.get('appended_objects', [])[:8])}"
                      f"{' ...' if len(result.get('appended_objects', [])) > 8 else ''})\n"
                      f"  collections: {', '.join(result.get('appended_collections', [])) or '(none)'}")
            if mode == "append":
                report += "\n  Materials, lights and cameras came across with the objects."
        else:
            report = f"Import failed: {result.get('error')}\n  blend: {path}"
        print("[AI Scene Builder] " + report.replace("\n", "\n[AI Scene Builder] "))
        return {"ui": {"text": [report]}, "result": (bool(result.get("ok")), report, path)}


class GapAIDebugLog:
    """Read the full debug log for a session: every prompt, every raw reply, every generated
    script, Blender's stdout, validation totals and every error, in order.

    When a build comes out wrong this tells you which stage went wrong and what it actually saw -
    the brief, the plan, or the code. The file is <session>/debug.log, so you can also open it in
    an editor or paste it somewhere.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "show": (["everything", "errors and warnings only", "prompts and replies only",
                          "last step only"], {"default": "everything"}),
                "max_chars": ("INT", {"default": 60000, "min": 1000, "max": 1000000, "step": 1000,
                              "tooltip": "Tail length. The whole file is always on disk regardless."}),
            },
            "optional": {
                "clear_log": ("BOOLEAN", {"default": False,
                              "tooltip": "Delete the log before reading, to start a clean record. "
                                         "Turn it off again after one run."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("log", "log_path")
    FUNCTION = "read"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def read(self, session, show, max_chars, clear_log=False):
        sess = SceneSession.from_payload(session)
        if clear_log:
            try:
                if os.path.exists(sess.debug_path):
                    os.remove(sess.debug_path)
            except OSError as e:
                print(f"[AI Scene Builder] could not clear the debug log: {e}")
        text = sess.read_debug(max_chars=1000000)
        if show == "errors and warnings only":
            keep = ("fail", "error", "warning", "traceback", "refused", "blocked", "rejected",
                    "missing", "could not", "timed out")
            text = "\n".join(l for l in text.splitlines() if any(k in l.lower() for k in keep))
        elif show == "prompts and replies only":
            blocks, current, keeping = [], [], False
            for line in text.splitlines():
                marker = ("PROMPT SENT" in line or "RAW MODEL REPLY" in line
                          or "BUILD SPECIFICATION" in line or line.startswith("=" * 20))
                if marker:
                    if keeping and current:
                        blocks.append("\n".join(current))
                    current, keeping = [line], not line.startswith("=" * 20)
                elif keeping:
                    current.append(line)
            if keeping and current:
                blocks.append("\n".join(current))
            text = "\n".join(blocks)
        elif show == "last step only":
            marker = text.rfind("STEP ")
            if marker > 0:
                start = text.rfind("=" * 20, 0, marker)
                text = text[start if start > 0 else marker:]
        if len(text) > max_chars:
            text = f"... showing the last {max_chars} characters of {len(text)} ...\n" + text[-max_chars:]
        header = (f"DEBUG LOG - session '{sess.name}'\nfile: {sess.debug_path}\n"
                  f"filter: {show}\n{'-' * 70}\n")
        return {"ui": {"text": [header + text]}, "result": (header + text, sess.debug_path)}


class GapAISceneValidator:
    """Validate (and optionally auto-fix) the session scene: polygons, normals, textures, names."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "execution_mode": (["headless", "live"], {"default": "headless"}),
                "auto_fix_normals": ("BOOLEAN", {"default": False}),
                "auto_fix_doubles": ("BOOLEAN", {"default": False}),
                "render_preview": ("BOOLEAN", {"default": True}),
                "render_engine": (["CYCLES", "BLENDER_EEVEE"], {"default": config.DEFAULT_RENDER_ENGINE}),
                "render_width": ("INT", {"default": config.DEFAULT_RENDER_WIDTH, "min": 64, "max": 4096, "step": 8}),
                "render_height": ("INT", {"default": config.DEFAULT_RENDER_HEIGHT, "min": 64, "max": 4096, "step": 8}),
                "render_samples": ("INT", {"default": config.DEFAULT_RENDER_SAMPLES, "min": 1, "max": 1024}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": ""}),
                "live_host": ("STRING", {"default": "127.0.0.1"}),
                "live_port": ("INT", {"default": 8119, "min": 1024, "max": 65535}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("preview", "report", "validation_json", "passed")
    FUNCTION = "validate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def validate(self, session, execution_mode, auto_fix_normals, auto_fix_doubles, render_preview, render_engine,
                 render_width, render_height, render_samples, blender_path="", live_host="127.0.0.1", live_port=8119):
        sess = SceneSession.from_payload(session)
        opts = AgentOptions(execution_mode=execution_mode, render_preview=render_preview, render_engine=render_engine,
                            render_width=render_width, render_height=render_height, render_samples=render_samples,
                            blender_path=blender_path, live_host=live_host, live_port=live_port)
        agent = SceneBuilderAgent(sess, LLMConfig(), opts)
        if execution_mode == "headless" and not sess.blend_exists():
            report = f"Session '{sess.name}' has no scene yet ({sess.blend_path})."
            return {"ui": {"text": [report]}, "result": (_blank_image(), report, "{}", False)}
        result, summary = agent.validate_only(auto_fix_normals, auto_fix_doubles, render_preview)
        v = result.get("validation") or {}
        preview = _load_image_batch(_collect_render_paths(result))
        report = f"Validation of {sess.blend_path}\n{summary}"
        if result.get("error") and not v:
            report += f"\nerror: {result['error']}"
        return {"ui": {"text": [report[-2000:]]}, "result": (preview, report, json.dumps(v, indent=1), bool(v.get("passed")))}


NODE_CLASS_MAPPINGS = {
    "GapAISceneSession": GapAISceneSession,
    "GapAILLMConfig": GapAILLMConfig,
    "GapAIReferenceAnalyzer": GapAIReferenceAnalyzer,
    "GapAIScenePlanner": GapAIScenePlanner,
    "GapAISceneBuilder": GapAISceneBuilder,
    "GapAIWholeObjectBuilder": GapAIWholeObjectBuilder,
    "GapAIStepBuilder": GapAIStepBuilder,
    "GapAIScriptRunner": GapAIScriptRunner,
    "GapAIVisualRefiner": GapAIVisualRefiner,
    "GapAIBlenderBridgeCheck": GapAIBlenderBridgeCheck,
    "GapAISendSceneToBlender": GapAISendSceneToBlender,
    "GapAIDebugLog": GapAIDebugLog,
    "GapAISceneValidator": GapAISceneValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GapAISceneSession": "AI Scene Session (Geekatplay)",
    "GapAILLMConfig": "AI Model Config (Geekatplay)",
    "GapAIReferenceAnalyzer": "AI Reference Analyzer (Multi-Image)",
    "GapAIScenePlanner": "AI Scene Planner",
    "GapAISceneBuilder": "AI Scene Builder (Complete Scene)",
    "GapAIWholeObjectBuilder": "AI Whole-Object Builder (One Script, See & Revise)",
    "GapAIStepBuilder": "AI Step Builder (Conversational)",
    "GapAIScriptRunner": "AI Script Runner (Review & Execute)",
    "GapAIVisualRefiner": "AI Visual Refiner (Compare to Reference & Fix)",
    "GapAIBlenderBridgeCheck": "Blender Bridge Check (Is Blender Open?)",
    "GapAISendSceneToBlender": "Send Scene to Blender (Append/Link/Open)",
    "GapAIDebugLog": "AI Debug Log (Full Step-by-Step Trace)",
    "GapAISceneValidator": "AI Scene Validator (Polygons/Normals/Textures/Names)",
}
