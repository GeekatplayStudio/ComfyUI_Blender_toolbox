# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Session state on disk.

A session is one scene being built over many ComfyUI runs. ComfyUI itself is stateless
between "Queue Prompt" clicks, so everything that must survive lives here:

  <sessions_root>/<name>/
      session.json      turns, plan, last scene probe, last validation, reference brief
      scene.blend       the scene itself - the single source of truth for multi-pass edits
      scripts/          every generated script, numbered: step_003_attempt1.py
      results/          the JSON result for every executed script
      renders/          preview renders
      refs/             copies of reference images used for analysis
"""

import json
import os
import re
import shutil
import time

from . import config


def _slug(name):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip()).strip("._")
    return slug[:80] or "scene"


class SceneSession:
    VERSION = 1

    def __init__(self, name, root=None):
        self.name = _slug(name)
        self.root = root or os.path.join(config.get_sessions_root(), self.name)
        self.scripts_dir = os.path.join(self.root, "scripts")
        self.results_dir = os.path.join(self.root, "results")
        self.renders_dir = os.path.join(self.root, "renders")
        self.refs_dir = os.path.join(self.root, "refs")
        self.state_path = os.path.join(self.root, "session.json")
        self.state = None

    # ---------------------------------------------------------------- lifecycle
    def open(self, reset=False, blend_path=""):
        for d in (self.root, self.scripts_dir, self.results_dir, self.renders_dir, self.refs_dir):
            os.makedirs(d, exist_ok=True)
        if reset and os.path.exists(self.state_path):
            self._archive_previous()
        if os.path.exists(self.state_path) and not reset:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = self._new_state()
        if blend_path:
            self.state["blend_path"] = blend_path
        elif not self.state.get("blend_path"):
            self.state["blend_path"] = os.path.join(self.root, "scene.blend")
        self.save()
        return self

    def _new_state(self):
        return {
            "version": self.VERSION,
            "name": self.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated": "",
            "blend_path": "",
            "reference_brief": "",
            "plan": [],
            "turns": [],
            "last_scene": {},
            "last_validation": {},
            "notes": "",
        }

    def _archive_previous(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = os.path.join(self.root, f"archive_{stamp}")
        os.makedirs(archive, exist_ok=True)
        for item in ("session.json", "scene.blend", "scripts", "results", "renders"):
            src = os.path.join(self.root, item)
            if os.path.exists(src):
                shutil.move(src, os.path.join(archive, item))
        for d in (self.scripts_dir, self.results_dir, self.renders_dir):
            os.makedirs(d, exist_ok=True)

    def save(self):
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)

    # ---------------------------------------------------------------- accessors
    @property
    def blend_path(self):
        return self.state["blend_path"]

    def blend_exists(self):
        return os.path.isfile(self.blend_path) and os.path.getsize(self.blend_path) > 0

    @property
    def turn_count(self):
        return len(self.state["turns"])

    def next_script_path(self, attempt=1, label="step"):
        idx = self.turn_count + 1
        return os.path.join(self.scripts_dir, f"{label}_{idx:03d}_attempt{attempt}.py")

    def result_path_for(self, script_path):
        base = os.path.splitext(os.path.basename(script_path))[0]
        return os.path.join(self.results_dir, base + ".json")

    def render_path_for(self, script_path):
        base = os.path.splitext(os.path.basename(script_path))[0]
        return os.path.join(self.renders_dir, base + ".png")

    def add_turn(self, turn):
        turn = dict(turn)
        turn.setdefault("index", self.turn_count + 1)
        turn.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.state["turns"].append(turn)
        if turn.get("scene"):
            self.state["last_scene"] = turn["scene"]
        if turn.get("validation"):
            self.state["last_validation"] = turn["validation"]
        self.save()
        return turn

    def set_plan(self, plan):
        self.state["plan"] = plan or []
        self.save()

    def set_reference_brief(self, brief):
        self.state["reference_brief"] = brief or ""
        self.save()

    def copy_reference_image(self, src_path):
        os.makedirs(self.refs_dir, exist_ok=True)
        dst = os.path.join(self.refs_dir, os.path.basename(src_path))
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)
        return dst

    # ---------------------------------------------------------------- summaries
    def history_summary(self, max_turns=12):
        """Compact text of previous turns for the model's context."""
        turns = self.state["turns"][-max_turns:]
        if not turns:
            return "No previous steps. The scene is empty."
        lines = []
        for t in turns:
            status = "OK" if t.get("ok") else "FAILED"
            summary = t.get("summary") or t.get("instruction", "")[:160]
            lines.append(f"- Step {t.get('index')}: [{status}] {summary}")
        return "\n".join(lines)

    def scene_summary(self, max_objects=60):
        """Readable description of the last scene probe."""
        scene = self.state.get("last_scene") or {}
        if not scene:
            return "Scene is empty (no objects yet)."
        parts = [
            f"Blender {scene.get('blender_version', '?')} | objects: {scene.get('object_count', 0)} | "
            f"collections: {', '.join(scene.get('collections', [])[:20]) or 'none'}"
        ]
        bbox = scene.get("bbox")
        if bbox:
            parts.append(f"Scene bounds min={bbox.get('min')} max={bbox.get('max')} (meters)")
        objs = scene.get("objects", [])[:max_objects]
        for o in objs:
            parts.append(
                f"  {o.get('type', '?'):<6} {o.get('name')}  loc={o.get('location')} dims={o.get('dimensions')} "
                f"mats={','.join(o.get('materials', [])) or '-'} coll={o.get('collection', '-')}"
            )
        if scene.get("object_count", 0) > len(objs):
            parts.append(f"  ... {scene['object_count'] - len(objs)} more objects")
        if scene.get("materials"):
            parts.append("Materials: " + ", ".join(scene["materials"][:40]))
        if scene.get("lights"):
            parts.append("Lights: " + ", ".join(f"{l['name']}({l['type']})" for l in scene["lights"][:20]))
        if scene.get("cameras"):
            parts.append("Cameras: " + ", ".join(scene["cameras"][:10]))
        if scene.get("world"):
            parts.append(f"World: {scene['world']}")
        return "\n".join(parts)

    def info_text(self):
        return (
            f"Session '{self.name}'\n"
            f"  folder: {self.root}\n"
            f"  blend : {self.blend_path} ({'exists' if self.blend_exists() else 'not created yet'})\n"
            f"  turns : {self.turn_count}\n"
            f"  plan  : {len(self.state.get('plan') or [])} steps\n"
            f"  refs  : {'yes' if self.state.get('reference_brief') else 'none'}"
        )

    # ---------------------------------------------------------------- transport
    def to_payload(self):
        return {"name": self.name, "root": self.root}

    @classmethod
    def from_payload(cls, payload):
        if isinstance(payload, SceneSession):
            return payload
        if not payload or "name" not in payload:
            raise ValueError("Invalid session payload - connect an 'AI Scene Session' node.")
        return cls(payload["name"], payload.get("root")).open()
