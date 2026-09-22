# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
step_runner - executes ONE generated script inside headless Blender and reports back as JSON.

    blender --background --python step_runner.py -- <job.json>

job.json (written by nodes/ai_builder/blender_runner.py):
{
  "blend_in": "path or empty (empty = start from an empty scene)",
  "blend_out": "where to save the scene afterwards",
  "code_path": "the generated script to execute",
  "result_path": "where to write this JSON result",
  "helpers_dir": "folder containing gap_helpers.py / scene_probe.py / validate_scene.py / job_executor.py",
  "session_dir": "session folder (exposed as env GAP_SESSION_DIR)",
  "validate": true, "auto_fix_normals": false, "auto_fix_doubles": false, "probe": true,
  "render": {"path": "...png", "engine": "CYCLES|BLENDER_EEVEE", "width": 768, "height": 512,
             "samples": 16, "fit_camera": true}   (or null)
}

The runner ALWAYS writes the result JSON (even on crash) and exits 0 so the caller can read it.
"""

import json
import os
import sys
import traceback

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_job():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("step_runner: missing job.json argument after '--'")
    with open(argv[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _open_scene(blend_in):
    if blend_in and os.path.isfile(blend_in) and os.path.getsize(blend_in) > 0:
        bpy.ops.wm.open_mainfile(filepath=blend_in, load_ui=False)
        return "opened"
    bpy.ops.wm.read_homefile(use_empty=True)
    return "new"


def main():
    job = _load_job()
    helpers_dir = job.get("helpers_dir") or _HERE
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    result_path = job["result_path"]
    try:
        source = _open_scene(job.get("blend_in", ""))
        import job_executor  # noqa: E402
        result = job_executor.execute_job(job, save=job.get("save", True), scene_source=source)
    except Exception as e:
        result = {"ok": False, "error": f"runner: {type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                  "blender_version": bpy.app.version_string, "stdout": "", "built": [], "validation": None,
                  "scene": None, "render_path": None, "timings": {}, "saved_blend": None}
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp, result_path)
    print(f"STEP_RUNNER_DONE ok={result.get('ok')} result={result_path}", flush=True)


if __name__ == "__main__":
    main()
