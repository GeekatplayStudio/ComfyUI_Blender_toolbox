# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder installer
"""
Prepares everything the AI Scene Builder needs and reports what is missing:

  1. Blender executable (BLENDER_PATH env var or auto-detected)
  2. Ollama running locally (starts it when installed, offers winget install on Windows)
  3. Ollama models: code, vision, embedding (pulled when missing)
  4. Session root folder and the reference-doc retrieval cache
  5. Optional smoke test: run a tiny helper script through Blender and validate it

Usage (from the toolbox folder, with ComfyUI's python):
  python installer/install_ai_builder.py
  python installer/install_ai_builder.py --code-model qwen2.5-coder:32b --vision-model qwen3-vl:4b
  python installer/install_ai_builder.py --skip-models --smoke-test
Nothing here is silent: every command that runs is printed first.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

from ai_builder import config  # noqa: E402  (nodes/ai_builder)
from utils_blender import get_blender_path  # noqa: E402


def say(msg):
    print(f"[AI Builder Setup] {msg}", flush=True)


def run(cmd, check=False, capture=False):
    say("running: " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, capture_output=capture, text=True)


def ollama_alive(url):
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def ensure_ollama(url):
    models = ollama_alive(url)
    if models is not None:
        say(f"Ollama is running at {url} with {len(models)} models.")
        return models
    have_cli = subprocess.run("ollama --version", shell=True, capture_output=True).returncode == 0
    if not have_cli:
        say("Ollama is not installed.")
        if sys.platform == "win32":
            say("Installing with winget (approve the prompt if one appears)...")
            run("winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements")
            have_cli = subprocess.run("ollama --version", shell=True, capture_output=True).returncode == 0
        if not have_cli:
            say("Please install Ollama from https://ollama.com/download and re-run this installer.")
            return None
    say("Starting the Ollama service...")
    if sys.platform == "win32":
        subprocess.Popen("ollama serve", shell=True, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    else:
        subprocess.Popen(["ollama", "serve"])
    for _ in range(20):
        time.sleep(1)
        models = ollama_alive(url)
        if models is not None:
            say("Ollama service is up.")
            return models
    say("Ollama did not answer in time; start it manually (ollama serve) and re-run.")
    return None


def pull_models(wanted, available):
    for name in wanted:
        if not name:
            continue
        base = name.split(":")[0]
        if name in available or (":" not in name and any(a.split(":")[0] == base for a in available)):
            say(f"model present: {name}")
            continue
        say(f"pulling model {name} (this can take a while)...")
        r = run(["ollama", "pull", name])
        if r.returncode != 0:
            say(f"WARNING: could not pull {name}. Pick another model in the 'AI Model Config' node.")


def smoke_test(blender):
    say("Smoke test: building a cube with gap_helpers in headless Blender and validating it...")
    from ai_builder.session import SceneSession
    from ai_builder.blender_runner import BlenderRunner
    session = SceneSession("_installer_smoke_test", root=os.path.join(config.get_sessions_root(), "_installer_smoke_test")).open(reset=True)
    script = session.next_script_path(1, label="smoke")
    with open(script, "w", encoding="utf-8") as f:
        f.write("from gap_helpers import *\ncoll = get_or_create_collection('Smoke')\nB = Builder()\nB.box((0,0,1),(2,2,2))\n"
                "log_built(B.build('Smoke_Cube', coll, make_material('Mat_Smoke', (0.8,0.4,0.2))))\nadd_sun(45,135,3)\n"
                "frame_camera_to_scene(add_camera())\n")
    runner = BlenderRunner(session, blender_path=blender, timeout=300)
    result = runner.run_headless(script, validate=True, render={"path": session.render_path_for(script), "engine": "CYCLES",
                                                                "width": 256, "height": 160, "samples": 4})
    ok = result.get("ok") and (result.get("validation") or {}).get("passed")
    say(f"smoke test {'PASSED' if ok else 'FAILED'}: {result.get('error') or 'cube built, validated and rendered'}")
    if not ok:
        say((result.get("traceback") or result.get("blender_output") or "")[-1500:])
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="Set up the AI Scene Builder")
    ap.add_argument("--ollama-url", default=config.DEFAULT_OLLAMA_URL)
    ap.add_argument("--code-model", default=config.FALLBACK_CODE_MODEL)
    ap.add_argument("--vision-model", default=config.FALLBACK_VISION_MODEL)
    ap.add_argument("--embed-model", default=config.FALLBACK_EMBED_MODEL)
    ap.add_argument("--skip-models", action="store_true", help="do not pull Ollama models")
    ap.add_argument("--smoke-test", action="store_true", help="run a tiny Blender build + validation")
    args = ap.parse_args()

    print("=" * 70)
    print(" ComfyUI-Blender-Toolbox - AI Scene Builder setup (Geekatplay Studio - Vladimir Chopine)")
    print("=" * 70)
    print(config.SANDBOX_WARNING)
    print("-" * 70)

    ok_all = True
    blender = get_blender_path()
    if blender:
        say(f"Blender: {blender}")
    else:
        ok_all = False
        say("Blender NOT found. Install Blender 4.x/5.x or set BLENDER_PATH=<path to blender executable>.")

    if not args.skip_models:
        available = ensure_ollama(args.ollama_url)
        if available is None:
            ok_all = False
        else:
            pull_models([args.code_model, args.vision_model, args.embed_model], available)
    else:
        say("skipping Ollama model checks (--skip-models)")

    root = config.get_sessions_root()
    say(f"session root: {root}")

    try:
        from ai_builder.rag import ReferenceIndex
        idx = ReferenceIndex()
        say(f"reference docs indexed: {len(idx.chunks)} chunks from {config.REFERENCE_DOCS_DIR}")
    except Exception as e:
        say(f"WARNING: could not index reference docs: {e}")

    if args.smoke_test and blender:
        ok_all = smoke_test(blender) and ok_all

    settings = {"ollama_url": args.ollama_url, "code_model": args.code_model, "vision_model": args.vision_model,
                "embed_model": args.embed_model, "blender": blender, "sessions_root": root,
                "checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    settings_path = os.path.join(root, "installer_report.json")
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    say(f"report written to {settings_path}")
    print("-" * 70)
    say("DONE" if ok_all else "DONE WITH WARNINGS - read the lines above")
    say("Next: restart ComfyUI and load workflows/Geekatplay_AI_Scene_Builder_Complete.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
