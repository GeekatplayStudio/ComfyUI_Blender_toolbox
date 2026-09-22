# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Executes a generated script in Blender and returns the result JSON.

Two modes:
  headless  `blender --background --python step_runner.py -- job.json`  (default; nothing else needed)
  live      sends "AI_EXEC:<job.json path>" to the toolbox addon's socket listener in a running
            Blender, then waits for the addon to write the result JSON. The addon refuses unless
            'Allow AI code execution' is enabled in its sidebar panel.
"""

import json
import os
import socket
import subprocess
import time

from . import config

try:
    from ..utils_blender import get_blender_path
except Exception:  # standalone / tests
    from utils_blender import get_blender_path  # type: ignore


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _failed(error, extra=None):
    result = {"ok": False, "error": error, "traceback": "", "stdout": "", "built": [], "validation": None,
              "scene": None, "render_path": None, "timings": {}, "saved_blend": None}
    if extra:
        result.update(extra)
    return result


class BlenderRunner:
    def __init__(self, session, blender_path="", timeout=config.DEFAULT_BLENDER_TIMEOUT,
                 live_host="127.0.0.1", live_port=8119, live_timeout=config.DEFAULT_LIVE_TIMEOUT):
        self.session = session
        self.blender_path = blender_path or get_blender_path()
        self.timeout = int(timeout)
        self.live_host = live_host
        self.live_port = int(live_port)
        self.live_timeout = int(live_timeout)

    # ------------------------------------------------------------------ job
    def make_job(self, code_path, validate=True, auto_fix_normals=False, auto_fix_doubles=False,
                 render=None, save=True, mode="headless", probe=True):
        result_path = self.session.result_path_for(code_path)
        job = {
            "mode": mode,
            "blend_in": self.session.blend_path if self.session.blend_exists() else "",
            "blend_out": self.session.blend_path,
            "code_path": code_path,
            "result_path": result_path,
            "helpers_dir": config.BLENDER_SCRIPTS_DIR,
            "session_dir": self.session.root,
            "validate": bool(validate),
            "auto_fix_normals": bool(auto_fix_normals),
            "auto_fix_doubles": bool(auto_fix_doubles),
            "probe": bool(probe),
            "save": bool(save),
            "render": render,
        }
        job_path = os.path.splitext(result_path)[0] + ".job.json"
        os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        return job, job_path

    # ------------------------------------------------------------------ headless
    def run_headless(self, code_path, **job_kwargs):
        if not self.blender_path or not os.path.exists(self.blender_path):
            return _failed("Blender executable not found. Install Blender or set the BLENDER_PATH environment variable.")
        job, job_path = self.make_job(code_path, mode="headless", **job_kwargs)
        if os.path.exists(job["result_path"]):
            os.remove(job["result_path"])
        cmd = [self.blender_path, "--background", "--factory-startup", "--python", config.STEP_RUNNER_SCRIPT, "--", job_path]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return _failed(f"Blender step timed out after {self.timeout}s (increase blender_timeout or simplify the step).",
                           {"command": cmd})
        except Exception as e:
            return _failed(f"Could not launch Blender: {e}", {"command": cmd})
        blender_out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if not os.path.exists(job["result_path"]):
            return _failed(f"Blender exited (code {proc.returncode}) without writing a result. Tail of output:\n{blender_out[-3000:]}",
                           {"command": cmd, "blender_output": blender_out[-8000:]})
        result = _read_json(job["result_path"])
        result["blender_output"] = blender_out[-8000:]
        result["command"] = cmd
        result["timings"]["subprocess_s"] = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------ live (addon)
    def run_live(self, code_path, **job_kwargs):
        job_kwargs.setdefault("save", False)  # the user's open Blender file is theirs; save is opt-in
        job, job_path = self.make_job(code_path, mode="live", **job_kwargs)
        if os.path.exists(job["result_path"]):
            os.remove(job["result_path"])
        message = f"AI_EXEC:{job_path}"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((self.live_host, self.live_port))
                s.sendall(message.encode("utf-8"))
        except Exception as e:
            return _failed(f"Could not reach the Blender addon listener at {self.live_host}:{self.live_port} ({e}). "
                           "Start Blender, enable the ComfyUI Blender Toolbox addon and click 'Start Listener'.")
        t0 = time.time()
        while time.time() - t0 < self.live_timeout:
            if os.path.exists(job["result_path"]):
                time.sleep(0.2)
                try:
                    result = _read_json(job["result_path"])
                    result["timings"]["wait_s"] = round(time.time() - t0, 2)
                    return result
                except json.JSONDecodeError:
                    time.sleep(0.3)
                    continue
            time.sleep(0.5)
        return _failed(f"Timed out after {self.live_timeout}s waiting for the Blender addon to finish the step. "
                       "Check the Blender console; if 'Allow AI code execution' is off the addon refuses and logs why.")

    def run(self, code_path, mode="headless", **job_kwargs):
        if mode == "live":
            return self.run_live(code_path, **job_kwargs)
        return self.run_headless(code_path, **job_kwargs)
