# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Talking to a RUNNING Blender.

The addon's socket listener is fire-and-forget: ComfyUI sends a string, Blender acts on it, and
nothing comes back down the socket. To get an answer we ask the addon to write a small JSON file
and poll for it. That is what makes a real health check possible - "the socket accepted my bytes"
only proves something is listening on the port, not that the toolbox addon is on the other end.

Commands used here (handled in blender_toolbox_addon.py):
    PING:<result json path>
    BLEND_APPEND:<blend path>|MODE:append|RESULT:<json>|COLLECTIONS:A,B|CLEAR:false|FRAME:true
"""

import json
import os
import re
import socket
import tempfile
import time

from . import config


def send(host, port, message, timeout=5):
    """Send one message to the addon listener. Returns (ok, error_text)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, int(port)))
            s.sendall(message.encode("utf-8"))
        return True, ""
    except ConnectionRefusedError:
        return False, (f"Nothing is listening on {host}:{port}. Start Blender, enable the "
                       f"'ComfyUI Blender Toolbox Sync' addon, then press N in the 3D viewport > "
                       f"ComfyUI tab > Start Listener.")
    except socket.timeout:
        return False, f"Timed out connecting to {host}:{port}."
    except OSError as e:
        return False, f"Socket error talking to {host}:{port}: {e}"


def _reply_path(prefix):
    folder = os.path.join(tempfile.gettempdir(), "gap_blender_bridge")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{prefix}_{int(time.time() * 1000)}.json")


def _await_reply(path, timeout, poll=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            time.sleep(0.1)  # let the writer finish
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                time.sleep(0.2)
                continue
        time.sleep(poll)
    return None


def ping(host="127.0.0.1", port=8119, timeout=10):
    """Full round-trip health check. Returns a status dict that always has ok/error."""
    reply = _reply_path("ping")
    ok, error = send(host, port, f"PING:{reply}", timeout=5)
    if not ok:
        return {"ok": False, "stage": "connect", "error": error}
    data = _await_reply(reply, timeout)
    try:
        os.remove(reply)
    except OSError:
        pass
    if data is None:
        return {"ok": False, "stage": "reply",
                "error": (f"Something is listening on {host}:{port} but it did not answer the PING "
                          f"within {timeout}s. That usually means the Blender addon is an older "
                          f"version without PING support - reinstall blender_scripts/"
                          f"blender_toolbox_addon.py (v2.2.1+) and restart the listener. "
                          f"It can also mean Blender is busy rendering or blocked by a modal "
                          f"operator (a dialog or a running tool).")}
    data["ok"] = True
    data["stage"] = "ok"
    return data


def append_blend(blend_path, host="127.0.0.1", port=8119, mode="append", collections=None,
                 clear_scene=False, frame=True, timeout=180):
    """Bring a finished .blend into the running Blender. Returns the addon's result dict."""
    if not os.path.isfile(blend_path):
        return {"ok": False, "error": f"blend file not found: {blend_path}"}
    reply = _reply_path("append")
    names = ",".join(collections) if collections else ""
    message = (f"BLEND_APPEND:{blend_path}|MODE:{mode}|RESULT:{reply}"
               f"|COLLECTIONS:{names}|CLEAR:{str(bool(clear_scene)).lower()}"
               f"|FRAME:{str(bool(frame)).lower()}")
    ok, error = send(host, port, message, timeout=5)
    if not ok:
        return {"ok": False, "stage": "connect", "error": error}
    data = _await_reply(reply, timeout)
    try:
        os.remove(reply)
    except OSError:
        pass
    if data is None:
        return {"ok": False, "stage": "reply",
                "error": (f"Blender did not confirm the import within {timeout}s. Check the Blender "
                          f"console. If the addon predates v2.2.1 it does not understand "
                          f"BLEND_APPEND - reinstall it and restart the listener.")}
    return data


def repo_addon_version():
    """The addon version shipped in this checkout, read from its bl_info."""
    path = os.path.join(os.path.dirname(config.BLENDER_SCRIPTS_DIR), "blender_toolbox_addon.py")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
        m = re.search(r'"version"\s*:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)', head)
        if m:
            return tuple(int(g) for g in m.groups())
    except OSError:
        pass
    return None


def _as_tuple(text):
    try:
        return tuple(int(p) for p in str(text).split("."))
    except (TypeError, ValueError):
        return None


def addon_version_warning(status):
    """Warn when the addon running in Blender is older than the one in this checkout.

    A stale addon is the single most confusing failure mode here: the node exists, the socket
    connects, and the feature silently does not work because that Blender is running last month's
    file from the user's addons folder.
    """
    running = _as_tuple(status.get("addon_version"))
    shipped = repo_addon_version()
    if not running or not shipped:
        return ""
    if running < shipped:
        return ("\n  ADDON OUT OF DATE: Blender is running "
                f"{'.'.join(map(str, running))} but this toolbox ships "
                f"{'.'.join(map(str, shipped))}.\n"
                "    Features added since then will not work. In Blender: Edit > Preferences >\n"
                "    Add-ons, remove 'ComfyUI Blender Toolbox Sync', Install...\n"
                "    blender_scripts/blender_toolbox_addon.py, enable it, then Start Listener again.")
    if running > shipped:
        return (f"\n  Note: Blender is running addon {'.'.join(map(str, running))}, newer than this "
                f"checkout's {'.'.join(map(str, shipped))}.")
    return ""


def describe(status):
    """Human-readable summary of a ping result."""
    if not status.get("ok"):
        return f"BRIDGE NOT WORKING\n{status.get('error', 'unknown error')}"
    lines = [
        "BRIDGE OK - ComfyUI can talk to your running Blender",
        f"  Blender        : {status.get('blender_version')}",
        f"  Toolbox addon  : {status.get('addon_version')}",
        f"  Listener       : {status.get('listener', {}).get('host')}:{status.get('listener', {}).get('port')}"
        f" ({'running' if status.get('listener', {}).get('running') else 'flagged stopped'})",
        f"  Open file      : {status.get('blend_file') or '(unsaved scene)'}",
        f"  Scene          : '{status.get('scene_name')}' - {status.get('object_count')} objects "
        f"({status.get('mesh_count')} meshes, {status.get('light_count')} lights, "
        f"camera: {'yes' if status.get('has_camera') else 'no'})",
        f"  Render engine  : {status.get('render_engine')}",
        f"  AI live exec   : {'ALLOWED' if status.get('ai_exec_allowed') else 'BLOCKED (switch is off)'}",
    ]
    stale = addon_version_warning(status)
    if stale:
        lines.append(stale.lstrip("\n"))
    if not status.get("ai_exec_allowed"):
        lines.append("    -> execution_mode='live' will be refused until you tick")
        lines.append("       ComfyUI tab > AI Scene Builder (Live) > Allow AI code execution.")
        lines.append("       Sending a finished scene with 'Send Scene to Blender' works without it.")
    return "\n".join(lines)
