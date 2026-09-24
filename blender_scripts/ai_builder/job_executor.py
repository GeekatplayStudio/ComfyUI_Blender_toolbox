# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
job_executor - the shared core that runs ONE generated script inside Blender.

Used by:
  step_runner.py           headless: opens/creates the .blend, executes, saves, renders
  blender_toolbox_addon.py live: executes in the Blender the user is looking at (opt-in)

Job dict fields (see step_runner.py docstring for the full list). Result dict:
{ ok, error, traceback, stdout, built:[{name,vertices}], validation, scene, render_path,
  blender_version, timings:{exec_s, validate_s, render_s, total_s}, saved_blend }
"""

import io
import json
import os
import sys
import time
import traceback

import bpy


class Tee(io.TextIOBase):
    """Duplicate prints to the real stdout AND keep a copy for the result JSON."""

    def __init__(self, *streams):
        self.streams = streams
        self.buffer_ = io.StringIO()

    def write(self, s):
        self.buffer_.write(s)
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def getvalue(self):
        return self.buffer_.getvalue()


def write_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp, path)


def set_engine(scene, engine):
    wanted = [engine] if engine else []
    if engine and engine.upper().startswith("BLENDER_EEVEE"):
        wanted = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
    for e in wanted + ["CYCLES"]:
        try:
            scene.render.engine = e
            return e
        except Exception:
            continue
    return scene.render.engine


def enable_gpu(scene):
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type != "CPU"
                    scene.cycles.device = "GPU"
                    return backend
            except Exception:
                continue
    except Exception:
        pass
    scene.cycles.device = "CPU"
    return "CPU"


# Directions a preview camera looks FROM, as multiples of the scene radius. One three-quarter view
# hides everything behind the object, which is exactly when parts look "piled together"; the
# orthogonal views are what tell you whether a fin is at the base or halfway up the hull.
PREVIEW_VIEWS = {
    "three_quarter": (1.0, -1.3, 0.75),
    "front": (0.0, -1.0, 0.06),
    "right": (1.0, 0.0, 0.06),
    "top": (0.0, -0.18, 1.0),
    "back": (0.0, 1.0, 0.06),
    "left": (-1.0, 0.0, 0.06),
}
VIEW_SETS = {
    "single": ["three_quarter"],
    "quad": ["three_quarter", "front", "right", "top"],
    "quad_batch": ["three_quarter", "front", "right", "top"],
    "six": ["three_quarter", "front", "right", "back", "left", "top"],
    "six_batch": ["three_quarter", "front", "right", "back", "left", "top"],
}


def setup_preview_environment(helpers, force=False):
    """Neutral studio world and a three-point rig, scaled to whatever is in the scene.

    `force` replaces whatever world the generated script set up. A review preview is for reading
    geometry, and a script that picked a near-black night sky makes its own model unreadable - so
    the builder's own camera mode always renders in a neutral studio.
    """
    scene = bpy.context.scene
    if scene.world is None or force:
        # A gradient, not a flat colour: metal reflects its environment, so a uniform world makes
        # polished surfaces render as the background colour and the object looks like it vanished.
        # Kept light so silhouettes read against it instead of sinking into a dark field.
        helpers.set_world(color=(0.88, 0.89, 0.92), strength=1.0, name="AI_Preview_World",
                          gradient=True, horizon_color=(0.55, 0.57, 0.60))
    preview_lights = [o for o in scene.objects if o.name.startswith("AI_Preview_") and o.type == "LIGHT"]
    user_lights = [o for o in scene.objects if o.type == "LIGHT" and not o.name.startswith("AI_Preview_")]
    if force or not user_lights:
        for pl in preview_lights:
            bpy.data.objects.remove(pl, do_unlink=True)
        lo, hi = helpers.scene_bounds()
        size = max((hi - lo).length, 0.1)
        center = (lo + hi) / 2.0
        # Area lights fall off with distance, so energy has to track the subject's size.
        energy = max(size * size * 220.0, 12.0)
        helpers.add_light("AREA", location=(center.x - size, center.y - size * 1.2, center.z + size),
                          energy=energy, size=size * 1.2, name="AI_Preview_Key", target=center)
        helpers.add_light("AREA", location=(center.x + size * 1.2, center.y - size * 0.8, center.z + size * 0.3),
                          energy=energy * 0.35, size=size * 1.4, name="AI_Preview_Fill", target=center)
        helpers.add_light("AREA", location=(center.x + size * 0.3, center.y + size * 1.3, center.z + size * 0.9),
                          energy=energy * 0.5, size=size, name="AI_Preview_Rim", target=center)


def render_preview(job_render, helpers):
    """Render the scene for review. Returns a list of image paths (one per view).

    Defaults to the builder's own camera rather than whatever camera the script happened to leave
    behind: a model-placed camera is frequently pointed at nothing, and a preview you cannot read
    is worse than no preview.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    camera_mode = str(job_render.get("camera", "preview")).lower()
    scene_cam = scene.camera
    use_scene_cam = camera_mode == "scene" and scene_cam is not None
    # Review mode gets a neutral studio; "scene" mode shows the lighting the script intended.
    setup_preview_environment(helpers, force=not use_scene_cam)
    if use_scene_cam:
        cam = scene_cam
    else:
        cam = bpy.data.objects.get("AI_Preview_Camera")
        if cam is None or cam.type != "CAMERA":
            cam = helpers.add_camera(name="AI_Preview_Camera", lens_mm=50, make_active=False)
        scene.camera = cam

    scene.render.resolution_x = int(job_render.get("width", 768))
    scene.render.resolution_y = int(job_render.get("height", 512))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    engine = set_engine(scene, job_render.get("engine", "CYCLES"))
    samples = int(job_render.get("samples", 16))
    if engine == "CYCLES":
        scene.cycles.samples = samples
        # Denoising is why a 16-sample preview can look clean instead of sandblasted. Previews are
        # for reading shapes, not for final quality, so this is always worth it.
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        enable_gpu(scene)
    else:
        try:
            scene.eevee.taa_render_samples = max(samples, 16)
        except Exception:
            pass
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass

    base = job_render["path"]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    stem, ext = os.path.splitext(base)
    views = VIEW_SETS.get(str(job_render.get("views", "single")).lower(), ["three_quarter"])
    if use_scene_cam:
        views = ["scene"]

    written = []
    for view in views:
        if not use_scene_cam:
            helpers.frame_camera_to_scene(cam, margin=1.18, direction=PREVIEW_VIEWS[view])
        path = base if len(views) == 1 else f"{stem}_{view}{ext}"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        if os.path.exists(path):
            written.append(path)
            print(f"PREVIEW {view} {path}", flush=True)
    if not use_scene_cam and scene_cam is not None:
        scene.camera = scene_cam      # leave the scene's own camera as the active one
    return written


def focus_viewport_on_new(built):
    """Live mode: select what was just built, frame it in every 3D view and force a redraw.

    Without this the objects exist but the user keeps looking at their old viewport framing and
    thinks nothing happened.
    """
    names = [b.get("name") for b in built if isinstance(b, dict) and b.get("name")]
    objects = [bpy.data.objects.get(n) for n in names]
    objects = [o for o in objects if o is not None]
    view_layer = bpy.context.view_layer
    for ob in bpy.context.scene.objects:
        try:
            ob.select_set(False)
        except Exception:
            pass
    for ob in objects:
        try:
            ob.select_set(True)
        except Exception:
            pass
    if objects:
        view_layer.objects.active = objects[-1]
    view_layer.update()
    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    if objects:
                        bpy.ops.view3d.view_selected(use_all_regions=False)
                    else:
                        bpy.ops.view3d.view_all(use_all_regions=False)
            except Exception:
                pass
            area.tag_redraw()
    print(f"[gap] viewport framed on {len(objects)} new object(s)", flush=True)


def signature_hint(exc, tb_text, helpers_module):
    """Turn a signature mistake into the correct signature.

    'trim_ring() got an unexpected keyword argument phase' is useless on its own - the model just
    guesses again. Returning the real signature plus its docstring makes the next attempt a fix.
    """
    import inspect
    import re as _re

    message = str(exc)
    names = set(_re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(\)", message))
    names |= set(_re.findall(r"Builder\.([A-Za-z_][A-Za-z0-9_]*)", message))
    for line in (tb_text or "").splitlines():
        m = _re.search(r"\b([a-z_][a-z0-9_]*)\(", line)
        if m:
            names.add(m.group(1))
    if isinstance(exc, (TypeError, AttributeError)):
        names |= set(_re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", message))

    lines = []
    seen = set()
    builder_cls = getattr(helpers_module, "Builder", None)
    for name in names:
        if name in seen or name.startswith("_"):
            continue
        target = getattr(helpers_module, name, None)
        label = name
        if target is None and builder_cls is not None:
            target = getattr(builder_cls, name, None)
            label = f"Builder.{name}"
        if target is None or not callable(target):
            continue
        try:
            sig = str(inspect.signature(target)).replace("(self, ", "(").replace("(self)", "()")
        except (TypeError, ValueError):
            continue
        seen.add(name)
        doc = (inspect.getdoc(target) or "").strip().splitlines()
        summary = doc[0] if doc else ""
        lines.append(f"  {label}{sig}" + (f"\n      {summary}" if summary else ""))
    if not lines:
        return ""
    return ("CORRECT SIGNATURES for the gap_helpers functions involved (parameters after '*' are "
            "KEYWORD-ONLY - you must write mat=..., n=..., not pass them positionally):\n"
            + "\n".join(sorted(lines)))


def execute_job(job, save=True, scene_source=None):
    """Execute the script, validate, probe, save, render. Never raises; returns the result dict."""
    t0 = time.time()
    result = {"ok": False, "error": None, "traceback": "", "stdout": "", "built": [], "validation": None,
              "scene": None, "render_path": None, "blender_version": bpy.app.version_string,
              "timings": {}, "saved_blend": None, "scene_source": scene_source, "mode": job.get("mode", "headless")}
    helpers_dir = job.get("helpers_dir") or os.path.dirname(os.path.abspath(__file__))
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    if job.get("session_dir"):
        os.environ["GAP_SESSION_DIR"] = job["session_dir"]

    tee = Tee(sys.__stdout__)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = tee, tee
    try:
        import gap_helpers  # noqa: E402
        import scene_probe  # noqa: E402
        import validate_scene  # noqa: E402

        with open(job["code_path"], "r", encoding="utf-8") as f:
            code = f.read()
        t_exec = time.time()
        try:
            namespace = {"__name__": "__main__", "__file__": job["code_path"]}
            exec(compile(code, job["code_path"], "exec"), namespace)
            result["ok"] = True
        except Exception as e:
            result["ok"] = False
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc()
            # Most failures are the model guessing a helper's signature. Hand back the real one so
            # the retry is a correction rather than another guess.
            hint = signature_hint(e, result["traceback"], gap_helpers)
            if hint:
                result["api_hint"] = hint
        result["timings"]["exec_s"] = round(time.time() - t_exec, 2)

        for line in tee.getvalue().splitlines():
            if line.startswith("BUILT "):
                parts = line.split()
                try:
                    result["built"].append({"name": " ".join(parts[1:-1]), "vertices": int(parts[-1])})
                except Exception:
                    pass

        if job.get("validate", True):
            t_val = time.time()
            result["validation"] = validate_scene.validate_scene(
                auto_fix_normals=job.get("auto_fix_normals", False),
                auto_fix_doubles=job.get("auto_fix_doubles", False),
                exclude_names=("AI_Preview_Camera", "AI_Preview_Sun"),
            )
            result["timings"]["validate_s"] = round(time.time() - t_val, 2)
            if result["ok"] and not result["validation"]["passed"]:
                result["ok"] = False
                result["error"] = "validation failed: " + "; ".join(
                    i["message"] for i in result["validation"]["issues"] if i["severity"] == "error")[:800]

        if job.get("probe", True):
            result["scene"] = scene_probe.probe_scene()

        # Save BEFORE rendering so a render crash never loses work. A failed script must not
        # overwrite the good .blend - it goes to *_FAILED_STEP.blend for inspection instead.
        blend_out = job.get("blend_out")
        if save and blend_out:
            if result["ok"]:
                os.makedirs(os.path.dirname(blend_out), exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=blend_out, compress=True)
                result["saved_blend"] = blend_out
            elif job.get("save_failed_copy", True):
                failed_copy = os.path.splitext(blend_out)[0] + "_FAILED_STEP.blend"
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=failed_copy, compress=True)
                    result["saved_blend"] = failed_copy
                except Exception:
                    pass

        if job.get("render") and job["render"].get("path"):
            t_ren = time.time()
            try:
                paths = render_preview(job["render"], gap_helpers)
                result["render_paths"] = paths
                result["render_path"] = paths[0] if paths else None
            except Exception as e:
                result["render_error"] = f"{type(e).__name__}: {e}"
            result["timings"]["render_s"] = round(time.time() - t_ren, 2)

        if job.get("mode") == "live" and job.get("focus_viewport", True):
            try:
                focus_viewport_on_new(result.get("built") or [])
            except Exception as e:
                print(f"[gap] could not focus the viewport: {e}")
    except Exception as e:
        result["ok"] = False
        result["error"] = result["error"] or f"executor: {type(e).__name__}: {e}"
        result["traceback"] = (result["traceback"] + "\n" + traceback.format_exc()).strip()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        result["stdout"] = tee.getvalue()[-20000:]
        result["timings"]["total_s"] = round(time.time() - t0, 2)
    return result
