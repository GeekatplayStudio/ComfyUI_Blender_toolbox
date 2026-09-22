# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""Turn the Blender-side validation JSON (blender_scripts/ai_builder/validate_scene.py) into text."""


def validation_passed(validation):
    if not validation:
        return False
    return bool(validation.get("passed"))


def validation_errors(validation):
    return [i for i in (validation or {}).get("issues", []) if i.get("severity") == "error"]


def validation_warnings(validation):
    return [i for i in (validation or {}).get("issues", []) if i.get("severity") == "warning"]


def summarize_validation(validation, max_issues=25):
    if not validation:
        return "No validation data."
    t = validation.get("totals", {})
    lines = [
        f"Validation: {'PASSED' if validation.get('passed') else 'FAILED'} | "
        f"meshes={t.get('mesh_objects', 0)} verts={t.get('vertices', 0):,} faces={t.get('faces', 0):,} tris={t.get('triangles', 0):,}",
        f"  nonmanifold={t.get('nonmanifold_edges', 0)} boundary={t.get('boundary_edges', 0)} loose_verts={t.get('loose_vertices', 0)} "
        f"loose_edges={t.get('loose_edges', 0)} zero_area={t.get('zero_area_faces', 0)} flipped_normals={t.get('flipped_normals', 0)} "
        f"doubles={t.get('duplicate_vertices', 0)} ngons={t.get('ngons', 0)}",
        f"  textures: missing={t.get('missing_textures', 0)} | naming issues={t.get('naming_issues', 0)} | "
        f"no material={t.get('objects_without_material', 0)} | no UV={t.get('objects_without_uv', 0)}",
    ]
    fixes = validation.get("auto_fixes") or []
    if fixes:
        lines.append("  auto-fixes applied: " + "; ".join(fixes[:10]))
    issues = validation.get("issues", [])
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    for label, group in (("ERRORS", errors), ("warnings", warnings)):
        if group:
            lines.append(f"  {label} ({len(group)}):")
            for i in group[:max_issues]:
                obj = f"[{i.get('object')}] " if i.get("object") else ""
                lines.append(f"    - {obj}{i.get('message')}")
            if len(group) > max_issues:
                lines.append(f"    ... {len(group) - max_issues} more")
    return "\n".join(lines)


def feedback_for_retry(result):
    """Build the 'what went wrong' text handed back to the model."""
    parts = []
    if result.get("error"):
        parts.append("PYTHON EXCEPTION:\n" + (result.get("traceback") or result["error"])[-3000:])
    if result.get("api_hint"):
        parts.append(result["api_hint"])
    v = result.get("validation")
    if v and not v.get("passed"):
        parts.append("VALIDATION ERRORS (must be zero):")
        for i in validation_errors(v)[:20]:
            obj = f"[{i.get('object')}] " if i.get("object") else ""
            parts.append(f"- {obj}{i.get('message')}")
    stdout = (result.get("stdout") or "").strip()
    if stdout:
        parts.append("SCRIPT OUTPUT (tail):\n" + stdout[-1500:])
    return "\n".join(parts) if parts else "Unknown failure."


def format_step_report(step_index, title, result, attempts, script_path):
    ok = result.get("ok", False)
    lines = [f"## Step {step_index}: {title}  ->  {'OK' if ok else 'FAILED'} ({attempts} attempt{'s' if attempts != 1 else ''})",
             f"script: {script_path}"]
    if result.get("blender_version"):
        lines.append(f"blender: {result['blender_version']} | exec {result.get('timings', {}).get('exec_s', '?')}s | total {result.get('timings', {}).get('total_s', '?')}s")
    if result.get("error"):
        lines.append("error: " + str(result["error"])[:500])
    if result.get("built"):
        lines.append("built: " + ", ".join(f"{b['name']}({b['vertices']})" for b in result["built"][:30]))
    if result.get("validation"):
        lines.append(summarize_validation(result["validation"]))
    if result.get("render_path"):
        lines.append(f"preview: {result['render_path']}")
    return "\n".join(lines)
