# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Static safety scan for generated scripts.

THIS IS NOT A SANDBOX. It is a readable list of patterns that should never appear in a
scene-building script. A determined script can evade a keyword scan; the real protections are:
saved scripts you can read, dry-run mode, and the addon's opt-in switch for live execution.
"""

import re

# (pattern, message). Any hit blocks execution unless the user explicitly disables the scan.
BLOCKED = [
    (r"\bimport\s+(subprocess|socket|shutil|urllib|requests|http|ctypes|multiprocessing|threading|pickle|marshal|importlib|webbrowser|pathlib|glob|tempfile|sys)\b", "imports a module with file/network/process access"),
    (r"\bfrom\s+(subprocess|socket|shutil|urllib|requests|http|ctypes|multiprocessing|threading|pickle|marshal|importlib|webbrowser|pathlib|glob|tempfile|sys)\b", "imports a module with file/network/process access"),
    (r"\bos\s*\.\s*(system|popen|remove|unlink|rmdir|removedirs|rename|replace|chmod|chown|kill|startfile|spawn\w*|exec\w*|makedirs|mkdir|listdir|walk|scandir|environ)\b", "uses os.* to touch files, processes or environment"),
    (r"\b__import__\s*\(", "dynamic import"),
    (r"\b(eval|exec|compile)\s*\(", "dynamic code execution"),
    (r"\bopen\s*\(", "opens a file (writes are not allowed from generated scripts)"),
    (r"\bbpy\s*\.\s*ops\s*\.\s*wm\s*\.", "bpy.ops.wm.* (open/save/quit) - the runner controls the .blend file"),
    (r"\bbpy\s*\.\s*ops\s*\.\s*script\s*\.", "bpy.ops.script.* (runs other scripts)"),
    (r"\bbpy\s*\.\s*ops\s*\.\s*preferences\s*\.", "modifies Blender preferences"),
    (r"\bbpy\s*\.\s*app\s*\.\s*handlers\b", "installs persistent handlers"),
    (r"\bbpy\s*\.\s*context\s*\.\s*preferences\b", "modifies Blender preferences"),
    (r"\baddon_utils\b", "enables/disables addons"),
    (r"\b(quit|exit)\s*\(", "terminates the Blender process"),
    (r"\bbpy\s*\.\s*data\s*\.\s*libraries\s*\.\s*(load|write)\b", "reads/writes external .blend libraries"),
]

# Patterns worth telling the user about but not blocking.
WARNINGS = [
    (r"\bclear_scene\s*\(", "clears the whole scene"),
    (r"\bdelete_objects\s*\(", "deletes objects"),
    (r"\bbpy\s*\.\s*data\s*\.\s*\w+\s*\.\s*remove\s*\(", "removes datablocks"),
    (r"\bbpy\s*\.\s*ops\s*\.\s*object\s*\.\s*delete\b", "deletes selected objects"),
    (r"\bbpy\s*\.\s*data\s*\.\s*images\s*\.\s*load\s*\(", "loads image files from disk (read only)"),
    (r"\bwhile\s+True\b", "unbounded loop (runner timeout will kill it)"),
    (r"\bsubdivision_type\b|\blevels\s*=\s*[3-9]", "high subdivision level - heavy geometry"),
]


def scan_code(code):
    """Return {"blocked": [...], "warnings": [...]} with human-readable reasons and line numbers."""
    blocked, warnings = [], []
    for lineno, line in enumerate((code or "").splitlines(), start=1):
        stripped = line.split("#", 1)[0]
        if not stripped.strip():
            continue
        for pattern, message in BLOCKED:
            if re.search(pattern, stripped):
                blocked.append(f"line {lineno}: {message}  ->  {line.strip()[:120]}")
        for pattern, message in WARNINGS:
            if re.search(pattern, stripped):
                warnings.append(f"line {lineno}: {message}  ->  {line.strip()[:120]}")
    return {"blocked": blocked, "warnings": warnings}


def format_scan(scan):
    lines = []
    if scan["blocked"]:
        lines.append("BLOCKED by safety scan:")
        lines.extend("  " + b for b in scan["blocked"])
    if scan["warnings"]:
        lines.append("Warnings:")
        lines.extend("  " + w for w in scan["warnings"])
    return "\n".join(lines) if lines else "Safety scan: clean."
