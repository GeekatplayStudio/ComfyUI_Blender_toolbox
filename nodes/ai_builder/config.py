# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""Central configuration. Plain constants, no hidden behaviour."""

import os
import tempfile

TOOLBOX_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER_SCRIPTS_DIR = os.path.join(TOOLBOX_ROOT, "blender_scripts", "ai_builder")
REFERENCE_DOCS_DIR = os.path.join(TOOLBOX_ROOT, "docs", "ai_builder", "reference")
STEP_RUNNER_SCRIPT = os.path.join(BLENDER_SCRIPTS_DIR, "step_runner.py")

SESSIONS_DIRNAME = "ai_scene_builder"

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_URL = "https://api.openai.com"

# "auto" makes the Model Config node ask Ollama what is installed and pick the strongest model for
# each job. Model size is the biggest quality factor here, and the best one is rarely whatever a
# workflow file happens to name, so auto-selection is the default. Type a name to pin it instead.
AUTO_MODEL = "auto"
DEFAULT_CODE_MODEL = AUTO_MODEL
DEFAULT_VISION_MODEL = AUTO_MODEL
DEFAULT_EMBED_MODEL = AUTO_MODEL

# Used only when nothing is installed yet, and by the installer as what to pull.
FALLBACK_CODE_MODEL = "qwen2.5-coder:14b"
FALLBACK_VISION_MODEL = "qwen2.5vl:7b"
FALLBACK_EMBED_MODEL = "nomic-embed-text"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-5"

# Used by the installer and the docs. Sizes are approximate download sizes.
RECOMMENDED_OLLAMA_MODELS = {
    "code": [
        ("qwen2.5-coder:14b", "~9 GB", "Default. Best balance of bpy correctness and speed on a 12-16 GB GPU."),
        ("qwen2.5-coder:32b", "~20 GB", "Highest local code quality. Needs ~24 GB VRAM (or slow CPU offload)."),
        ("qwen3-coder:30b", "~19 GB", "MoE, only ~3B active params, fast; built for agentic tool loops."),
        ("qwen2.5-coder:7b", "~4.7 GB", "Budget option for 8 GB GPUs. Expect more retries."),
    ],
    "vision": [
        ("qwen2.5vl:7b", "~6 GB", "Default. Strong layout/material extraction from reference images."),
        ("qwen3-vl:4b", "~3.3 GB", "Smaller, fast, supports tools."),
        ("gemma3:4b", "~3.3 GB", "Generalist already used by the toolbox's other Ollama nodes."),
    ],
    "embedding": [
        ("nomic-embed-text", "~275 MB", "Optional. Improves reference-doc retrieval; BM25 fallback works without it."),
    ],
}

DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 16384
DEFAULT_KEEP_ALIVE = "10m"
DEFAULT_TIMEOUT = (10, 900)  # (connect, read) seconds - big local models are slow on first load
DEFAULT_MAX_TOKENS = 8192

DEFAULT_MAX_RETRIES = 2
DEFAULT_BLENDER_TIMEOUT = 900  # seconds per step (headless subprocess)
DEFAULT_LIVE_TIMEOUT = 600     # seconds waiting for the addon to write a result file

DEFAULT_RENDER_ENGINE = "CYCLES"
DEFAULT_RENDER_WIDTH = 768
DEFAULT_RENDER_HEIGHT = 512
DEFAULT_RENDER_SAMPLES = 16
# One three-quarter view hides everything behind the object, which is exactly when parts look
# "piled together". Four views tell you where things actually are.
DEFAULT_PREVIEW_VIEWS = "quad"

# Blender-side probe/validation are imported by the generated scripts too.
HELPERS_MODULE_NAME = "gap_helpers"

SANDBOX_WARNING = (
    "SECURITY WARNING - READ BEFORE USE\n"
    "The AI Scene Builder executes Python code written by a language model inside Blender.\n"
    "There is NO sandbox. Blender's Python has full access to your files, network and processes.\n"
    "What the toolbox does to keep this transparent (0% black box):\n"
    "  * Every generated script is saved to <ComfyUI output>/ai_scene_builder/<session>/scripts/ BEFORE it runs.\n"
    "  * Every execution result (stdout, errors, validation JSON) is saved next to it in results/.\n"
    "  * A static safety scan (nodes/ai_builder/safety.py) refuses scripts that import subprocess/socket,\n"
    "    call os.system/os.remove/shutil.rmtree, open files for writing outside the session folder, etc.\n"
    "    This is a keyword scan, not a sandbox - it can be fooled. Review scripts you do not trust.\n"
    "  * 'dry_run' mode generates and saves the script WITHOUT executing it, so you can read it first\n"
    "    and run it later with the 'AI Script Runner' node.\n"
    "  * Live execution inside a running Blender is OFF until you enable\n"
    "    'Allow AI code execution' in the addon's ComfyUI sidebar panel.\n"
    "Use a dedicated ComfyUI output folder and back up .blend files you care about."
)


def get_sessions_root():
    """Where sessions live: <ComfyUI output>/ai_scene_builder (falls back to temp when not inside ComfyUI)."""
    try:
        import folder_paths  # type: ignore
        base = folder_paths.get_output_directory()
    except Exception:
        base = os.path.join(tempfile.gettempdir(), "comfyui_output")
    root = os.path.join(base, SESSIONS_DIRNAME)
    os.makedirs(root, exist_ok=True)
    return root
