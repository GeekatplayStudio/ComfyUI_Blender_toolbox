# Security

## API credentials

Version 2.0 stores credentials through Python `keyring`, which delegates secret storage to the operating system. Workflow files should contain only a connection from the Credential Manager node, not a literal secret.

The extension stores a local `geekatplay_keystore.json` index containing credential names only. It is excluded from Git. Legacy `geekatplay_keystore.enc` files are migrated automatically and renamed to `geekatplay_keystore.enc.migrated`.

Do not publish workflows containing API keys entered directly into service-node widgets. Prefer the Credential Manager or the `TRIPO_API_KEY` environment variable.

## AI Scene Builder (model-generated code execution)

The AI Scene Builder nodes execute Python written by a language model inside Blender. **There is no sandbox**; Blender's Python has full access to the machine.

Mitigations shipped with the toolbox (all visible in code):

- Every script is written to `output/ai_scene_builder/<session>/scripts/` before it runs, with a header naming session, step and model; every result is logged to `results/`.
- `nodes/ai_builder/safety.py` blocks scripts that import `subprocess`, `socket`, `shutil`, `urllib`, `requests`, `sys`, ..., call `os.system`/`os.remove`/..., `open()`, `eval`/`exec`, `bpy.ops.wm.*`, or touch preferences/handlers. This is a keyword scan and can be evaded by adversarial code; it is not a security boundary.
- `dry_run` generates and saves without executing; `AI Script Runner` executes only what you reviewed.
- Headless mode runs `blender --background --factory-startup` on the session's own `.blend`. Live execution inside a running Blender requires enabling **Allow AI code execution** in the addon preferences/panel and is logged to `ai_exec_log.txt`.
- A failing step is saved as `scene_FAILED_STEP.blend`; the last good scene is never overwritten.

Use a dedicated output folder, back up `.blend` files, and prefer local models for private prompts. Treat the addon socket listener (port 8119, `127.0.0.1` by default) as a local-only interface; do not bind it to `0.0.0.0` on untrusted networks.

## Reporting a vulnerability

Please report security issues privately through GitHub's security-advisory feature for this repository. Do not open a public issue containing API keys, tokens, or exploit details.
