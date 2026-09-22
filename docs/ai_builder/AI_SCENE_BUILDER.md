# AI Scene Builder — prompt (+ reference images) → modeled, textured, lit Blender scene

Part of **ComfyUI-Blender-Toolbox** by Geekatplay Studio — Vladimir Chopine.

The AI Scene Builder lets a language model do the actual Blender work: it writes a Python script
for each build step, the toolbox runs it in Blender, validates the result (polygons, normals,
textures, names), renders a preview, and hands the model its own errors to fix. Scenes are built in
passes — "build the forest", then "add a castle", then "add a river with a bridge" — and every pass
sees what already exists. Complex requests are planned into ordered steps and executed one by one.

> **0 % black box.** Every prompt is in [`nodes/ai_builder/prompts.py`](../../nodes/ai_builder/prompts.py).
> Every generated script is saved to disk *before* it runs. Every execution writes a JSON result.
> The reference docs the model reads are plain Markdown in [`reference/`](reference/). Nothing is hidden.

---

## Security — read this first

**The builder executes Python written by a model inside Blender. There is no sandbox.** Blender's
Python interpreter has the same access to your computer as you do: files, network, processes.

What the toolbox does about it:

| Protection | What it does | Where |
|---|---|---|
| Scripts saved first | Every script lands in `<session>/scripts/step_NNN_attemptK.py` before execution, with a header naming session/step/model | `agent.py` |
| Results logged | `<session>/results/*.json` holds stdout, errors, validation, scene probe for every run | `job_executor.py` |
| Safety scan | Refuses scripts that import `subprocess/socket/shutil/urllib/requests/sys/...`, call `os.system/os.remove/...`, `open()`, `eval/exec`, `bpy.ops.wm.*` (open/save/quit), preferences/handlers. **It is a keyword scan — not a sandbox — it can be evaded.** | `safety.py` |
| `dry_run` | Generates + saves the script and stops. Read it, then execute with **AI Script Runner** | all builder nodes |
| Live mode opt-in | Executing in a *running* Blender requires switching **Allow AI code execution** ON in the addon panel; otherwise jobs are refused and logged | `blender_toolbox_addon.py` |
| Headless isolation | Default mode runs `blender --background --factory-startup` on the *session's* `.blend`, never on your open file | `blender_runner.py` |
| Failed steps never overwrite | A failing script is saved to `scene_FAILED_STEP.blend`; the good `scene.blend` is untouched | `job_executor.py` |

Recommendations: use a dedicated ComfyUI output folder; back up `.blend` files you care about; read
the scripts of any session you did not watch; keep live mode OFF unless you are using it right now;
run with a local model when the prompt contains anything private.

---

## Requirements

- **Blender 4.x or 5.x** installed (5.2 LTS tested). Auto-detected from `C:\Program Files\Blender Foundation\`,
  common Linux/macOS paths or `PATH`; override with the `BLENDER_PATH` environment variable or the
  node's `blender_path` input.
- **A model**. The Model Config node defaults every model field to **`auto`**, which asks Ollama
  what is installed and takes the largest model able to do each job, printing what it chose:

  ```
  AUTO-SELECTED (size is the biggest quality factor, so the largest capable model wins):
    model        auto -> qwen2.5-coder:32b (32.8B, purpose-built coder)
    vision_model auto -> qwen3.8:latest (27.3B, largest model with vision)
    embed_model  auto -> nomic-embed-text:latest
  ```

  This matters because the best model is rarely the one a workflow file happens to name — a tag
  like `qwen3.8:latest` hides a 27B model behind an unremarkable name, and a 7B vision model
  reports every part as the same size. Type a model name to pin one instead.

  Local **Ollama** options if you prefer to choose:
  - code/planning: `qwen2.5-coder:14b` (default), `qwen2.5-coder:32b` (best local), `qwen3-coder:30b`
  - vision (reference images): `qwen2.5vl:7b` (default), `qwen3-vl:4b`, `gemma3`
  - embeddings (optional, better doc retrieval): `nomic-embed-text`
  Cloud alternatives via the **AI Model Config** node: `anthropic` (`claude-sonnet-5`, `claude-opus-5`)
  or any `openai_compatible` endpoint. Keys come from the API Key Manager (names `Anthropic`,
  `OpenAI`, `Ollama Cloud`) or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables.
- Python deps: only what the toolbox already needs (`requests`, `pillow`, `numpy`, `torch` via ComfyUI).

### Install

```bash
# from the toolbox folder, with ComfyUI's python (portable: ..\..\..\python_embeded\python.exe)
python installer/install_ai_builder.py --smoke-test
```
or double-click `installer/install_ai_builder.bat`. It checks Blender, starts Ollama, pulls the three
default models (`--code-model/--vision-model/--embed-model` to change, `--skip-models` to skip),
indexes the reference docs, and builds+validates+renders a test cube in headless Blender. The main
`install.bat` / `installer/install.py` call it too.

For live mode also (re)install the Blender addon: `blender_scripts/blender_toolbox_addon.py`
(Edit → Preferences → Add-ons → Install), then in the 3D viewport sidebar → **ComfyUI** tab →
**Start Listener** and, only when you want live execution, **AI Scene Builder (Live) → Allow AI code execution**.

---

## Nodes (category `Geekatplay Studio/AI Scene Builder`)

| Node | Purpose |
|---|---|
| **AI Scene Session** | Names the scene. Creates/continues `output/ai_scene_builder/<name>/` with `scene.blend`, `scripts/`, `results/`, `renders/`, `refs/`, `session.json`. `reset` archives everything (switch it off after one run). `blend_file` continues from an existing `.blend`. |
| **AI Model Config** | Provider, model, vision model, URL, temperature, context size, API key. Lists what your Ollama has and flags missing models. |
| **AI Reference Analyzer (Multi-Image)** | Up to three image inputs (batches count as multiple references) + your text → one structured **reference brief** (subject, style, layout, elements with sizes, materials, colors, lighting, camera, must-not-miss). Copies references to `refs/`. |
| **AI Scene Planner** | Prompt (+ brief, + current scene) → ordered JSON plan of build steps. Or write your own steps in `manual_plan` (one per line, `Title: instruction`) to skip the model. |
| **AI Scene Builder (Complete Scene)** | Executes every plan step: generate → safety scan → run in Blender → validate → retry (`max_retries`) → preview. `stop_on_failure`, `dry_run`, auto-fixes, render settings. Outputs preview, blend path, report, all scripts, validation JSON, session. |
| **AI Step Builder (Conversational)** | One instruction per run, building on the session scene: "now add a bridge". Same options and outputs. |
| **AI Script Runner (Review & Execute)** | Runs a script you pasted/edited (e.g. from a dry run) with the same scan, validation and render. |
| **AI Visual Refiner (Compare to Reference & Fix)** | Renders the scene, puts it beside your reference image, asks the vision model what is wrong (silhouette, missing parts, shape, placement, materials, detail density), turns the critique into correction steps and executes them. Repeats until the resemblance score hits `target_score`. This is what closes the gap to a reference — one generate-and-hope pass never sees its own output. |
| **Blender Bridge Check (Is Blender Open?)** | Full round trip to a running Blender: reports its version, the addon version, the open file, what is in the scene, and whether live AI execution is allowed. Outputs `connected` and `live_exec_allowed` booleans. |
| **Send Scene to Blender (Append/Link/Open)** | Loads a finished `.blend` into the Blender you already have open, with materials, collections, lights and cameras intact. |
| **AI Debug Log (Full Step-by-Step Trace)** | Reads `<session>/debug.log`: every prompt sent, every raw model reply, every generated script, the retrieved reference chunks, Blender's stdout, validation totals and every error, in order. Filters: everything / errors and warnings only / prompts and replies only / last step only. |
| **AI Scene Validator** | QA any session scene: polygons, normals, textures, names; optional auto-fix; preview render; `passed` boolean. |

Execution options shared by the builder nodes:

- `execution_mode` — `headless` (default; separate background Blender on the session file) or
  `live` (inside your running Blender via the addon; requires the opt-in switch).
- `dry_run` — generate and save only.
- `max_retries` — how many times the model may fix its own script after an exception or failed validation.
- `validate`, `auto_fix_normals`, `auto_fix_doubles` — see Validation below.
- `safety_scan` — keep ON.
- `render_preview`, `render_engine` (`CYCLES` default, GPU when available; `BLENDER_EEVEE`),
  `render_width/height/samples`.
- `blender_timeout` — seconds per step (headless).
- optional: `blender_path`, `live_host/port/save`, `extra_reference_notes` (your own API notes, added to retrieval).

---

## Workflows (folder `workflows/`)

| File | What it does |
|---|---|
| `Geekatplay_AI_Scene_Builder_Complete.json` | References → brief → plan → complete multi-step build with report. |
| `Geekatplay_AI_Single_Step_From_References.json` | Reference images → one validated build step. |
| `Geekatplay_AI_Step_Builder_Conversational.json` | Type, queue, look, type the next instruction. Multi-pass building. |
| `Geekatplay_AI_Script_Review_Then_Run.json` | Dry run → read the script → paste into the runner → execute. Safest. |
| `Geekatplay_AI_Scene_Validator.json` | QA + auto-fix + preview for any session scene. |

All workflows are generated from the node definitions by `tools/generate_ai_workflows.py`, so the
sockets and widget values always match the code. Each contains a note with instructions and the
security summary.

---

## How a step works

```
instruction ──► RAG: top-k chunks from docs/ai_builder/reference/*.md (BM25 + optional embeddings)
            ──► prompt = CODEGEN_SYSTEM + CODEGEN_USER(instruction, scene probe, history, brief, chunks)
            ──► model returns one ```python block
            ──► saved to scripts/step_NNN_attemptK.py  ──► safety scan (blocked? → model fixes it)
            ──► dry_run? stop.
            ──► headless: blender -b --factory-startup --python step_runner.py -- job.json
                live:     AI_EXEC:<job.json> → addon → job_executor.execute_job()
                   open scene.blend (or empty) → exec(script) → validate → probe → save → render
            ──► result JSON (ok, error, traceback, stdout, built[], validation, scene, render_path)
            ──► failed? feedback (exception + validation errors + stdout tail) → retry
            ──► session.json turn: instruction, script, result, summary, scene, validation
```

Generated scripts get `from gap_helpers import *` — a small, tested library
(`blender_scripts/ai_builder/gap_helpers.py`) with a `Builder` for closed primitives, version-safe
`make_material`, lights/camera/world helpers, `terrain()` and `scatter()`. It is what makes 7B–32B
local models produce manifold geometry on the first or second try. Its API is documented for the
model in [`reference/04_gap_helpers_reference.md`](reference/04_gap_helpers_reference.md).

### Multi-pass memory

The `.blend` file is the memory. Before each step the runner probes the scene (objects with
location/dimensions/materials/collection, lights, cameras, world, bounds) and the agent puts that,
plus a summary of previous steps, into the prompt. That is why "add a building to that forest"
works across separate ComfyUI runs, and why deleting the session folder is the only way to forget.

### Multi-step plans

The planner produces `{"steps":[{"title","category","instruction"}]}` (foundation → structures →
props → materials → lighting → camera). The builder runs them in order; each step gets its own
collection `Step_NN_Title`, its own script, validation and render. `stop_on_failure` halts the plan
when a step cannot be repaired within `max_retries`; the report tells you exactly where and why.

---

## Validation (what "verify broken polygons / normals / textures / names" means here)

Runs inside Blender after every step (`blender_scripts/ai_builder/validate_scene.py`):

| Check | Severity | Auto-fix |
|---|---|---|
| Non-manifold edges | error | – (model retries) |
| Loose vertices / loose edges | error | `auto_fix_doubles` deletes them |
| Zero-area (degenerate) faces | error | dissolved by `auto_fix_doubles` |
| Flipped / inconsistent normals (closed meshes) | error | `auto_fix_normals` recalculates outward |
| Missing or unloadable image textures | error | – |
| Python exception in the script | error | – (model retries with the traceback) |
| Boundary edges (open mesh) | warning | – |
| Duplicate vertices | warning | `auto_fix_doubles` merges |
| N-gons | warning | – |
| Default names (`Cube.001`, `Material`, `Light`, `Collection`…) on objects/meshes/materials/lights/cameras/collections | warning | – |
| No material / image textures without UVs | warning | – |
| Unused images or materials, unconnected shader outputs | warning | – |
| No camera / no lights | warning | preview adds `AI_Preview_Camera` / `AI_Preview_Sun` |

Errors fail the step (the model gets the list and retries); warnings are reported. Linked duplicates
(scatter) share mesh data and are validated once. Totals and per-object numbers are in the
`validation_json` output and in `results/*.json`.

---

## Getting the result into the Blender you have open

A headless build runs in a **background** Blender, so nothing appears in your open window. There are
two ways to see it on your desk, and they need different things switched on:

| You want | Use | Requires in Blender |
|---|---|---|
| The finished model dropped into my current scene | **Send Scene to Blender** node (`mode=append`) | addon v2.2.1+, **Start Listener** |
| To watch it being built, step by step | builder node with `execution_mode=live` | addon v2.2.1+, **Start Listener**, **Allow AI code execution** ON |

**Send Scene to Blender** appends the session `.blend`, so material node trees, collections, lights
and cameras all survive — exporting through GLB would flatten most of that. Modes:

- `append` (default) — copies the objects into your current scene. Non-destructive: everything you
  already had stays, and the build arrives in its own `Step_NN_*` collections.
- `link` — references the objects read-only from the `.blend`; edits belong to the source file.
- `open` — **replaces** the file you have open. Unsaved work is lost. Only use it deliberately.

Optional inputs: `collections` (comma-separated, to import only part of a build), `clear_scene_first`,
`frame_viewport` (selects and zooms onto what arrived).

### Checking the bridge before you commit to a long build

**Blender Bridge Check** does a real round trip rather than just opening a socket — a plain connect
only proves *something* holds the port. The addon writes back its status, so the node can tell you:

```
BRIDGE OK - ComfyUI can talk to your running Blender
  Blender        : 5.2.2 LTS
  Toolbox addon  : 2.2.1
  Listener       : 127.0.0.1:8119 (running)
  Open file      : (unsaved scene)
  Scene          : 'Scene' - 7 objects (5 meshes, 1 lights, camera: yes)
  Render engine  : BLENDER_EEVEE
  AI live exec   : BLOCKED (switch is off)
```

Outputs `connected` and `live_exec_allowed` as booleans (wire them into a Logic Switch to branch),
plus the text above and the raw JSON. `raise_on_failure` stops the workflow when the bridge is down;
leave it off to report and continue.

Ready-made workflow: **`Geekatplay_AI_Build_And_Send_To_Blender.json`**.

> **One listener at a time.** Two Blender instances used to be able to bind port 8119 at once on
> Windows, which meant ComfyUI silently talked to whichever won the race. v2.2.1 claims the port
> exclusively, so the second instance now reports the clash in its console instead. If the listener
> refuses to start, close the other Blender (or give it a different port on both ends).

## Session folder layout

```
ComfyUI/output/ai_scene_builder/<session>/
  session.json          turns, plan, reference brief, last scene probe, last validation
  scene.blend           the scene (open it in Blender any time)
  scene_FAILED_STEP.blend   only when a step failed — for inspection
  scripts/step_001_attempt1.py ...     every generated script (dry runs too), manual_*.py, validate_*.py
  results/step_001_attempt1.json / .job.json   what ran and what happened
  renders/step_001_attempt1.png        preview renders
  refs/                 reference images given to the analyzer
  ai_exec_log.txt       live-mode log written by the Blender addon
  archive_YYYYMMDD_HHMMSS/             previous content when you used reset
```

---

## When a build comes out wrong: read the debug log

`<session>/debug.log` records the whole run in order, so you can see *which stage* failed rather
than guessing from the final render. Add the **AI Debug Log** node, or open the file directly.

Read it top to bottom and stop at the first thing that looks wrong:

| What you see in the log | What it means | What to do |
|---|---|---|
| The build specification has every part at a similar size, or fewer than 8 parts | The vision model did not really look at the image | Use a bigger vision model — the Model Config node names the best one you have installed |
| `WARNING: <pass> did not return usable JSON` | The reply was truncated or malformed | Raise `max_tokens`, or use a bigger vision model |
| The plan says "create a cylinder for the body" | The planner is producing primitives instead of assemblies | Raise `max_steps`, or use a bigger code model |
| `TypeError: ... got an unexpected keyword argument` / `missing 1 required positional argument` | The model guessed a helper's signature | The log's `api_hint` shows the correct signature and the retry usually fixes it; three failures in a row means the code model is too small |
| The script is right but the render looks wrong | A modelling problem, not a pipeline problem | Use the **AI Visual Refiner** — it compares the render to your reference and fixes the differences |

`log_path` is an output, so you can wire it into a file node, and the text box has a **Copy text**
button for pasting the log somewhere.

## Tuning tips

- **Model size matters most.** `qwen2.5-coder:32b` needs far fewer retries than 7B. On a 12 GB GPU
  use `qwen2.5-coder:14b` with `num_ctx` 16384. Cloud `claude-sonnet-5` is the most reliable.
- Keep steps concrete: sizes in meters, counts, positions relative to existing objects.
- Use the Reference Analyzer even for text-only projects: it produces a consistent brief that all
  steps share, which keeps style and scale coherent.
- `extra_reference_notes` is the place for your own conventions ("all roofs are copper", "use
  `Mat_` prefix") — it is indexed with the docs for that run.
- Retrieval improves with `nomic-embed-text` pulled; without it BM25 keyword search is used.
- Long plans: raise `blender_timeout`; lower `render_samples` for faster previews.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Blender executable not found` | Install Blender or set `BLENDER_PATH`; check `blender_path` on the node. |
| `Ollama model 'x' not found` | `ollama pull x` or change the model in AI Model Config; run the installer. |
| Reply contained no code block | Model too small / temperature too high; retry, lower temperature, bigger model. |
| Step keeps failing validation on non-manifold edges | Instruct "use Builder primitives, no boolean modifiers"; add to `extra_reference_notes`. |
| Live mode: `REFUSED` in the Blender console | Switch on **Allow AI code execution** in the addon panel (ComfyUI tab). |
| Live mode: timeout | Listener not running (Start Listener), wrong port (8119), or the step is heavy — raise `live_timeout`/`blender_timeout`. |
| ComfyUI Manager install and `subprocess` errors | Headless mode launches Blender with `subprocess`; if your ComfyUI restricts custom-node subprocesses, start ComfyUI with `--allow-subprocess` or use live mode. |
| Preview is grey/blank | Rendering failed (see `render_error` in the result JSON) or `render_preview` was off. |

## Files

```
nodes/ai_builder_nodes.py            ComfyUI nodes
nodes/ai_builder/                    engine: config, session, llm, rag, prompts, safety, blender_runner, validation, agent
blender_scripts/ai_builder/          runs inside Blender: gap_helpers, scene_probe, validate_scene, job_executor, step_runner
blender_scripts/blender_toolbox_addon.py   live mode (AI_EXEC) + opt-in switch + panel
docs/ai_builder/reference/*.md       what the model reads (edit freely; re-index is automatic)
installer/install_ai_builder.py      setup + smoke test
tools/generate_ai_workflows.py       regenerates the workflow JSON files
tests/test_ai_builder.py             unit tests + real Blender integration test
```
