# Changelog

## 2.2.8

We compared our pipeline with a ChatGPT-built Blender character (102 parts, rigged, 8K PBR) and its scripts. It used the same primitives `gap_helpers` has — lathe profiles, rods, boxes, plates, tori, booleans, mirroring — so the difference is not the toolset. It is (a) one model that *sees* the reference writes the numbers straight into code, (b) one script for the whole object, revised after looking at each render, and (c) a frontier model. Two of the three are now in the toolbox:

- **The coder sees the reference.** Every builder node attaches the reference images to the code request when the code model reports the `vision` capability (Claude, GPT, `qwen3-vl`, `qwen3.8`). Previously only the analyzer and the critic saw the picture; the coder worked from prose, which is where the shape got lost. A blind coder gets no images and the log says so.
- **AI Whole-Object Builder (One Script, See & Revise)** — new node and workflow `Geekatplay_AI_Whole_Object_See_And_Revise.json`. One complete script builds the entire object into an empty scene; the render is critiqued against the reference; the model revises the *script*; the scene is wiped and rebuilt. `rounds` / `target_score`; the best-scoring round is kept and its script is an output. Corrections no longer pile up on a scene, and parts interlock because one author wrote them together.
- **Ollama connection drops are retried** (3 attempts, 5/15/40 s, after checking the server is back). A single `RemoteDisconnected` while Ollama swapped a 27B vision model for the 32B coder used to kill a whole build.
- **Scale guard fixed.** It failed the landing legs of a 0.3 m rocket for being "0.1× too short" — early steps build parts, not the whole object. "Too tall" is still judged against the specification; "too small" is now judged only against the z range the step's own instruction gives.
- **`fin_blade` accepts a direction vector or a point** for `outward`, not only an angle. The 32B coder passed `(cos a, sin a, 0)` four attempts in a row; a helper that accepts the natural call beats a prompt that forbids it.
- Docs: what the ChatGPT build actually did and why the model, not the architecture, is the bottleneck (`docs/ai_builder/AI_SCENE_BUILDER.md`, "Two ways to build a whole object").

## 2.2.7

Traced the "four cylinders that don't resemble a rocket" result to its actual causes, using the session's own debug log:

- **Two builds superimposed.** The complete workflow was re-run into a session that still held the previous attempt, so the new rocket was built on top of the old one's debris — 21 meshes from two runs in one scene. `AI Scene Builder (Complete Scene)` now has **`start_fresh` (default ON)**: it archives whatever the session holds and builds into an empty scene. The Step Builder keeps adding, as it should, but now **warns when the reference describes a different object** than the session contains (a robot brief into a rocket session).
- **z positions stacked as heights.** The model wrote a hull outline as `(z, radius)` points and passed it to `stepped_profile`, which takes `(height_of_ring, radius)` and stacks them — a 0.198 m hull came out 0.59 m tall. `stepped_profile` now **refuses steadily increasing "heights"** with a message naming the right helper; the prompt spells the trap out.
- **A cylinder where a bulbous body was asked for.** The profile had the same radius at every point while its comments said "widest part". The codegen prompt now carries a **shape-word → helper table** (bulbous → `barrel_profile`, ogive → `ogive_profile`, dome → `dome_profile` not `sphere`, curved blade → `fin_blade`, hole → `hollow_port`…) and forbids substituting a plain cylinder for any of them. The planner uses the same vocabulary.
- **Scale guard.** After each step the built scene's height is compared with the specification's overall height; more than 1.6× or under 0.5× fails the step and the retry is told exactly what to check. This would have caught the 0.59 m hull on the first attempt.

## 2.2.6

- **Previews are readable now.** Three defects made every preview hard to judge:
  - Cycles denoising was **off**, so a 16-sample preview came back sandblasted with noise. It is now always on — previews exist to show shapes, not to be final frames.
  - The preview reused whatever camera the generated script left behind and **never re-framed it**, so a script that aimed its camera at nothing produced a useless render. The builder now frames its own camera by default (`preview_camera = preview`); set it to `scene` to see the look the script intended.
  - Only one three-quarter angle was rendered, which hides everything behind the object — exactly when parts look piled on top of each other. **`preview_views` now defaults to `quad`**: three-quarter, front, right and top, returned as a single IMAGE batch. The top view is usually what reveals parts stacked in the same place. `single` and `six` are also available.
- Review previews render in a neutral bright studio world regardless of what the script set up, so a model that chose a near-black night sky no longer makes its own geometry invisible.

## 2.2.5

- **All 20 workflows audited, repaired and verified.** New `tools/check_workflows.py` validates node types (against a running ComfyUI when given `--server`), link-table integrity, socket/link agreement, required inputs, and whether saved widget values still line up with the current node definitions. The test suite now fails if any shipped workflow breaks, so they cannot rot silently again.
- Fixed in the process:
  - `Geekatplay_Blender_RoundTrip_Sync.json` had **two different links sharing id 23**, and the Reroute node was wired to the wrong one.
  - `Geekatplay_Blender_RoundTrip_Simple.json` left the required `albedo_map` unconnected, so it could not be queued — now wired from `LoadBlenderPBR.Albedo`, completing the round trip.
  - `Geekatplay_texture_sdxl_seamless_workflow.json` and `Geekatplay_SDXL_360_HDRI.json` carried stale link ids in output sockets, left over from earlier edits.
  - `Geekatplay_Tripo_3D_Workflow.json` shipped pointing at a `.glb` generated on another machine.
- All 18 UI workflows confirmed to load in a live ComfyUI with every node type resolved and every link valid.
- README: workflow list completed (360°/terrain, round-trip simple, autorig template) and documents how to run the checker.

## 2.2.4

- **Models now default to `auto`**: the Model Config node asks Ollama what is installed and takes the largest model able to do each job — the biggest purpose-built coder for the code, the biggest vision-capable model for reading the reference — and prints what it chose and why. Model size is the biggest quality factor in this pipeline, and the best model is rarely the one a workflow file happens to name (a tag like `qwen3.8:latest` hides a 27B model behind an unremarkable name). Type a model name to pin one instead; `-base` models are never chosen because they cannot follow instructions. For the anthropic provider `auto` means `claude-sonnet-5`.
- **Stale addon warning**: the Blender Bridge Check compares the addon version running in Blender against the one shipped in this checkout and spells out the reinstall steps when it is older. A stale addon was the most confusing failure mode — the node exists, the socket connects, and the feature silently does nothing.

## 2.2.3

- **Every build failure in the wild was a helper signature mistake**, not a modelling problem: `Builder.cylinder() got multiple values for argument 'mat'`, `trim_ring() got an unexpected keyword argument 'phase'`, `panel_seams() missing 1 required positional argument: 'z1'`. Three fixes:
  - `mat` and the other options are now **keyword-only** on every Builder primitive and detail helper, and the argument order is consistent (`cylinder` used to take `mat` before `r2` while `cone` took `r2` before `mat` — an API bug that guaranteed confusion).
  - A failed script now comes back with the **correct signature and docstring** of the helper it called wrongly, so the retry is a correction rather than another guess.
  - The signatures in the prompt are now exact, and a test introspects the real module inside Blender and fails if they ever drift apart.
- **Full debug log per session** (`<session>/debug.log` and the new **AI Debug Log** node): every prompt sent, every raw model reply, every generated script, the retrieved reference chunks, Blender's stdout, validation totals and every error, in order — with filters for errors only, prompts only, or the last step.
- **Report text now renders as a proper scrollable monospace text box** instead of one truncated line, with a "Copy text" button. Applies to the Build report, Plan, Reference brief and the new debug log.
- `trim_ring` accepts (and ignores) `phase`, since it is natural to pass alongside `rivet_ring` and a full ring looks identical at any rotation.

## 2.2.2

- **Send Scene to Blender** node: loads a finished `.blend` into the Blender you already have open, keeping material node trees, collections, lights and cameras (a GLB export would flatten them). Modes: `append` (non-destructive, the default), `link`, `open`.
- **Blender Bridge Check** node: a real round trip, not just a socket connect. The addon answers with its version, the open file, the scene contents and whether live AI execution is allowed, so you can confirm the connection before starting a long build. Outputs `connected` and `live_exec_allowed`.
- Addon 2.2.1 adds the `PING` and `BLEND_APPEND` commands that make the above possible. The socket protocol is fire-and-forget, so replies are written as JSON files the caller polls for.
- **Duplicate listeners are no longer silent**: on Windows `SO_REUSEADDR` let a second Blender bind port 8119 as well, and ComfyUI then talked to whichever instance won `accept()`. The listener now claims the port exclusively, so a clash is reported instead of sending your build to the wrong window.
- **AI Visual Refiner** node: renders the scene, compares it to the reference, and executes correction steps until the resemblance score reaches the target.
- New workflow `Geekatplay_AI_Build_And_Send_To_Blender.json`.

## 2.2.1

- **Auto-Rigger CLI completed**: the ComfyUI generation path was a stub that always exited with an error. It now uploads the reference image, patches the workflow's `LoadImage` node, queues the prompt, polls `/history` until completion, downloads the generated mesh and hands it to the Blender clean+rig step. Added `--timeout` and `--server` options and real HTTP error reporting.
- **`workflows/autorig_api.json` rebuilt**: the template referenced `Hunyuan3D_Wrapper` and `Save3DMesh`, which do not exist in ComfyUI. Replaced with the working Hunyuan3D v2.1 chain (`ImageOnlyCheckpointLoader` → `CLIPVisionEncode` → `Hunyuan3Dv2Conditioning` → `KSampler` → `VAEDecodeHunyuan3D` → `VoxelToMesh` → `SaveGLB`). Verified end to end: image → generated mesh → rigged GLB.
- **Credential lookup is now case-insensitive**: a key saved as `Tripo3d` is found when a node asks for `Tripo3D`. Previously such a mismatch silently reported a missing API key.
- Added the missing display name for `SaveAndSendPBRToBlender` (it showed its raw class name in the node menu).

## 2.2.0

- **AI Scene Builder** (new node family, `Geekatplay Studio/AI Scene Builder`):
  - Prompt + up to three reference images → a model writes Blender Python per step; the toolbox executes it in headless Blender, validates, renders a preview and feeds errors back for self-repair.
  - Nodes: AI Scene Session, AI Model Config, AI Reference Analyzer (Multi-Image), AI Scene Planner, AI Scene Builder (Complete Scene), AI Step Builder (Conversational), AI Script Runner (Review & Execute), AI Scene Validator.
  - Multi-pass building: the session `.blend` is the memory; every pass sees the current scene probe and history.
  - Multi-step plans with per-step collections, validation gates and `stop_on_failure`.
  - Validation inside Blender: non-manifold edges, loose geometry, zero-area faces, flipped normals, duplicate vertices, n-gons, missing/broken textures, default names, materials/UVs, camera/lights; optional auto-fixes.
  - Providers: Ollama (local or cloud), Anthropic, OpenAI-compatible — plain `requests`, no SDKs.
  - Retrieval over `docs/ai_builder/reference/*.md` (BM25 + optional Ollama embeddings) grounds generation in version-correct bpy usage.
  - `gap_helpers` library for generated scripts: closed-primitive `Builder`, version-safe materials, lights/cameras/world, terrain, scatter.
  - Safety: scripts saved before execution, static safety scan, `dry_run`, failed steps never overwrite the scene, live mode opt-in switch in the addon.
  - Five ready workflows generated from the node definitions (`tools/generate_ai_workflows.py`).
  - Installer `installer/install_ai_builder.py` (Blender check, Ollama models, reference index, smoke test), hooked into `install.py`.
  - Tests: `tests/test_ai_builder.py` (offline units + real headless Blender integration).
- **Blender addon 2.2.0**: `AI_EXEC` live execution command (off by default, preference + sidebar switch, `ai_exec_log.txt`), listener now reads whole messages instead of a single 4 KB buffer, "Open AI Session Folder" button.
- Branding: Geekatplay Studio — Vladimir Chopine.

## 2.1.0

- **Tripo3D API Update**:
  - Added support for `P2-20260801` (flagship quad topology, game-ready model) and `v2.5-20250123`.
  - Added automatic fallback between `openapi.tripo3d.com` and `openapi.tripo3d.ai` endpoints.
  - Added OS Keyring (`get_key("Tripo")`) and `TRIPO_API_KEY` environment variable resolution.
- **Meshy API Update**:
  - Added support for `meshy-7` high-fidelity 3D generation model in Text-to-3D and Image-to-3D.
  - Added `3mf` format download and export support.
  - Added OS Keyring (`get_key("Meshy")`) and `MESHY_API_KEY` environment variable resolution.
- **Hi3D / HiTem3D API Update**:
  - Added `hitem3dv3.0` high-precision model (2048³ voxel geometry) and `2048` resolution option.
  - Added `3mf` slicer-ready format export for 3D printing workflows.
  - Added dual authentication: direct Bearer token (e.g., `hi3d_live_...` or single tokens) in addition to legacy `AccessKey:SecretKey`.
  - Added OS Keyring (`get_key("Hi3D")` / `get_key("HiTem3D")`) and `HI3D_API_KEY`/`HITEM3D_API_KEY` environment variable resolution.
  - Improved model URL and download endpoint resolution across API updates.
- **Blender 4.5 & 5.0 Compatibility**:
  - Implemented dynamic Blender Foundation directory scanning, automatically selecting the newest installed version (including Blender 5.0 and Blender 4.5).
  - Modernized `trimesh` Scene geometry extraction (`to_geometry()` with fallback).
  - Bumped Blender Sync addon `bl_info` to `(2, 1, 0)` with verified Blender 4.x/5.0 shader mappings.
- **Workflows & Testing**:
  - Updated all workflows (`Geekatplay_Tripo_3D_Workflow.json`, `Geekatplay_HiTem3D_Workflow.json`, `Geekatplay_Meshy_3D_Workflow.json`) with latest recommended model defaults.
  - Added `tests/conftest.py` providing complete offline ComfyUI environment mocking.
  - Expanded test coverage across all integrations to 32 passing unit tests.

## 2.0.0

- Migrated API credentials from reversible file obfuscation to the operating-system credential vault.
- Added automatic migration of legacy credentials and password-style node inputs.
- Rebuilt the credential-manager frontend with in-place refresh, error notifications, and protected secret entry.
- Upgraded Tripo integration from v2 to v3, including current file upload, generation, task, rigging, and animation endpoints.
- Added Tripo H3.1 and P1 models, current quality controls, deterministic seeds, and expanded animation presets.
- Corrected Meshy OpenAPI endpoints and implemented its required Text-to-3D preview/refine sequence.
- Added Meshy 6, Smart Topology, PBR, HD texture, lighting removal, and safer model downloads.
- Added HiTem3D 2.1 and Scene Portrait 2.1, current resolution modes, PBR, USDZ, and multi-view bitmaps.
- Added authenticated Ollama Cloud support, request timeouts, keep-alive, and a current multimodal default.
- Updated supported dependencies for Python 3.10 through 3.12.
- Added offline contract tests for Meshy and Tripo.
