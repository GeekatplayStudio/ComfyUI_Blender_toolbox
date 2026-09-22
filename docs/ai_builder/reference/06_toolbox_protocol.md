# ComfyUI-Blender-Toolbox protocol (how scripts, sessions and the addon fit together)

## Session folder
`<ComfyUI>/output/ai_scene_builder/<session_name>/`
- `session.json` - turns (instruction, script, result path, summary, ok), plan, reference brief,
  last scene probe, last validation.
- `scene.blend` - the scene. The single source of truth between steps and between ComfyUI runs.
- `scripts/step_NNN_attemptK.py` - every generated script, with a header naming session/step/model.
- `results/step_NNN_attemptK.json` + `.job.json` - what was executed and what happened.
- `renders/step_NNN_attemptK.png` - preview renders.
- `refs/` - reference images given to the analyzer. `ai_exec_log.txt` - live-mode log (addon).

## Step lifecycle (nodes/ai_builder/agent.py)
1. Retrieve reference chunks from `docs/ai_builder/reference/*.md` for the instruction (RAG).
2. Ask the model for one complete script (prompts.py: CODEGEN_SYSTEM / CODEGEN_USER).
3. Save the script, run the safety scan (safety.py). Blocked -> ask the model to fix, never execute.
4. `dry_run` -> stop here. Otherwise execute headless (`blender -b --factory-startup --python
   step_runner.py -- job.json`) or live (`AI_EXEC:<job.json>` to the addon socket, port 8119).
5. Result JSON: `ok, error, traceback, stdout, built[], validation, scene, render_path, saved_blend`.
6. Validation errors or exceptions -> feedback to the model, retry up to `max_retries`.
7. Record the turn in session.json; the next step sees the new scene probe and history.

## Scene probe fields available to the model
`objects[{name,type,location,rotation_deg,dimensions,collection,vertices,faces,materials,modifiers}]`,
`collections[]`, `materials[]`, `lights[{name,type,energy,color}]`, `cameras[]`, `active_camera`,
`world{color,strength,hdri}`, `images[{name,filepath,packed,size,users}]`, `bbox{min,max}`,
`blender_version`.

## Validation severities
- error (step fails): non-manifold edges, loose vertices/edges, zero-area faces, flipped normals,
  missing/broken image textures, Python exception.
- warning (reported): boundary edges (open mesh), n-gons, duplicate vertices, default names, no
  material, no UV with image textures, unused images/materials, no camera, no light.
Auto-fixes (when enabled by the node): recalc normals outward, merge doubles, delete loose geometry.

## Live mode addon commands (blender_toolbox_addon.py socket listener)
Existing: `<image path>` (HDRI), `HEIGHTMAP:...`, `TEXTURE_UPDATE:...`, `MODEL:<glb>`, `LIGHTING:...`.
New: `AI_EXEC:<job.json path>` - executes the job's `code_path` in the running Blender ONLY when the
addon preference "Allow AI code execution" is on; writes `result_path`; appends to `ai_exec_log.txt`.
The runner in live mode does not save the user's file unless `live_save` is enabled.

## Environment available to scripts
`GAP_SESSION_DIR` environment variable = session folder (informational; scripts may not write files).
`sys.path` includes `blender_scripts/ai_builder/` so `gap_helpers`, `scene_probe`, `validate_scene`
are importable. Scripts can call `validate_scene.validate_scene()` themselves to self-check before
the runner does.
