# Research: AI-Driven Blender Scene Building (Prompt/Image → Modeled, Textured, Lit Scene)

Status: research only, no implementation yet. Compiled 2026-09-22.

## 1. The question, restated

Can this toolbox add a workflow where a user describes what they want (text + optional
reference image), and the system performs the actual Blender modeling, texturing, and
lighting — with two extra requirements:

- **Multi-pass / conversational**: "build this forest" → later "now add that building to
  the same scene" → later "now add fog" — each turn builds on the previous scene, not a
  fresh one.
- **Multi-step / scripted**: a single complex request ("build a fantasy village with a
  castle, a market, and a river") gets decomposed into an ordered sequence of sub-builds,
  each executed and checked before the next runs.

Short answer: **yes, and this project already has most of the required infrastructure.**
The missing piece isn't a new protocol — it's an LLM-authored-Python execution loop with
a validation gate, wired to the headless-Blender pattern this repo already uses for the
auto-rigger.

## 2. What "Astra" actually is (worth clearing up first)

There is no Blender-specific product called Astra. **OpenAI's GPT-6 Astra** is a general
frontier model released Sept 2026 with strong computer-use/coding ability; third-party
blogs have published unofficial "drive Blender with Astra" tutorials that just point its
general coding ability at a Python console — same mechanism as pointing any strong coding
model at Blender. Separately, **Google DeepMind's Project Astra** is an unrelated
real-time multimodal voice/video assistant with no Blender tie-in. The Tesla Castle
example at `D:\Desktop-Stuff-Dump\TeslaCastle` wasn't produced by a live "Astra session"
either — its own logs show it was built by **this project's own Blender addon**
(`[ComfyUI-360] Addon Registered Successfully` appears in every log) via a headless
build→texture→validate→render pipeline (see §3). The naming is a coincidence; the
mechanism is the thing to copy, and it's not exotic.

## 3. Two architectural paradigms (both real, serving different needs)

### A. Live MCP tool-calling (ahujasid/blender-mcp, and the Blender Foundation's own
experimental `blender.org/lab/mcp-server`)

A Blender add-on runs a socket server inside a **running** Blender instance. An MCP
client (Claude Desktop, Cursor, etc.) calls tools like `get_scene_info`,
`get_viewport_screenshot`, and — the powerful/dangerous one — `execute_blender_code`,
which `exec()`s raw Python with full `bpy` access. Documented limitations:

- Every operation is a separate socket round trip; complex builds become hundreds of
  small calls, each adding latency and ~5–7K tokens of tool-schema overhead per request.
- No persistent Python state between calls — **the `.blend` file is the only durable
  state**, which turns out to be the right design principle to keep regardless of
  architecture (see §5).
- Heavy operations (dense remesh, 4K bake, GLTF export) can exceed the bridge's call
  timeout.
- No native sandboxing — `execute_blender_code` runs whatever the model wrote.

This paradigm is what most "watch it build live in the viewport" demos use. It's good for
conversational, small, interactive edits inside a GUI session a person is watching.

### B. Headless batch-script pipeline (what this project already implements, and what
academic work like **SceneCraft**, arXiv 2403.01248, formalizes)

The LLM writes a **complete, self-contained Python script**, which gets executed via
`blender --background --python script.py`, with the caller reading back stdout/log/JSON
for the next decision — no live GUI, no per-statement round trip. This repo's
[`blender_scripts/auto_rigger_cli.py`](../blender_scripts/auto_rigger_cli.py) already does
exactly this (`subprocess.run([blender_cmd, "--background", "--python", BLENDER_SCRIPT,
"--", ...])`), and the Tesla Castle example is a textbook instance of the *full* pipeline:
`build_castle.py` (procedural modeling, logging `BUILT <name> <vertex-count>` per object)
→ `make_textures.py` (procedural 8K PBR maps, no diffusion model needed) →
`mesh_validation.json` + `check_delivery.py` (automated topology/UV/packing QA gate) →
`render_review.py` (multi-camera renders). Each stage is a discrete, inspectable
checkpoint — precisely the "self-verification loop" SceneCraft describes academically
(inner loop: script → render → critique → refine; outer loop: reusable skill library).

**This is the paradigm to build on.** It scales to genuinely complex builds (the castle
is 543K vertices, 33 closed objects, zero topology defects — verified by a strict JSON
report, not a human eyeballing a viewport), it gives the LLM crisp pass/fail feedback
instead of a screenshot to interpret, and this project already has the subprocess
plumbing, just not yet wired to an LLM.

**Recommendation: don't choose one paradigm — use both, for different jobs.** Extend the
existing live socket bridge (currently a fixed protocol: `HEIGHTMAP:`, `MODEL:`,
`TEXTURE_UPDATE:`, etc. — see `blender_toolbox_addon.py`) with a narrow, logged
`EXEC_PYTHON:` command for small conversational nudges to a session the user is watching
live; use the headless batch pattern (extending `auto_rigger_cli.py`'s approach) for the
"build a complex thing / run many steps" case, since it isn't bound by the socket
bridge's timeout or GUI-thread constraints.

## 4. Local LLM recommendations (via Ollama)

The toolbox already has Ollama vision nodes (`OllamaVision`, `OllamaLightingEstimator`,
defaulting to `gemma3`), but both are single-shot `/api/generate` calls — no tool-calling,
no chat history. A code-generation + agentic-loop capability would be new.

**Code generation, with native tool-calling** (needed if you want the model to call
discrete tools like `execute_python`/`get_scene_info` rather than free-texting one giant
script):
- **Qwen3-Coder-30B-A3B** — MoE with only ~3B active params, so it runs at a fast/light
  footprint despite the "30B" label; built specifically for agentic tool-calling loops
  (Qwen Code, Cline). Best fit for the tool-loop architecture.
- **Qwen2.5-Coder-14B/32B** — strongest raw bpy-code correctness in the practically
  runnable range (32B: 92.7% HumanEval, ahead of GPT-4's 87.1%); native Ollama
  tool-calling support since 2.5.
- **DeepSeek-Coder-V2-Lite (16B, ~2.4B active)** — budget option, fits 10–12GB VRAM,
  tool-calling less mature.
- **Devstral Small (24B)** — edges ahead on tool-call accuracy for multi-edit agentic
  sequences in 2026 benchmarks.

**Vision, for reference-image grounding** (extract composition/materials/layout from a
reference image before planning a build):
- **Qwen2.5-VL-7B** — already Ollama-supported, strong general spatial grounding; natural
  pairing with Qwen-Coder for a same-family code+vision stack.
- **InternVL3.5** (1B–8B variants) — now edges out Qwen2.5-VL on fine-grained
  compositional benchmarks (MMStar/MMVet) while staying small.
- Current default `gemma3` is a fine generalist but not benchmark-leading here — worth
  adding one of the above alongside it rather than replacing it outright.

## 5. Does a Blender-specific model already exist? (yes — directly relevant to the LoRA question)

Two published research efforts already target exactly this:

- **BlenderLLM** (arXiv 2412.14203, open weights on
  [HuggingFace](https://huggingface.co/FreedomIntelligence/BlenderLLM),
  [GitHub](https://github.com/FreedomIntelligence/BlenderLLM)) — full fine-tune of
  **Qwen2.5-Coder-7B-Instruct** on ~8K instruction→script pairs (`BlendNet`), refined with
  an iterative self-improvement loop: generate → execute in Blender → validate → retrain.
  It beats OpenAI o1-preview and CodeLLaMA-7B-Instruct on their own CADBench benchmark.
  Weights are open; getting it into Ollama would require a manual GGUF conversion.
- **BlenderRAG** (arXiv 2605.00632) — the retrieval-augmented alternative to fine-tuning,
  same goal, no training required. Validates the "local reference doc" idea in §6 directly
  — this is the same approach, published.

No standalone Blender-targeted LoRA adapters for Qwen/Llama were found on HuggingFace or
Civitai; BlenderLLM's full fine-tune appears to be the only published Blender-specific
model. That's a usable head start if fine-tuning is pursued later (§7), rather than a
from-scratch effort.

## 6. Is a local reference doc / RAG for bpy + this project's own commands worth it?

**Yes, clearly**, for two independent reasons:

1. Blender's Python API drifts between versions — the Tesla Castle build log already
   shows `DeprecationWarning: 'Material.use_nodes' is expected to be removed in Blender
   6.0` on Blender 5.2. A general-purpose model's pretraining knowledge of bpy will
   silently lag whatever Blender version is actually installed; grounding generation in a
   curated, version-pinned reference cuts hallucinated/removed API calls.
2. This project's **own** protocol (the `blender_toolbox_addon.py` socket commands, the
   `geometry_ops.py` node set, the Mesh-building helper pattern used in
   `build_castle.py`) isn't something any pretrained model has seen. Any agent generating
   code that's supposed to cooperate with this toolbox's existing nodes/addon needs that
   as retrieval context, not just generic bpy docs.

BlenderRAG (§5) is direct academic validation that this approach works and can match a
fine-tuned model's results without training. Recommended content for the local doc/index:
a curated bpy API subset scoped to what this project's builds actually use (mesh/bmesh,
materials/shader nodes, modifiers, compositor, camera/render settings — i.e., roughly the
surface area `build_castle.py` and `render_review.py` already exercise), plus this
project's own socket protocol and node reference pulled from the existing README.

## 7. Is training a custom LoRA worth it?

**Not as a first step — but with a concrete, low-risk path to get there.** BlenderRAG's
existence is evidence that a strong general coder (Qwen2.5-Coder-14B/32B, or cloud Claude
for the hardest passes) plus good retrieval context gets most of the value with none of
the training cost or iteration latency. Recommended sequencing:

1. Ship the RAG + agentic-loop system first (§3B architecture + §6 reference doc).
2. **Log every successful run**: (prompt, generated script, execution result, validation
   report) — the project's own `mesh_validation.json`-style JSON gate is exactly the
   signal needed to auto-filter for only execution-validated, high-quality examples.
3. Once a few hundred to ~1,000+ validated examples accumulate, LoRA fine-tune — this is
   literally BlenderLLM's own self-improvement loop, applied to this project's specific
   node vocabulary instead of general CAD.
4. Tooling: **Unsloth** (fastest, lowest VRAM, best starting point for a single-GPU QLoRA
   run), Axolotl (multi-GPU), LLaMA-Factory (flexible CLI/UI). Hardware: 7B–14B QLoRA
   fine-tuning is feasible on a single 24GB consumer GPU (3090/4090).

This turns the LoRA question from "should we speculatively train one" into "we'll have
the training data as a byproduct of shipping the RAG system, and can fine-tune once it's
worth it" — lower risk, and it starts from BlenderLLM's proven recipe rather than
inventing one.

## 8. Where should the agent loop actually live — inside ComfyUI's graph, or beside it?

Researched ComfyUI's own capacity to host this, specifically because the multi-pass and
multi-step requirements both need state that survives across separate executions, and a
stock ComfyUI graph is a **stateless DAG evaluated once per Queue Prompt** — its only
persistence hook (`IS_CHANGED`) is for cache invalidation, not session memory. Every
real "remember the last turn" implementation found in the ComfyUI ecosystem
(`ApoStudio`'s History node, community "LLM Session Chat" node) bolts state onto the
filesystem via a JSON session file the user keeps pointing subsequent runs at, because
the graph engine itself has no session concept — even ComfyUI's 2025 Subgraph feature
addressed packaging/reuse, not iteration.

There's also a concrete gotcha: **`subprocess` calls from custom nodes are restricted by
ComfyUI itself** unless the node is installed manually (not via Manager) or ComfyUI is
launched with `--allow-subprocess` (per `JayLyu/blender-in-comfyui`'s documentation of
this exact issue). This project's existing headless-Blender pattern sidesteps the problem
entirely — `auto_rigger_cli.py` is a **standalone CLI**, not a registered ComfyUI node; it
talks to ComfyUI over HTTP (`/prompt`, `/upload/image`) and to Blender via `subprocess`
from outside ComfyUI's node-execution process. That's the model to keep following: don't
try to express the plan→generate→execute→validate→iterate loop as ComfyUI graph
nodes/edges. Hide the whole loop inside one Python node (or an external orchestrator CLI,
matching the existing `auto_rigger_cli.py` precedent) that internally owns session state
(a JSON file, ApoStudio-style) and exposes only coarse I/O to the graph — prompt and
reference image in; resulting preview render, `.blend` path, and status text out.

Also worth registering as reusable rather than reinventing: **`ComfyUI_LLM_party`**
(`heshengtao/comfyui_LLM_party`) already bundles an MCP server and a code-interpreter
tool that lets an LLM generate and auto-run Python — evaluate it before building an LLM
chat/tool-calling framework from scratch.

## 9. Security note (don't skip this when implementing)

Every source examined — the official Blender Foundation MCP server's own docs included —
flags the same thing: there is **no sandboxing**; `execute_blender_code`-style tools and
this project's proposed headless-script pattern both run LLM-authored Python with full
filesystem/process access. ComfyUI's own custom-node trust boundary is already weak
(arbitrary Python execution is the norm), which compounds when the code being executed is
itself model-generated rather than author-reviewed. When this gets built, it should follow
the same posture this session operates under: default to showing the generated script
before executing it (or gating the first execution of any given session behind explicit
confirmation), and treat headless/background execution as the mode you opt into once
you trust the loop, not the default from message one.

## 10. Claude model recommendation (for doing this work, not for the target LLM)

- **Research** (this phase): **Opus 5, high or xhigh effort.** Best pure reasoning model
  for weighing tradeoffs across several domains (3D/Blender internals, LLM tooling,
  fine-tuning economics, ComfyUI's execution model) at once. Sonnet 5 — what actually did
  this research — is a solid cheaper alternative and handled it fine; Opus is the upgrade
  if you want the deepest single pass.
- **Build** (implementation phase): **Sonnet 5, default/high effort.** Anthropic's
  flagship agentic-coding model — exactly suited to the iterative
  write-node/run-Blender/read-log/fix loop this feature needs. Escalate a specific hard
  sub-problem (e.g., tuning the self-verification loop's prompt/validation design) to
  Opus 5 high effort only if you get stuck; default to Sonnet for the bulk of it.
- **Fable 5.1**: not a fit for this. Its naming and apparent tuning point at creative/
  narrative writing, not code or architecture research. Possible minor, optional role
  later: generating vivid creative-brief scene descriptions for a coding model to turn
  into a technical build plan — not core to the technical work.

## 11. Summary architecture sketch (conceptual — not an implementation plan)

```
User prompt + optional reference image
        │
        ▼
[Session state] ── JSON file per scene/session (ApoStudio-style), holds:
        │            chat/turn history, path to the live .blend, last validation report
        ▼
[Planner] ── LLM decomposes complex asks into an ordered sub-task list
        │      (SceneCraft/CityX-style staged plan; trivial pass-through for simple asks)
        ▼
[Per sub-task loop]
   1. Query current scene state (light "describe scene" script → object/collection list,
      materials, cameras — same idea as blender-mcp's get_scene_info)
   2. RAG: pull relevant bpy + this-project's-own-protocol reference chunks (§6)
   3. LLM (local Ollama coder, or cloud Claude for hard passes) generates a Python script
      for this sub-task only — incremental, not a full scene rewrite
   4. Execute:
        - Live socket bridge (extend blender_toolbox_addon.py, new EXEC_PYTHON: command)
          for interactive/GUI-visible small edits, OR
        - Headless `blender --background --python` (existing auto_rigger_cli.py pattern)
          for heavy/batch sub-tasks
   5. Validate: mesh_validation.json-style topology/UV/asset checks; re-render a preview
   6. On failure: feed the error/validation report back to the LLM for one retry;
      on success: save the .blend (state persists here, not in chat context) and proceed
        │
        ▼
Result surfaced back into the ComfyUI graph: preview image, updated .blend path, status
```

Next request in a later turn re-enters at "Query current scene state" against the same
`.blend` — that's what makes "now add a building to that forest" work without re-deriving
the whole scene from chat history.

## Sources

- blender-mcp: https://github.com/ahujasid/blender-mcp ·
  https://mcp-for-blender.com/concepts/how-it-works ·
  https://github.com/ahujasid/blender-mcp/issues/347 (token cost)
- Official Blender Foundation MCP server: https://blender.org/lab/mcp-server
- BlenderGPT: https://github.com/gd3kr/BlenderGPT · Blender Copilot:
  https://github.com/pramishp/BlenderCopilot
- SceneCraft: https://arxiv.org/abs/2403.01248
- CityX: https://arxiv.org/pdf/2407.17572
- Planner-Actor-Critic (2026): https://arxiv.org/pdf/2601.05016
- blender-cli (headless): https://github.com/renezander030/blender-cli · bpy.dev:
  https://bpy.dev/
- Qwen3-Coder: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct · Qwen2.5-Coder
  (Ollama): https://ollama.com/library/qwen2.5-coder · Ollama tool calling:
  https://docs.ollama.com/capabilities/tool-calling
- BlenderLLM: https://arxiv.org/abs/2412.14203 ·
  https://github.com/FreedomIntelligence/BlenderLLM ·
  https://huggingface.co/FreedomIntelligence/BlenderLLM
- BlenderRAG: https://arxiv.org/pdf/2605.00632
- InternVL3.5: https://arxiv.org/html/2508.18265v1 · MiniCPM-V 4.5:
  https://arxiv.org/pdf/2509.18154
- ComfyUI-Ollama: https://github.com/stavsap/comfyui-ollama · ComfyUI_LLM_party:
  https://github.com/heshengtao/comfyui_LLM_party
- ApoStudio (session/History node): https://github.com/ApoloniArt/ApoStudio
- ComfyUI subprocess restriction: https://github.com/JayLyu/blender-in-comfyui ·
  ComfyUI Blender wrapper precedent: https://github.com/IRCSS/comfyUI-blender-wrapper
- ComfyUI Subgraphs (2025): https://blog.comfy.org (subgraph release notes)
- Local project precedents referenced: `blender_scripts/auto_rigger_cli.py`,
  `blender_scripts/blender_toolbox_addon.py`, `nodes/ollama_nodes.py`,
  `D:\Desktop-Stuff-Dump\TeslaCastle\scripts\*.py` (build_castle.py, make_textures.py,
  render_review.py, check_delivery.py)
