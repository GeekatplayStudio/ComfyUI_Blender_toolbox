# ComfyUI-Blender-Toolbox

A comprehensive suite of ComfyUI nodes designed for **3D Generation**, **Blender Synchronization**, **PBR Texturing**, and **HDRI Creation**.

## 🚀 Features at a Glance

*   **AI 3D Model Generation**: Native support for **Tripo3D** (v3, P2 quad topology), **Meshy** (meshy-7), and **Hi3D / HiTem3D** (v3.0 2048³ voxel). Generate high-fidelity 3D models from text prompts or single/multi-view images directly in ComfyUI.
*   **AI 3D Animation & Rigging**: Automatically rig and animate characters using **Tripo3D's** animation API (Walk, Run, Dance, etc.).
*   **PBR Material Extraction**: Turn any image into High-Quality PBR maps (Albedo, Normal, Roughness, Depth, Metallic) using **Ubisoft CHORD** AI.
*   **360° Workflow**: Tools to resize, heal seams, and generate masks specifically for equirectangular images.
*   **Seamless Tiling**: Two methods—Image-based edge blending and Model-based circular padding for true seamless generation.
*   **Blender Bridge**: Live preview of your HDRI skies, Terrain heightmaps, and 3D Models directly in Blender with dynamic auto-detection for **Blender 5.0, 4.5, 4.x, and 3.6 LTS**.
*   **Round-Trip Sync**: **NEW!** Send meshes/UVs from Blender to ComfyUI, texture them with AI, and send them back to Blender instantly.
*   **Mesh Prep & Auto-Rig Export**: Local tools to Voxel Remesh, Decimate, and Export clean FBXs ready for external tools like Mixamo or AccuRig.
*   **Ollama Vision**: Analyze images and suggest lighting/sun positions using local LLMs.
*   **AI Scene Builder**: **NEW in 2.2!** Describe a scene (and drop in up to three reference images) and a local Ollama model — or Claude / OpenAI — writes and runs the Blender Python that models, textures and lights it. Reference images are read in three focused vision passes; sizes come back in real meters. Multi-pass ("now add a bridge"), multi-step plans, automatic validation of polygons / normals / textures / names, self-repair retries with the correct API signature, preview renders, and a **visual refiner** that compares the render to your reference and fixes the differences. Models default to **`auto`** — the strongest one installed is chosen for you. Every prompt, reply, script and error is written to a per-session **debug log**. Fully transparent: every script is saved before it runs. See [docs/ai_builder/AI_SCENE_BUILDER.md](docs/ai_builder/AI_SCENE_BUILDER.md).
*   **Live Blender Bridge**: Check from ComfyUI that Blender is open and the addon is current, then push a finished scene straight into your open session with materials, collections, lights and cameras intact.

> ⚠️ **AI Scene Builder security**: it executes model-generated Python inside Blender. There is **no sandbox**. Scripts are saved to `output/ai_scene_builder/<session>/scripts/` before execution, a safety scan blocks file/network/process access, `dry_run` lets you read before running, and live execution inside your open Blender is **off** until you enable it in the addon. Read the [security section](docs/ai_builder/AI_SCENE_BUILDER.md#security--read-this-first) before use.

---

## 📦 Installation

### 1. Install ComfyUI
Ensure you have [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed.

### 2. Clone Repository
Navigate to your `ComfyUI/custom_nodes/` folder and run:
```bash
git clone https://github.com/GeekAtPlay/ComfyUI_Blender_toolbox
```

### 3. Install Dependencies & Models
This suite contains standard nodes and the advanced AI PBR Extractor. The standard dependencies are installed automatically by ComfyUI Manager.

For a manual installation, use the Python interpreter that launches ComfyUI:

```bash
python -m pip install -r requirements.txt
```

Version 2.1 requires Python 3.10 or newer. Restart ComfyUI after installing or upgrading.

### 4. AI Scene Builder setup (Blender + local models)

```bash
python installer/install_ai_builder.py --smoke-test
```
(or `installer\install_ai_builder.bat`). Detects Blender 4.x/5.x, starts Ollama, pulls `qwen2.5-coder:14b`, `qwen2.5vl:7b` and `nomic-embed-text` (change with `--code-model` / `--vision-model` / `--embed-model`, skip with `--skip-models`), indexes the reference docs and builds + validates + renders a test cube in headless Blender. `install.bat` runs it as part of the normal install.

### API authentication

Add a **Credential Manager (OS Keyring)** node and use its **Save credential** button. Secrets are stored by the operating-system credential vault (Windows Credential Manager, macOS Keychain, or the configured Linux Secret Service), not in workflow JSON.

Recommended credential names are `Meshy`, `Tripo`, `HiTem3D`, and `Ollama Cloud`. Connect the manager's `api_key` output to the corresponding service node. Existing `geekatplay_keystore.enc` credentials are migrated to the OS vault once and the old file is renamed with a `.migrated` suffix.

Tripo can also read `TRIPO_API_KEY` from the environment. Ollama requires no key for a local `http://localhost:11434` server; direct `https://ollama.com` access accepts a connected API key.

#### **Required AI Models**
Some workflows require specific models to be placed in your ComfyUI folders.

| Model | Path | Description | Download |
| :--- | :--- | :--- | :--- |
| **PBR Extractor** | `models/ubsoft_pbr/chord_v1.safetensors` | Generates PBR maps (Albedo, Normal, etc). | [HuggingFace](https://huggingface.co/Ubisoft/ubisoft-laforge-chord) |
| **360-HDRI LoRA (Flux)** | `models/loras/human_360diffusion_lora_flux_dev_v1.safetensors` | For generating 360° panoramas with Flux. | [HuggingFace](https://huggingface.co/ProGamerGov/human-360-lora-flux-dev) |
| **360 Redmond (SDXL)** | `models/loras/360redmond_sdxl_v1.safetensors` | The best 360° panorama LoRA for SDXL 1.0. | [Civitai](https://civitai.com/models/118025/360redmond-a-360-view-panorama-lora-for-sd-xl-10) |
| **Seamless Texture LoRA** | `models/loras/seamless_texture.safetensors` | For Flux seamless tile generation. | [HuggingFace](https://huggingface.co/gokaygokay/Flux-Seamless-Texture-LoRA/tree/main) |

**⚠️ Note on PBR Extractor (Ubisoft CHORD):**
The `chord_v1.safetensors` model is **Gated**.
The automatic installer will *attempt* to download it, but if it fails (due to lack of HuggingFace Login), you must:
1.  Go to [Ubisoft LaForge CHORD on HuggingFace](https://huggingface.co/Ubisoft/ubisoft-laforge-chord) and accept the license.
2.  Download `chord_v1.safetensors` manually.
3.  Place it in your ComfyUI folder at: `ComfyUI/models/ubsoft_pbr/chord_v1.safetensors`.

**IMPORTANT: To enable the PBR Extractor (Ubisoft CHORD), you must run the installer manually:**

**Windows:**
Double-click `installer\install_pbr_extractor.bat`.

**Manual / Linux / Mac:**
```bash
python installer/install_pbr_extractor.py
```

> **License Note**: The Ubisoft CHORD model is gated. You must accept the license at [Hugging Face](https://huggingface.co/Ubisoft/ubisoft-laforge-chord).
> If the installer fails to download the model due to authentication, download `chord_v1.safetensors` manually and place it in `ComfyUI/models/ubsoft_pbr/`.

---

## 📂 Included Workflows
Inside the `workflows/` folder you will find 20 production-ready JSON workflows. All of them are
verified on every release — `tools/check_workflows.py` validates node types, link integrity and
that saved widget values still match the current node definitions, and the test suite fails if any
workflow breaks:

```bash
python tools/check_workflows.py --server http://127.0.0.1:8188
```

### 🤖 AI Scene Builder (2.2)
*   **`Geekatplay_AI_Scene_Builder_Complete.json`**: Reference images → brief → step plan → complete multi-step build with validation, retries and report.
*   **`Geekatplay_AI_Single_Step_From_References.json`**: Reference images → one validated build step.
*   **`Geekatplay_AI_Step_Builder_Conversational.json`**: Type an instruction, queue, look, type the next one — multi-pass building on the same scene.
*   **`Geekatplay_AI_Script_Review_Then_Run.json`**: Dry run → read the generated script → paste into the runner → execute. The safest flow.
*   **`Geekatplay_AI_Build_And_Send_To_Blender.json`**: Check the bridge, build headlessly, then load the finished scene into the Blender you have open.
*   **`Geekatplay_AI_Scene_Validator.json`**: QA + auto-fix + preview render for any session scene.

![AI Scene Builder complete workflow](docs/images/ai_scene_builder_complete.png)

### 🌟 New & Featured
*   **`Geekatplay_Hunyuan3d_v2.1.json`**: **(New)** State-of-the-art Image-to-3D generation using Hunyuan3D v2.1.
*   **`Geekatplay_Flux2_Heightmap_Generator.json`**: **(New)** Generate 16-bit precision terrain heightmaps using Flux.
*   **`Geekatplay_Heightmap_To_Blender.json`**: **(New)** Bridge to visualize heightmaps in Blender instantly.
*   **`Geekatplay_Flux_360_HDRI_Updated.json`**: **(New)** Create 360° Panoramic HDRIs with Flux.

### 3D Generation
*   **`Geekatplay_Tripo_3D_Workflow.json`**: Fast Text/Image-to-3D via Tripo API.
*   **`Geekatplay_Meshy_3D_Workflow.json`**: High-Quality 3D via Meshy API.
*   **`Geekatplay_HiTem3D_Workflow.json`**: Single/Multi-view generation via HiTem3D.

### 360° HDRI & Terrain
*   **`Geekatplay_Flux_360_HDRI.json`** / **`_Updated.json`**: 360° panoramic HDRIs with Flux.
*   **`Geekatplay_SDXL_360_HDRI.json`**: 360° HDRI pipeline with SDXL, Ollama lighting estimation and Blender sky preview.

### Texturing & Materials
*   **`Geekatplay_PBR_Texture_Studio_workflow.json`**: Extract Albedo, Normal, Roughness, Metallic from any image (Ubisoft CHORD).
*   **`Geekatplay_Blender_RoundTrip_Sync.json`**: Send generic meshes from Blender -> ComfyUI -> Texturing -> Blender.
*   **`Geekatplay_Blender_RoundTrip_Simple.json`**: Minimal round trip: generate, preview in Blender, receive the PBR set back.
*   **`Geekatplay_texture_sdxl_seamless_workflow.json`**: Generate seamless textures with SDXL.

### Automation
*   **`autorig_api.json`**: API-format template used by `blender_scripts/auto_rigger_cli.py` — image → Hunyuan3D mesh → Blender clean + rig → rigged GLB.

---

## �📚 Node Reference Guide

### 🧱 PBR & Texture Tools

#### **PBR Extractor (Ubisoft CHORD)**
*Category: `Geekatplay Studio/PBR`*
Extracts full PBR material maps from a single image using the state-of-the-art **Ubisoft LaForge CHORD** model.
- **Inputs**: `albedo_image`.
- **Outputs**: `Albedo`, `Normal`, `Roughness`, `Depth`, `Metallic`.
- **Fallback**: If the model is missing, it automatically switches to a lightweight algorithmic mode (Sobel/Luminance) so your workflow never breaks.

#### **Save Material (PBR)**
*Category: `Geekatplay Studio/Core`*
Batch saver for PBR maps. Saves all connected maps (Albedo, Normal, Roughness, etc.) into a dedicated subfolder with standardized naming.
- **Inputs**: All map types + `folder_name` (e.g., "MyTexturePack").

#### **Channel Packer**
*Category: `Geekatplay Studio/Core`*
Combines 3 or 4 grayscale images into a single RGB(A) image. Essential for Game Engine workflows (ORM textures).
- **Structure**: Red, Green, Blue, Alpha inputs.

#### **Image Comparator**
*Category: `Geekatplay Studio/Core`*
Simple utility to view two images side-by-side or vertically to compare changes.

#### **Simple PBR Generator**
*Category: `360_HDRI`*
A lightweight alternative to CHORD. Generates basic Normal and Roughness maps using image processing algorithms.

#### **Texture Scrambler (Style Transfer)**
*Category: `360_HDRI/Utils`*
Randomizes texture phase to "scramble" structure while keeping style. Useful for style transfer inputs.

---

### 🔄 Seamless Tiling Tools

#### **Seamless Tile (Simple)**
*Category: `Geekatplay Studio/Core`*
**Post-Processing**. Takes an existing image and blends the edges (Overlay or Blend mode) to make it tileable.
- Fast and effective for simple textures.

#### **Simple Seamless Tile (Model)**
*Category: `360_HDRI`*
**Generation**. Patches the Diffusion Model (U-Net) to use "Circular Padding".
- Connect this to your model *before* the KSampler.
- Makes the AI *generate* a seamless image natively.

#### **Seamless Tile (VAE)**
*Category: `360_HDRI`*
**Decoding**. Patches the VAE decoder to fix seams that appear during decoding.

#### **Heal 360 Seam**
*Category: `360_HDRI`*
**Post-Processing**. Specifically designed for Equirectangular (360°) images. Blends the left/right seam to fix "lines" in the sky.

#### **Preview Seamless Tile**
*Category: `360_HDRI`*
**Utility**. Creates a grid (default 3x3) of the input image to visually verify seamless tiling.
- **Inputs**: `images`, `tiles` (int, default 3).

---

### 🌐 360° HDRI Tools

#### **Save Fake HDRI (EXR)**
*Category: `Geekatplay Studio/360 HDRI`*
Saves an LDR image as an `.exr` file (32-bit float fake), compatible with 3D software lighting.

#### **Image to 360 Latent**
*Category: `Geekatplay Studio/360 HDRI`*
Resizes and masks latents specifically for 2:1 aspect ratio generation.

#### **Rotate 360 Image**
*Category: `Geekatplay Studio/360 HDRI`*
shifts the pixels of a 360 image horizontally (Yaw), Pitch, or Roll.

#### **Generate Pole Mask**
*Category: `Geekatplay Studio/360 HDRI`*
Creates a mask covering the top and bottom "poles" of a 360 image, useful for inpainting distortions.

---

### 🐵 Blender Integration
*Requires installing the scripts in `blender_scripts/` to your Blender addons.*

#### **Preview in Blender (360 Sky)**
*Category: `Geekatplay Studio/360 HDRI`*
Sends the image to Blender and sets it as the World Background environment automatically.

#### **Preview Heightmap in Blender**
*Category: `Geekatplay Studio/360 HDRI`*
Sends an image to Blender and displaces a plane geometry to visualize 3D terrain.

#### **Preview Model in Blender (GLB)**
*Category: `Geekatplay Studio/360 HDRI`*
Sends a `.glb` or `.gltf` file path to Blender for immediate loading.

#### **Preview Mesh in Blender (Send)**
*Category: `Geekatplay Studio/360 HDRI`*
Directly sends raw mesh data (vertices/faces) to Blender. Useful for procedural geometry nodes or converting `MESH` types.

#### **Sync Lighting to Blender**
*Category: `Geekatplay Studio/360 HDRI`*
Updates Blender's lighting creation based on estimated parameters.

---

## 🏗️ Auto-Rigger & Cleanup
Tools for cleaning AI-generated meshes and applying skeletons.

### Blender Addon Features
*   **Clean Active Mesh:** Applies Voxel Remesh and Decimate to make mesh watertight and game-ready.
*   **Quick Rigging:** Automatically adds a basic humanoid skeleton and binds the mesh.
*   **Export for Mixamo:** Preps and exports mesh for third-party rigging services.

### Testing Workflows
*   `Test_Rigging_Generation.json`: A sample workflow demonstrating Image-to-3D generation (using Hunyuan3D) ending in a saved mesh ready for the Auto-Rigger.

---

### 🦙 Ollama (Local AI) Integration
*Requires local [Ollama](https://ollama.com) installation.*

#### **Ollama Vision Analysis**
*Category: `Geekatplay Studio/Ollama`*
Uses a vision model (e.g., LLaVA) to describe an image. Great for auto-captioning or interrogation.

#### **Ollama Lighting Estimator**
*Category: `Geekatplay Studio/Ollama`*
Analyzes an image to guess the sun's position (elevation/azimuth) and color temperature.

---

### 🛠️ Prompt & Heightmap Utilities

#### **Terrain Prompt Maker**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Helper to generate rich terrain descriptions for SDXL or standard models.

#### **Flux Terrain Prompt Generator**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
New specialized prompt generator for Flux.
*   **Features**: Controls for Erosion Type (Hydraulic, Thermal, Glacial, etc.), Erosion Strength, Season, and Color Scheme (Grayscale/Rainbow).
*   **Outputs**: separate Positive and Negative prompts optimized for Flux heightmap generation.

#### **Terrain Texture Prompt Maker**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Helper for satellite-style texture prompts.

#### **Terrain HeightField Prompt Maker**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Generates prompts tuned for grayscale displacement maps (linear, non-optical).

#### **Color to Heightmap**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Converts RGB images to high-quality Grayscale heightmaps with Gamma and Level controls.

#### **Simple Heightmap Normalizer**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Ensures heightmap values span the full 0.0 - 1.0 range.

#### **Terrain Erosion Prompt Maker (Detailer)**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Adds specific erosion keywords to your prompt to simulate realistic geological weathering.
*   **Modes**: Hydraulic, Thermal, Glacial, Aeolian, Coastal, Terrace.
*   **Strength**: Controls the emphasis (syntax weighting) of the erosion effect. 0.0 disables it.

#### **Material Texture Prompt Maker (Preset)**
*Category: `Geekatplay Studio/360 HDRI/Texture`*
Generates optimized prompts for seamless textures based on a specific material type.
*   **Presets**: Over 20 types including Rock, Soil, Water, Snow, Metal, Wood, Fabric, Sci-Fi, etc.
*   **Output**: Creates distinct Positive (high resolution, seamless) and Negative (3d render, perspective) prompts.

### 🗿 Blender Integration

The suite includes a powerful **Blender Addon** (`ComfyUI 360 HDRI Sync`).

*   **Round-Trip Texturing Workflow**:
    1.  **Select Object** in Blender.
    2.  Click **"1. Send Mesh & UVs"** in the ComfyUI tab.
    3.  In ComfyUI, load the `Geekatplay_Blender_RoundTrip_Sync.json` workflow.
    4.  It automatically loads your mesh's UV layout and Albedo reference.
    5.  Generate your texture (using ControlNet, img2img, standard generation).
    6.  The result is automatically saved back to your project folder and updated in Blender instantly.

*   **One-Click Installation**: Go to `Edit > Preferences > Add-ons`, click "Install", and select `blender_scripts/with_dependencies/comfyui_360_hdri_addon_v1_1_2.zip`.
*   **Live Preview**: Send generated HDRIs or Heightmaps directly to Blender's viewport.
*   **Lighting Sync**: Sync Sun position and color from ComfyUI (via Ollama) to Blender lights.

### 🦴 Auto-Rigging & Mesh Prep (Blender)

The addon also includes a suite of tools to prepare raw ComfyUI generated meshes (e.g. from Hunyuan3D or text-to-3d) for animation.

*   **Clean Active Mesh**: Automates **Voxel Remeshing** followed by **Decimation**. This turns messy, non-manifold generated geometry into clean, riggable topology in one click.
*   **Quick Rig**: Instantly adds a basic Humanoid Metarig to your object for quick posing testing.
*   **Export for Mixamo/AccuRig**: One-click solution to export your character. It automatically centers the mesh, applies transforms, and sets the correct axis orientation (Y-Up) required by external auto-riggers like Mixamo or ActorCore AccuRig.

---

## 🧰 Geekatplay 3D Toolbox

A set of utility nodes for advanced workflow control, visual debugging, and smart resizing.

#### **3D Toolbox Smart Resizer (Geekatplay)**
*Category: `Geekatplay Studio/3D Toolbox`*
Resizes images based on *Target Model* pixel counts (SD1.5, SDXL, Flux) and *Aspect Ratio*.
*   **Model Target**: SD 1.5 (0.25MP), SDXL (1MP), Flux (2MP).
*   **Processing**: Scale, Stretch, Center Crop, or Pad.

#### **3D Toolbox Visual Comparator**
*Category: `Geekatplay Studio/3D Toolbox`*
Compares two images side-by-side with an interactive slider.
*   **Features**: Split-view slider, Zoom (0.1x to 10x), and Panning support.

#### **3D Toolbox Workflow Pauser**
*Category: `Geekatplay Studio/3D Toolbox`*
Halts the workflow execution. A "Continue" button appears on the node in the UI to resume.
*   **Usage**: Connect any signal input. Workflow pauses until you click "Continue".

#### **3D Toolbox Logic Switch**
*Category: `Geekatplay Studio/3D Toolbox`*
Routes data to Output A or Output B based on a boolean condition. Useful for branching logic.

#### **3D Toolbox Dynamic Group Manager**
*Category: `Geekatplay Studio/3D Toolbox`*
scans your workflow for Groups and adds "Enable/Disable" toggles for each.
*   **Function**: Toggling OFF creates a "Mute" effect for all nodes inside the visual group box.

#### **3D Toolbox String Viewer**
*Category: `Geekatplay Studio/3D Toolbox`*
Displays multi-line text directly on the node. Useful for debugging prompts or LLM outputs.

#### **3D Toolbox VRAM Purge**
*Category: `Geekatplay Studio/3D Toolbox`*
Forces unloading of all models and clears soft VRAM cache. Use between heavy model switches (e.g. SDXL -> Flux).

---

## 🎲 3D Generators (Tripo, Meshy, HiTem3D)

A set of nodes to integrate leading AI Text-to-3D and Image-to-3D services directly into your ComfyUI workflow.
**Note**: These services require API keys. Use the included **API Key Manager** to securely manage them.

#### **Geekatplay API Key Manager**
*Category: `Geekatplay/Authentication`*
Stores and retrieves credentials for Tripo, Meshy, HiTem3D, and Ollama Cloud.
*   **Secure storage**: Secret values live in the operating-system credential vault. Only credential names are stored by the extension.
*   **Safer UI**: Secret entry uses a password field, and the list updates without refreshing ComfyUI.
*   **Migration**: Legacy XOR-obfuscated credentials are migrated automatically.

#### **Tripo3D Model Generator**
*Category: `Geekatplay/Tripo3D`*
Generate 3D models using Tripo's v3 API with automatic endpoint failover (`openapi.tripo3d.com` and `openapi.tripo3d.ai`).
*   **Inputs**: Single image or Multi-view images (Front, Left, Back, Right).
*   **Models**: `v3.1-20260211` (flagship high fidelity), `P2-20260801` (quad topology, game-ready), `P1-20260311`, `v3.0-20250812`, and `v2.5-20250123`.
*   **Features**: PBR materials, detailed/extreme textures, geometry quality, seeds, quad output, auto-scale, and image auto-fix.
*   **Authentication**: Resolves from node input, OS Keyring (`Tripo`), or `TRIPO_API_KEY` environment variable.

#### **Tripo3D Animator**
*Category: `Geekatplay/Tripo3D`*
Uses Tripo v3 rig-check, rig, and animation-retarget endpoints.
*   **Presets**: Idle, walk, run, dive, climb, jump, combat, and quadruped walk.
*   **Rigging**: Automatic detection, biped, or quadruped with Mixamo/Tripo skeleton output.

#### **Meshy Text/Image to 3D**
*Category: `Geekatplay/Meshy`*
Access Meshy.ai through OpenAPI endpoints.
*   **Text to 3D**: `meshy-7` (latest), `meshy-6`, and `meshy-5` preview generation, with an automatic Preview → Refine sequence when textured output is selected.
*   **Image to 3D**: Standard or Smart Topology generation from a ComfyUI image (`meshy-7`, `meshy-6`, `meshy-t2`, etc.).
*   **Output controls**: Remeshing, polygon target, pose, PBR maps, HD textures, lighting removal, texture prompts, and `.3mf` / `.glb` exports.
*   **Authentication**: Resolves from node input, OS Keyring (`Meshy`), or `MESHY_API_KEY` environment variable.

#### **Hi3D / HiTem3D Generator**
*Category: `Geekatplay/HiTem3D`*
Generate production-ready models using Hi3D (formerly HiTem3D) with 3D print and game asset pipelines.
*   **Modes**: Geometry Only, Staged, All-in-One.
*   **Models**: `hitem3dv3.0` (2048³ voxel geometry), `hitem3dv2.1`, `hitem3dv2.0`, `hitem3dv1.5`, and `scene-portraitv2.1`/`v2.0`/`v1.5`.
*   **Resolution**: 2048, 1536profast, 1536pro, 1536fast, 1536, 1024, 512, with PBR output.
*   **Export Formats**: GLB, OBJ, STL, FBX, USDZ, and `.3mf` (slicer-ready for Bambu Studio, OrcaSlicer, etc.).
*   **Authentication**: Dual-auth support—direct Bearer token (e.g. `hi3d_live_...`) or legacy `AccessKey:SecretKey`, with OS Keyring (`Hi3D` / `HiTem3D`) and `HI3D_API_KEY`/`HITEM3D_API_KEY` environment resolution.

#### **Ollama Vision and Lighting**
*Category: `Geekatplay Studio/Ollama`*
Supports current multimodal Ollama models such as `gemma3`, configurable keep-alive, local servers without authentication, and Ollama Cloud with bearer authentication.

---

## 🤖 AI Scene Builder

*Category: `Geekatplay Studio/AI Scene Builder` — full documentation: [docs/ai_builder/AI_SCENE_BUILDER.md](docs/ai_builder/AI_SCENE_BUILDER.md)*

A language model writes one Blender Python script per build step; the toolbox runs it in Blender (headless by default), validates polygons / normals / textures / names, renders a preview and feeds errors back to the model for repair. The session `.blend` is the memory between passes.

![AI Step Builder conversational workflow](docs/images/ai_step_builder_conversational.png)

Example: first pass of a fully local run (`qwen2.5-coder:32b`, headless Blender 5.2) — "stone watchtower on grassy terrain with 20 pines, sun, camera". The model needed two self-repairs (traceback fed back automatically); the result passed validation (28 closed meshes, 0 topology errors) and rendered in under a second:

![Local model example render](docs/images/example_watchtower_step1_qwen32b.png)

Second pass in the same session, a separate ComfyUI run: "add a low wooden palisade fence around the tower at ~8 m radius with a south opening and a dirt path to the edge". The model saw the existing scene probe and added 52 fence posts/rails plus the path without touching the tower (80 meshes, validation passed):

![Local model second pass](docs/images/example_watchtower_step2_fence_qwen32b.png)

Quality scales with the model: a 32B local coder gets you this in ~7 minutes per pass on an RTX 3090 including two self-repairs; `claude-sonnet-5` / `claude-opus-5` via the Anthropic provider produce far richer geometry and materials with fewer retries.

| Node | Purpose |
| :--- | :--- |
| **AI Scene Session** | Names the scene; creates/continues `output/ai_scene_builder/<name>/` (scene.blend, scripts, results, renders, refs, session.json). |
| **AI Model Config** | Provider (`ollama` local/cloud, `anthropic`, `openai_compatible`), code model, vision model, URL, temperature, context, API key (or API Key Manager). |
| **AI Reference Analyzer (Multi-Image)** | Up to three image inputs + your text → one structured build brief (layout, elements with sizes, materials, lighting, camera). |
| **AI Scene Planner** | Prompt (+ brief, + current scene) → ordered step plan; or your own `manual_plan`. |
| **AI Scene Builder (Complete Scene)** | Runs every plan step: generate → safety scan → execute → validate → retry → preview. |
| **AI Step Builder (Conversational)** | One instruction per run, building on the existing scene. |
| **AI Script Runner (Review & Execute)** | Executes a script you reviewed/edited (from a `dry_run`). |
| **AI Visual Refiner** | Renders the scene, compares it side by side with your reference image, lists what is wrong (silhouette, missing parts, shapes, placement, materials, detail) and executes the fixes. Repeats until the resemblance score hits your target. |
| **Blender Bridge Check** | Full round trip to a running Blender: its version, the addon version, the open file, the scene contents, and whether live AI execution is allowed. Outputs `connected` / `live_exec_allowed`. |
| **Send Scene to Blender** | Loads a finished `.blend` into the Blender you already have open — append (default, keeps your work), link, or open. Materials, collections, lights and cameras come across intact. |
| **AI Scene Validator** | Checks non-manifold edges, loose geometry, zero-area faces, flipped normals, duplicate vertices, missing textures, default names, camera/lights; optional auto-fix; `passed` output. |

**Models**: local Ollama `qwen2.5-coder:14b` (default) / `qwen2.5-coder:32b` (best local) / `qwen3-coder:30b` for code, `qwen2.5vl:7b` or `qwen3-vl:4b` for references, `nomic-embed-text` for retrieval; or `claude-sonnet-5` / `claude-opus-5` via Anthropic, or any OpenAI-compatible endpoint.

**Live mode**: to build inside the Blender you are looking at, start the toolbox addon listener and switch **AI Scene Builder (Live) → Allow AI code execution** ON in the addon's ComfyUI sidebar panel. Headless mode never needs this switch.

**Transparency**: prompts in `nodes/ai_builder/prompts.py`, reference docs the model reads in `docs/ai_builder/reference/`, helper library in `blender_scripts/ai_builder/gap_helpers.py`, every generated script and result on disk. No hidden behaviour.

---

## 👤 Credits

**Geekatplay Studio — Vladimir Chopine.** Toolbox design, Blender integration, AI Scene Builder.

## 📄 License
(c) Geekatplay Studio.
Ubisoft CHORD model follows its own license (Research-Only Copyleft).
Other components MIT.
