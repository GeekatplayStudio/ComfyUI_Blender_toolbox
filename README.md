# ComfyUI-360-HDRI-Suite

A comprehensive suite of ComfyUI nodes designed for 360° HDRI panorama creation, Seamless Texture generation, PBR Material extraction (including Ubisoft CHORD support), and Blender synchronization.

## 🚀 Features at a Glance

*   **PBR Material Extraction**: Turn any image into High-Quality PBR maps (Albedo, Normal, Roughness, Depth, Metallic) using **Ubisoft CHORD** AI.
*   **360° Workflow**: Tools to resize, heal seams, and generate masks specifically for equirectangular images.
*   **Seamless Tiling**: Two methods—Image-based edge blending and Model-based circular padding for true seamless generation.
*   **Blender Bridge**: Live preview of your HDRI skies, Terrain heightmaps, and 3D Models directly in Blender.
*   **Round-Trip Sync**: **NEW!** Send meshes/UVs from Blender to ComfyUI, texture them with AI, and send them back to Blender instantly.
*   **Ollama Vision**: Analyze images and suggest lighting/sun positions using local LLMs.

---

## 📦 Installation

### 1. Install ComfyUI
Ensure you have [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed.

### 2. Clone Repository
Navigate to your `ComfyUI/custom_nodes/` folder and run:
```bash
git clone https://github.com/GeekAtPlay/ComfyUI-360-HDRI-Suite
```

### 3. Install Dependencies & Models
This suite contains standard nodes and the advanced AI PBR Extractor. The standard dependencies are installed automatically by ComfyUI Manager.

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

## 📚 Node Reference Guide

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

#### **Terrain Prompt Maker (Ollama)**
*Category: `Geekatplay Studio/360 HDRI/Terrain`*
Helper to generate rich terrain descriptions.

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

## 📄 License
(c) Geekatplay Studio.
Ubisoft CHORD model follows its own license (Research-Only Copyleft).
Other components MIT.
