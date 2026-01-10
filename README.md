# ComfyUI-360-HDRI-Suite

A comprehensive suite of ComfyUI nodes designed for 360° HDRI panorama creation, Seamless Texture generation, PBR Material extraction (including Ubisoft CHORD support), and Blender synchronization.

## 🚀 Features at a Glance

*   **PBR Material Extraction**: Turn any image into High-Quality PBR maps (Albedo, Normal, Roughness, Depth, Metallic) using **Ubisoft CHORD** AI.
*   **360° Workflow**: Tools to resize, heal seams, and generate masks specifically for equirectangular images.
*   **Seamless Tiling**: Two methods—Image-based edge blending and Model-based circular padding for true seamless generation.
*   **Blender Bridge**: Live preview of your HDRI skies, Terrain heightmaps, and 3D Models directly in Blender.
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

**IMPORTANT: To enable the PBR Extractor (Ubisoft CHORD), you must run the installer manually:**

**Windows:**
Double-click `installer\install_pbr_extractor.bat`.

**Manual / Linux / Mac:**
```bash
python installer/install_pbr_extractor.py
```

> **License Note**: The Ubisoft CHORD model is gated. You must accept the license at [Hugging Face](https://huggingface.co/Ubisoft/ubisoft-laforge-chord).
> If the installer fails to download the model due to authentication, download `chord_v1.safetensors` manually and place it in `ComfyUI/models/gemini_pbr/`.

---

## 📚 Node Reference Guide

### 🧱 PBR & Texture Tools

#### **PBR Extractor (Ubisoft CHORD)**
*Category: `360_HDRI/Gemini`*
Extracts full PBR material maps from a single image using the state-of-the-art **Ubisoft LaForge CHORD** model.
- **Inputs**: `albedo_image`.
- **Outputs**: `Albedo`, `Normal`, `Roughness`, `Depth`, `Metallic`.
- **Fallback**: If the model is missing, it automatically switches to a lightweight algorithmic mode (Sobel/Luminance) so your workflow never breaks.

#### **Save Material (PBR)**
*Category: `360_HDRI/Gemini`*
Batch saver for PBR maps. Saves all connected maps (Albedo, Normal, Roughness, etc.) into a dedicated subfolder with standardized naming.
- **Inputs**: All map types + `folder_name` (e.g., "MyTexturePack").

#### **Channel Packer**
*Category: `360_HDRI/Gemini`*
Combines 3 or 4 grayscale images into a single RGB(A) image. Essential for Game Engine workflows (ORM textures).
- **Structure**: Red, Green, Blue, Alpha inputs.

#### **Image Comparator**
*Category: `360_HDRI/Gemini`*
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
*Category: `360_HDRI/Gemini`*
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

---

### 🌐 360° HDRI Tools

#### **Save Fake HDRI (EXR)**
*Category: `360_HDRI`*
Saves an LDR image as an `.exr` file (32-bit float fake), compatible with 3D software lighting.

#### **Image to 360 Latent**
*Category: `360_HDRI`*
Resizes and masks latents specifically for 2:1 aspect ratio generation.

#### **Rotate 360 Image**
*Category: `360_HDRI`*
shifts the pixels of a 360 image horizontally (Yaw), Pitch, or Roll.

#### **Generate Pole Mask**
*Category: `360_HDRI`*
Creates a mask covering the top and bottom "poles" of a 360 image, useful for inpainting distortions.

---

### 🐵 Blender Integration
*Requires installing the scripts in `blender_scripts/` to your Blender addons.*

#### **Preview in Blender (360 Sky)**
*Category: `360_HDRI/Blender`*
Sends the image to Blender and sets it as the World Background environment automatically.

#### **Preview Heightmap in Blender**
*Category: `360_HDRI/Blender`*
Sends an image to Blender and displaces a plane geometry to visualize 3D terrain.

#### **Preview Model in Blender (GLB)**
*Category: `360_HDRI/Blender`*
Sends a `.glb` or `.gltf` file path to Blender for immediate loading.

#### **Sync Lighting to Blender**
*Category: `360_HDRI/Blender`*
Updates Blender's lighting creation based on estimated parameters.

---

### 🦙 Ollama (Local AI) Integration
*Requires local [Ollama](https://ollama.com) installation.*

#### **Ollama Vision Analysis**
*Category: `360_HDRI/Ollama`*
Uses a vision model (e.g., LLaVA) to describe an image. Great for auto-captioning or interrogation.

#### **Ollama Lighting Estimator**
*Category: `360_HDRI/Ollama`*
Analyzes an image to guess the sun's position (elevation/azimuth) and color temperature.

---

### 🛠️ Prompt & Heightmap Utilities

#### **Terrain Prompt Maker (Ollama)**
*Category: `360_HDRI/Terrain`*
Helper to generate rich terrain descriptions.

#### **Terrain Texture Prompt Maker**
*Category: `360_HDRI/Terrain`*
Helper for satellite-style texture prompts.

#### **Terrain HeightField Prompt Maker**
*Category: `360_HDRI/Terrain`*
Generates prompts tuned for grayscale displacement maps (linear, non-optical).

#### **Color to Heightmap**
*Category: `360_HDRI/Terrain`*
Converts RGB images to high-quality Grayscale heightmaps with Gamma and Level controls.

#### **Simple Heightmap Normalizer**
*Category: `360_HDRI/Terrain`*
Ensures heightmap values span the full 0.0 - 1.0 range.

---

## 📄 License
(c) Geekatplay Studio.
Ubisoft CHORD model follows its own license (Research-Only Copyleft).
Other components MIT.
