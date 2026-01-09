# ComfyUI-360-HDRI-Suite

A suite of ComfyUI nodes designed for working with 360° HDRI images, seamless textures, and Ollama-based vision analysis.

## Features

- **360 HDRI Tools**:
  - `Save Fake HDRI (EXR)`: Save images as EXR format.
  - `Image to 360 Latent`: Resize and encode images specifically for 360 outputs.
  - `Heal 360 Seam`: Post-processing to fix seams in 360 images.
  - `Preview in Blender`: Preview generated 360 skies directly in Blender (requires addon).
  - `Sync Lighting to Blender`: Synchronize estimated lighting to Blender.

- **Seamless Textures**:
  - `Simple Seamless Tile`: Make images seamlessly tileable.
  - `Seamless Tile (VAE)`: VAE-based tiling.

- **PBR & Textures**:
  - `Simple PBR Generator`: Generate PBR maps from images.
  - `Texture Scrambler`: Style transfer utility.
  - `Terrain tools`: Helpers for generating terrain prompts and heightmaps.

- **Ollama Integration**:
  - `Ollama Vision Analysis`: Use local LLMs to describe images or generate prompts from images.
  - `Ollama Lighting Estimator`: Estimate sun position and lighting from an image using LLMs.

## Installation

1.  **Install ComfyUI**: Ensure you have ComfyUI installed and working.
2.  **Clone the Repository**:
    Navigate to your `ComfyUI/custom_nodes/` folder and clone this repository:
    ```bash
    git clone https://github.com/GeekAtPlay/ComfyUI-360-HDRI-Suite
    ```
3.  **Install Dependencies**:
    The dependencies should install automatically upon restart if you are using the Manager, or you can run:
    ```bash
    install.bat
    ```
    (Windows)

## Ollama Setup (Required for Ollama Nodes)

To use the Ollama nodes (`Ollama Vision Analysis` and `Ollama Lighting Estimator`), you must have **Ollama** installed and running.

1.  Download and install Ollama from [ollama.com](https://ollama.com).
2.  Pull a vision-capable model, for example **LLaVA**:
    ```bash
    ollama run llava
    ```
3.  Ensure the Ollama server is running (default: `http://127.0.0.1:11434`).

## Blender Integration (Optional)

To use the Blender preview nodes:
1.  Install the Blender addon located in `blender_scripts/`.
2.  Ensure Blender is running and the addon is enabled/listening when using the Preview nodes.

### Preview Heightmap in Blender
Sends a heightmap (and optional texture/PBR maps) to Blender to generate a real 3D terrain mesh.

**Parameters:**
- **`use_texture_as_heightmap`**: (Boolean) If enabled, uses the connected `texture` input as the source for the heightmap (converting it to grayscale) instead of the `images` input. useful if you only have a texture.
- **`auto_level_height`**: (Boolean) Automatically stretches the heightmap data to the full 0-255 range. Prevents flat terrains from low-contrast images.
- **`height_gamma`**: (Float) Adjusts the curve of the heightmap.
    - Values < 1.0 (e.g. 0.6) lift mid-tones (fix deep holes).
    - Values > 1.0 (e.g. 1.4) darken mid-tones (accentuate peaks).
- **`smoothing_amount`**: (Float) Applies Gaussian Blur to the heightmap. Value of `1.0` - `3.0` removes sharp spikes and high-frequency noise from textures.
- **`edge_falloff`**: (Float) Fades the terrain height to zero at the edges. Value `0.1` - `0.2` creates a nice "Mountain Island" effect.
- **`rotation`**: (Enum) Rotates the heightmap (0, 90, 180, 270) to fix orientation issues (e.g. if terrain looks "sideways").

## License

(c) Geekatplay Studio
