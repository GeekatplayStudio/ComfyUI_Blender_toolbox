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

## License

(c) Geekatplay Studio
