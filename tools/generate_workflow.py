# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import json
import os

def generate_workflow():
    workflow = {}

    # 1. Checkpoint Loader (Flux)
    workflow["1"] = {
        "inputs": {
            "ckpt_name": "flux1-dev.safetensors"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": { "title": "Load Checkpoint" }
    }

    # 2. LoRA Loader
    workflow["2"] = {
        "inputs": {
            "lora_name": "human_360_flux_dev_v1.safetensors",
            "strength_model": 1.0,
            "strength_clip": 1.0,
            "model": ["1", 0],
            "clip": ["1", 1]
        },
        "class_type": "LoraLoader",
        "_meta": { "title": "Load 360 LoRA" }
    }

    # 3. Seamless Tile
    workflow["3"] = {
        "inputs": {
            "axis": "both",
            "model": ["2", 0]
        },
        "class_type": "SimpleSeamlessTile",
        "_meta": { "title": "Make Seamless" }
    }

    # 4. Empty Latent Image
    workflow["4"] = {
        "inputs": {
            "width": 2048,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": { "title": "Empty Latent" }
    }

    # 5. CLIP Text Encode (Positive)
    workflow["5"] = {
        "inputs": {
            "text": "360 degree equirectangular panorama, hdri sky, sunny day, clouds, high resolution",
            "clip": ["2", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": { "title": "Positive Prompt" }
    }

    # 6. CLIP Text Encode (Negative)
    workflow["6"] = {
        "inputs": {
            "text": "seams, distortion, blur, low quality",
            "clip": ["2", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": { "title": "Negative Prompt" }
    }

    # 7. KSampler
    workflow["7"] = {
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["3", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["4", 0]
        },
        "class_type": "KSampler",
        "_meta": { "title": "KSampler" }
    }

    # 8. VAE Decode
    workflow["8"] = {
        "inputs": {
            "samples": ["7", 0],
            "vae": ["1", 2]
        },
        "class_type": "VAEDecode",
        "_meta": { "title": "VAE Decode" }
    }

    # 9. SaveFakeHDRI
    workflow["9"] = {
        "inputs": {
            "filename_prefix": "ComfyUI_HDRI",
            "exposure_boost": 2.0,
            "gamma": 2.2,
            "images": ["8", 0]
        },
        "class_type": "SaveFakeHDRI",
        "_meta": { "title": "Save HDRI (EXR)" }
    }

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "workflows", "flux_360_hdri.json")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Workflow saved to {output_path}")

if __name__ == "__main__":
    generate_workflow()
