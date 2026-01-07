# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import re

def tensor_to_base64(image_tensor):
    """
    Converts a ComfyUI image tensor (batch) to a base64 encoded PNG string.
    Takes the first image from the batch.
    """
    i = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class OllamaVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {"default": "llava"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "prefix": ("STRING", {"multiline": True, "default": "photorealistic, 8k, high quality, masterpiece, "}),
                "suffix": ("STRING", {"multiline": True, "default": ", highly detailed, hdr, 360 panorama"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_image"
    CATEGORY = "Ollama"

    def analyze_image(self, images, prompt, ollama_url, model, temperature, seed, prefix="", suffix=""):
        try:
            # Convert image to base64
            img_str = tensor_to_base64(images)
            
            # Prepare payload
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_str],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": seed
                }
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            description = result.get("response", "")
            
            # Combine with prefix/suffix
            final_prompt = f"{prefix}{description}{suffix}"
            return (final_prompt,)
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return (f"Error: {e}",)

class OllamaLightingEstimator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {"default": "llava"}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("azimuth", "elevation", "intensity", "color_hex")
    FUNCTION = "estimate_lighting"
    CATEGORY = "Ollama"

    def estimate_lighting(self, images, ollama_url, model, temperature, seed):
        # Prompt designed to get structured JSON output
        prompt = """Analyze the lighting in this image. 
        Estimate the sun position (azimuth and elevation), light intensity, and light color.
        Return ONLY a JSON object with the following keys:
        - "azimuth": float (0-360 degrees, where 0 is North/Front)
        - "elevation": float (0-90 degrees, where 0 is horizon, 90 is zenith)
        - "intensity": float (0.0-10.0, relative brightness)
        - "color": string (hex code like #FFFFFF)
        
        Example: {"azimuth": 45.0, "elevation": 30.0, "intensity": 1.5, "color": "#FFD700"}
        """
        
        try:
            img_str = tensor_to_base64(images)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_str],
                "stream": False,
                "format": "json", # Force JSON mode if supported by model/ollama version
                "options": {
                    "temperature": temperature,
                    "seed": seed
                }
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            text_response = result.get("response", "")
            
            # Parse JSON
            try:
                # Find the first '{' and last '}' to extract JSON if there's extra text
                start = text_response.find('{')
                end = text_response.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = text_response[start:end]
                    data = json.loads(json_str)
                    
                    azimuth = float(data.get("azimuth", 0.0))
                    elevation = float(data.get("elevation", 45.0))
                    intensity = float(data.get("intensity", 1.0))
                    color = data.get("color", "#FFFFFF")
                    
                    return (azimuth, elevation, intensity, color)
                else:
                    print("No JSON found in response, falling back to regex.")
                    raise json.JSONDecodeError("No parsed JSON", text_response, 0)
            except json.JSONDecodeError:
                print(f"OllamaLightingEstimator: JSON decode failed. Response was: {text_response}")
                # Try a fallback regex parsing if JSON fails
                azimuth = 0.0
                elevation = 45.0
                intensity = 1.0
                color = "#FFFFFF"
                
                # Regex patterns
                az_match = re.search(r'"?azimuth"?\s*[:=]\s*(\d+(?:\.\d+)?)', text_response, re.IGNORECASE)
                if az_match: azimuth = float(az_match.group(1))

                el_match = re.search(r'"?elevation"?\s*[:=]\s*(\d+(?:\.\d+)?)', text_response, re.IGNORECASE)
                if el_match: elevation = float(el_match.group(1))

                int_match = re.search(r'"?intensity"?\s*[:=]\s*(\d+(?:\.\d+)?)', text_response, re.IGNORECASE)
                if int_match: intensity = float(int_match.group(1))
                
                col_match = re.search(r'"?color"?\s*[:=]\s*"?([^"\s,]+)"?', text_response, re.IGNORECASE)
                if col_match: color = col_match.group(1)
                
                print(f"OllamaLightingEstimator: Regex Fallback -> Az:{azimuth}, El:{elevation}, In:{intensity}, Col:{color}")
                return (azimuth, elevation, intensity, color)
                
                # More robust regex to handle "Azimuth: 45" or "azimuth": 45
                az_match = re.search(r'(?:azimuth|Azimuth)["\s:]+([\d\.]+)', text_response)
                if az_match: azimuth = float(az_match.group(1))
                
                el_match = re.search(r'(?:elevation|Elevation)["\s:]+([\d\.]+)', text_response)
                if el_match: elevation = float(el_match.group(1))
                
                int_match = re.search(r'(?:intensity|Intensity)["\s:]+([\d\.]+)', text_response)
                if int_match: intensity = float(int_match.group(1))
                
                col_match = re.search(r'(?:color|Color)["\s:]+"?(#[A-Fa-f0-9]{6})"?', text_response)
                if col_match: color = col_match.group(1)
                
                return (azimuth, elevation, intensity, color)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return (0.0, 45.0, 1.0, "#FFFFFF")

