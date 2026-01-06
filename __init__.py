# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

from .nodes.hdri_nodes import SaveFakeHDRI, ImageTo360Latent
from .nodes.seamless_nodes import SimpleSeamlessTile, Heal360Seam, SeamlessTileVAE
from .nodes.blender_nodes import PreviewInBlender, PreviewHeightmapInBlender, SyncLightingToBlender
from .nodes.pbr_nodes import SimplePBRGenerator, TerrainPromptMaker, SimpleHeightmapNormalizer, TerrainTexturePromptMaker
from .nodes.texture_nodes import TextureScrambler
from .nodes.ollama_nodes import OllamaVision, OllamaLightingEstimator

NODE_CLASS_MAPPINGS = {
    "SaveFakeHDRI": SaveFakeHDRI,
    "ImageTo360Latent": ImageTo360Latent,
    "SimpleSeamlessTile": SimpleSeamlessTile,
    "SeamlessTileVAE": SeamlessTileVAE,
    "Heal360Seam": Heal360Seam,
    "PreviewInBlender": PreviewInBlender,
    "PreviewHeightmapInBlender": PreviewHeightmapInBlender,
    "SyncLightingToBlender": SyncLightingToBlender,
    "SimplePBRGenerator": SimplePBRGenerator,
    "TerrainPromptMaker": TerrainPromptMaker,
    "TerrainTexturePromptMaker": TerrainTexturePromptMaker,
    "SimpleHeightmapNormalizer": SimpleHeightmapNormalizer,
    "TextureScrambler": TextureScrambler,
    "OllamaVision": OllamaVision,
    "OllamaLightingEstimator": OllamaLightingEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFakeHDRI": "Save Fake HDRI (EXR)",
    "ImageTo360Latent": "Image to 360 Latent (Resize & Encode)",
    "SimpleSeamlessTile": "Simple Seamless Tile (Model)",
    "SeamlessTileVAE": "Seamless Tile (VAE)",
    "Heal360Seam": "Heal 360 Seam (Post-Process)",
    "PreviewInBlender": "Preview in Blender (360 Sky)",
    "PreviewHeightmapInBlender": "Preview Heightmap in Blender",
    "SyncLightingToBlender": "Sync Lighting to Blender",
    "SimplePBRGenerator": "Simple PBR Generator",
    "TerrainPromptMaker": "Terrain Prompt Maker (Ollama)",
    "TerrainTexturePromptMaker": "Terrain Texture Prompt Maker (Ollama)",
    "SimpleHeightmapNormalizer": "Simple Heightmap Normalizer",
    "TextureScrambler": "Texture Scrambler (Style Transfer)",
    "OllamaVision": "Ollama Vision Analysis",
    "OllamaLightingEstimator": "Ollama Lighting Estimator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
