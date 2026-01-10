# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite
"""
ComfyUI-360-HDRI-Suite
======================
A suite of nodes for creating 360-degree HDRI panoramas and 3D terrain heightmaps in ComfyUI.
Includes Blender integration for live preview, PBR maps generation, and scene synchronization.
"""

from .nodes.hdri_nodes import SaveFakeHDRI, ImageTo360Latent, Rotate360Image, GeneratePoleMask
from .nodes.seamless_nodes import SimpleSeamlessTile, Heal360Seam, SeamlessTileVAE
from .nodes.blender_nodes import PreviewInBlender, PreviewHeightmapInBlender, SyncLightingToBlender, PreviewModelInBlender, PreviewMeshInBlender
from .nodes.pbr_nodes import SimplePBRGenerator, TerrainPromptMaker, SimpleHeightmapNormalizer, TerrainTexturePromptMaker, TerrainHeightFieldPromptMaker, ColorToHeightmap
from .nodes.texture_nodes import TextureScrambler
from .nodes.ollama_nodes import OllamaVision, OllamaLightingEstimator
from .nodes.geekatplay_nodes import GapSeamlessTiler, GapPBRExtractor, GapChannelPacker, GapImageMerger, GapMaterialSaver
from .nodes.geekatplay_toolbox import GapSmartResizer, GapStringViewer, GapPauser, GapLogicSwitch, GapGroupManager, GapVRAMPurge, GapVisualComparator

NODE_CLASS_MAPPINGS = {
    "SaveFakeHDRI": SaveFakeHDRI,
    "ImageTo360Latent": ImageTo360Latent,
    "Rotate360Image": Rotate360Image,
    "GeneratePoleMask": GeneratePoleMask,
    "SimpleSeamlessTile": SimpleSeamlessTile,
    "SeamlessTileVAE": SeamlessTileVAE,
    "Heal360Seam": Heal360Seam,
    "PreviewInBlender": PreviewInBlender,
    "PreviewHeightmapInBlender": PreviewHeightmapInBlender,
    "PreviewModelInBlender": PreviewModelInBlender,
    "PreviewMeshInBlender": PreviewMeshInBlender,
    "SyncLightingToBlender": SyncLightingToBlender,
    "SimplePBRGenerator": SimplePBRGenerator,
    "TerrainPromptMaker": TerrainPromptMaker,
    "TerrainTexturePromptMaker": TerrainTexturePromptMaker,
    "TerrainHeightFieldPromptMaker": TerrainHeightFieldPromptMaker,
    "ColorToHeightmap": ColorToHeightmap,
    "SimpleHeightmapNormalizer": SimpleHeightmapNormalizer,
    "TextureScrambler": TextureScrambler,
    "OllamaVision": OllamaVision,
    "OllamaLightingEstimator": OllamaLightingEstimator,
    "GapSeamlessTiler": GapSeamlessTiler,
    "GapPBRExtractor": GapPBRExtractor,
    "GapChannelPacker": GapChannelPacker,
    "GapImageMerger": GapImageMerger,
    "GapMaterialSaver": GapMaterialSaver,
    "GapSmartResizer": GapSmartResizer,
    "GapStringViewer": GapStringViewer,
    "GapPauser": GapPauser,
    "GapLogicSwitch": GapLogicSwitch,
    "GapGroupManager": GapGroupManager,
    "GapVRAMPurge": GapVRAMPurge,
    "GapVisualComparator": GapVisualComparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFakeHDRI": "Save Fake HDRI (EXR)",
    "ImageTo360Latent": "Image to 360 Latent (Resize & Stretch)",
    "Rotate360Image": "Rotate 360 Image (Pitch/Yaw/Roll)",
    "GeneratePoleMask": "Generate Pole Mask",
    "SimpleSeamlessTile": "Simple Seamless Tile (Model)",
    "SeamlessTileVAE": "Seamless Tile (VAE)",
    "Heal360Seam": "Heal 360 Seam (Post-Process)",
    "PreviewInBlender": "Preview in Blender (360 Sky)",
    "PreviewHeightmapInBlender": "Preview Heightmap in Blender",
    "PreviewModelInBlender": "Preview Model in Blender (GLB)",
    "PreviewMeshInBlender": "Preview Mesh in Blender (Send)",
    "SyncLightingToBlender": "Sync Lighting to Blender",
    "SimplePBRGenerator": "Simple PBR Generator",
    "TerrainPromptMaker": "Terrain Prompt Maker (Ollama)",
    "TerrainTexturePromptMaker": "Terrain Texture Prompt Maker (Ollama)",
    "TerrainHeightFieldPromptMaker": "Terrain HeightField Prompt Maker (Linear)",
    "ColorToHeightmap": "Color to Heightmap (Grayscale)",
    "SimpleHeightmapNormalizer": "Simple Heightmap Normalizer",
    "TextureScrambler": "Texture Scrambler (Style Transfer)",
    "OllamaVision": "Ollama Vision Analysis",
    "OllamaLightingEstimator": "Ollama Lighting Estimator",
    "GapSeamlessTiler": "Seamless Tile (Geekatplay)",
    "GapPBRExtractor": "PBR Extractor (Ubisoft CHORD)",
    "GapChannelPacker": "Channel Packer (Geekatplay)",
    "GapImageMerger": "Image Merger (Geekatplay)",
    "GapMaterialSaver": "Geekatplay Material Saver (PBR)",
    "GapSmartResizer": "3D Toolbox Smart Resizer (Geekatplay)",
    "GapStringViewer": "3D Toolbox String Viewer",
    "GapPauser": "3D Toolbox Workflow Pauser",
    "GapLogicSwitch": "3D Toolbox Logic Switch",
    "GapGroupManager": "3D Toolbox Dynamic Group Manager",
    "GapVRAMPurge": "3D Toolbox VRAM Purge",
    "GapVisualComparator": "3D Toolbox Visual Comparator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"
