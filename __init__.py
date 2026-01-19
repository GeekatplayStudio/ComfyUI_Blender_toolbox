# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox
"""
ComfyUI-Blender-Toolbox
======================
A complete toolbox for 3D Gen, HDRI, PBR, and Blender synchronization in ComfyUI.
"""

from .nodes.hdri_nodes import SaveFakeHDRI, ImageTo360Latent, Rotate360Image, GeneratePoleMask
from .nodes.seamless_nodes import SimpleSeamlessTile, Heal360Seam, SeamlessTileVAE, PreviewSeamlessTile
from .nodes.blender_nodes import PreviewInBlender, PreviewHeightmapInBlender, SyncLightingToBlender, PreviewModelInBlender, PreviewMeshInBlender, LoadBlenderPBR, PreviewTextureOnMesh, SaveAndSendPBRToBlender
from .nodes.pbr_nodes import SimplePBRGenerator, TerrainPromptMaker, SimpleHeightmapNormalizer, TerrainTexturePromptMaker, TerrainHeightFieldPromptMaker, ColorToHeightmap, TerrainErosionPromptMaker, MaterialTexturePromptMaker
from .nodes.texture_nodes import TextureScrambler
from .nodes.ollama_nodes import OllamaVision, OllamaLightingEstimator
from .nodes.geekatplay_nodes import GapSeamlessTiler, GapPBRExtractor, GapChannelPacker, GapImageMerger, GapMaterialSaver
from .nodes.geekatplay_toolbox import GapSmartResizer, GapStringViewer, GapPauser, GapLogicSwitch, GapGroupManager, GapVRAMPurge, GapVisualComparator
from .nodes.tripo_nodes import Geekatplay_Tripo_ModelGen, Geekatplay_Tripo_AnimateRig
from .nodes.meshy_nodes import Geekatplay_Meshy_TextTo3D, Geekatplay_Meshy_ImageTo3D
from .nodes.hitem3d_nodes import Geekatplay_HiTem3D_Gen
from .nodes.geekatplay_key_manager import Geekatplay_ApiKey_Manager
from .nodes.flux_terrain_nodes import FluxTerrainPromptGenerator, FluxOptionalImageRef
from .nodes.geometry_ops import GapFillHoles, GapBlenderVoxelRemesh, GapFaceNormalsFix, GapBlenderDecimate, GapBlenderSubdivide, GapBlenderBoolean, GapBlenderSmartUV

NODE_CLASS_MAPPINGS = {
    "FluxTerrainPromptGenerator": FluxTerrainPromptGenerator,
    "FluxOptionalImageRef": FluxOptionalImageRef,
    "SaveFakeHDRI": SaveFakeHDRI,
    "ImageTo360Latent": ImageTo360Latent,
    "Rotate360Image": Rotate360Image,
    "GeneratePoleMask": GeneratePoleMask,
    "SimpleSeamlessTile": SimpleSeamlessTile,
    "SeamlessTileVAE": SeamlessTileVAE,
    "Heal360Seam": Heal360Seam,
    "PreviewSeamlessTile": PreviewSeamlessTile,
    "PreviewInBlender": PreviewInBlender,
    "PreviewHeightmapInBlender": PreviewHeightmapInBlender,
    "PreviewModelInBlender": PreviewModelInBlender,
    "PreviewMeshInBlender": PreviewMeshInBlender,
    "PreviewTextureOnMesh": PreviewTextureOnMesh,
    "SaveAndSendPBRToBlender": SaveAndSendPBRToBlender,
    "SyncLightingToBlender": SyncLightingToBlender,
    "LoadBlenderPBR": LoadBlenderPBR,
    "SimplePBRGenerator": SimplePBRGenerator,
    "TerrainPromptMaker": TerrainPromptMaker,
    "TerrainTexturePromptMaker": TerrainTexturePromptMaker,
    "TerrainHeightFieldPromptMaker": TerrainHeightFieldPromptMaker,
    "TerrainErosionPromptMaker": TerrainErosionPromptMaker,
    "MaterialTexturePromptMaker": MaterialTexturePromptMaker,
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
    "Geekatplay_Tripo_ModelGen": Geekatplay_Tripo_ModelGen,
    "Geekatplay_Tripo_AnimateRig": Geekatplay_Tripo_AnimateRig,
    "Geekatplay_Meshy_TextTo3D": Geekatplay_Meshy_TextTo3D,
    "Geekatplay_Meshy_ImageTo3D": Geekatplay_Meshy_ImageTo3D,
    "Geekatplay_HiTem3D_Gen": Geekatplay_HiTem3D_Gen,
    "Geekatplay_ApiKey_Manager": Geekatplay_ApiKey_Manager,
    "GapFillHoles": GapFillHoles,
    "GapBlenderVoxelRemesh": GapBlenderVoxelRemesh,
    "GapFaceNormalsFix": GapFaceNormalsFix,
    "GapBlenderDecimate": GapBlenderDecimate,
    "GapBlenderSubdivide": GapBlenderSubdivide,
    "GapBlenderBoolean": GapBlenderBoolean,
    "GapBlenderSmartUV": GapBlenderSmartUV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFakeHDRI": "Save Fake HDRI (EXR)",
    "ImageTo360Latent": "Image to 360 Latent (Resize & Stretch)",
    "Rotate360Image": "Rotate 360 Image (Pitch/Yaw/Roll)",
    "GeneratePoleMask": "Generate Pole Mask",
    "SimpleSeamlessTile": "Simple Seamless Tile (Model)",
    "SeamlessTileVAE": "Seamless Tile (VAE)",
    "Heal360Seam": "Heal 360 Seam (Post-Process)",
    "PreviewSeamlessTile": "Preview Seamless Tile",
    "PreviewInBlender": "Preview in Blender (360 Sky)",
    "PreviewHeightmapInBlender": "Preview Heightmap in Blender",
    "PreviewModelInBlender": "Preview Model in Blender (GLB)",
    "PreviewMeshInBlender": "Preview Mesh in Blender (Send)",
    "PreviewTextureOnMesh": "Preview Texture on Selected Object",
    "SyncLightingToBlender": "Sync Lighting to Blender",
    "LoadBlenderPBR": "Receive from Blender (PBR)",
    "SimplePBRGenerator": "Simple PBR Generator",
    "TerrainPromptMaker": "Terrain Prompt Maker (Ollama)",
    "TerrainTexturePromptMaker": "Terrain Texture Prompt Maker (Ollama)",
    "TerrainHeightFieldPromptMaker": "Terrain HeightField Prompt Maker (Linear)",
    "TerrainErosionPromptMaker": "Terrain Erosion Prompt Maker (Detailer)",
    "MaterialTexturePromptMaker": "Material Texture Prompt Maker (Preset)",
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
    "Geekatplay_Tripo_ModelGen": "Tripo3D Model Generator (Geekatplay)",
    "Geekatplay_Tripo_AnimateRig": "Tripo3D Animator (Geekatplay)",
    "Geekatplay_Meshy_TextTo3D": "Meshy Text to 3D (Geekatplay)",
    "Geekatplay_Meshy_ImageTo3D": "Meshy Image to 3D (Geekatplay)",
    "Geekatplay_HiTem3D_Gen": "HiTem3D Generator (Geekatplay)",
    "Geekatplay_ApiKey_Manager": "API Key Manager (Geekatplay)",
    "FluxTerrainPromptGenerator": "Flux Terrain Prompt Generator",
    "FluxOptionalImageRef": "Flux Optional Image Ref",
    "GapFillHoles": "Fill Holes (Topology)",
    "GapBlenderVoxelRemesh": "Remesh (Blender Voxel)",
    "GapFaceNormalsFix": "Fix Normals (Topology)",
    "GapBlenderDecimate": "Decimate Mesh (Blender)",
    "GapBlenderSubdivide": "Subdivide Mesh (Blender)",
    "GapBlenderBoolean": "Boolean Operation (Blender)",
    "GapBlenderSmartUV": "Smart UV Unwrap (Blender)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"
