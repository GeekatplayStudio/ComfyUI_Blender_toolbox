from .nodes.hdri_nodes import SaveFakeHDRI
from .nodes.seamless_nodes import SimpleSeamlessTile, Heal360Seam, SeamlessTileVAE
from .nodes.blender_nodes import PreviewInBlender, PreviewHeightmapInBlender
from .nodes.pbr_nodes import SimplePBRGenerator

NODE_CLASS_MAPPINGS = {
    "SaveFakeHDRI": SaveFakeHDRI,
    "SimpleSeamlessTile": SimpleSeamlessTile,
    "SeamlessTileVAE": SeamlessTileVAE,
    "Heal360Seam": Heal360Seam,
    "PreviewInBlender": PreviewInBlender,
    "PreviewHeightmapInBlender": PreviewHeightmapInBlender,
    "SimplePBRGenerator": SimplePBRGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFakeHDRI": "Save Fake HDRI (EXR)",
    "SimpleSeamlessTile": "Simple Seamless Tile (Model)",
    "SeamlessTileVAE": "Seamless Tile (VAE)",
    "Heal360Seam": "Heal 360 Seam (Post-Process)",
    "PreviewInBlender": "Preview in Blender (360 Sky)",
    "PreviewHeightmapInBlender": "Preview Heightmap in Blender",
    "SimplePBRGenerator": "Simple PBR Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
