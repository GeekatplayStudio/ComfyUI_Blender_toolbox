# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox Test Suite

import unittest
import sys
import os
import shutil
import numpy as np

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import trimesh
except ImportError:
    trimesh = None

from nodes.utils_blender import get_blender_path, run_blender_script
# Import the node classes directly
from nodes.geometry_ops import GapBlenderVoxelRemesh, GapFillHoles, GapBlenderDecimate, GapBlenderSubdivide, GapBlenderBoolean, GapBlenderSmartUV

class TestBlenderGeometry(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- Starting Geometry Tests ---")
        if trimesh is None:
            raise unittest.SkipTest("Trimesh not installed")
            
        cls.blender_path = get_blender_path()
        if not cls.blender_path:
            logo = """
            WARNING: Blender not found! 
            Many tests will be skipped. 
            Please ensure Blender is installed or BLENDER_PATH is set.
            """
            print(logo)
        else:
            print(f"Targeting Blender: {cls.blender_path}")

    def test_01_blender_connection(self):
        """Test if we can run a basic python script in Blender"""
        if not self.blender_path: self.skipTest("No Blender")
        
        script = "import bpy; print('Alive')"
        result = run_blender_script(script)
        self.assertEqual(result.returncode, 0)
        
    def test_02_fill_holes_trimesh(self):
        """Test GapFillHoles (Pure Trimesh)"""
        print("\nTesting GapFillHoles...")
        # Create cube
        mesh = trimesh.creation.box(extents=(1,1,1))
        # Remove a face (faces are N,3 indices)
        # We delete the last face
        mesh.faces = mesh.faces[:-1]
        self.assertFalse(mesh.is_watertight, "Mesh should have a hole")
        
        node = GapFillHoles()
        # Node returns (mesh, string)
        result_mesh, info = node.fill_holes(mesh)
        
        print(f"Info: {info.strip()}")
        self.assertTrue(result_mesh.is_watertight, "Mesh should be watertight after fill")

    def test_03_voxel_remesh(self):
        """Test GapBlenderVoxelRemesh (Blender Bridge)"""
        if not self.blender_path: self.skipTest("No Blender")
        print("\nTesting GapBlenderVoxelRemesh...")
        
        mesh = trimesh.creation.icosphere(radius=1.0)
        node = GapBlenderVoxelRemesh()
        
        # Remesh with large voxel size for speed
        result_mesh, info = node.remesh(mesh, voxel_size=0.2, adaptivity=0.0)
        
        print(f"Info: {info}")
        self.assertIsInstance(result_mesh, trimesh.Trimesh)
        self.assertNotEqual(len(result_mesh.vertices), len(mesh.vertices))

    def test_04_blender_decimate(self):
        """Test GapBlenderDecimate"""
        if not self.blender_path: self.skipTest("No Blender")
        print("\nTesting GapBlenderDecimate...")
        
        # High res sphere
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        orig_faces = len(mesh.faces)
        
        node = GapBlenderDecimate()
        result_tuple = node.decimate(mesh, ratio=0.5)
        result_mesh = result_tuple[0]
        
        new_faces = len(result_mesh.faces)
        print(f"Decimate: {orig_faces} -> {new_faces} faces")
        
        self.assertLess(new_faces, orig_faces)

    def test_05_blender_subdivide(self):
        """Test GapBlenderSubdivide"""
        if not self.blender_path: self.skipTest("No Blender")
        print("\nTesting GapBlenderSubdivide...")
        
        mesh = trimesh.creation.box()
        orig_faces = len(mesh.faces)
        
        node = GapBlenderSubdivide()
        result_tuple = node.subdivide(mesh, levels=1, type="SIMPLE")
        result_mesh = result_tuple[0]
        
        new_faces = len(result_mesh.faces)
        print(f"Subdivide: {orig_faces} -> {new_faces} faces")
        self.assertGreater(new_faces, orig_faces)
        
    def test_06_blender_boolean(self):
        """Test GapBlenderBoolean"""
        if not self.blender_path: self.skipTest("No Blender")
        print("\nTesting GapBlenderBoolean...")
        
        # Cube at origin
        mesh_a = trimesh.creation.box(extents=(2,2,2))
        # Sphere at corner
        mesh_b = trimesh.creation.icosphere(radius=1.5)
        mesh_b.apply_translation([1,1,1])
        
        node = GapBlenderBoolean()
        # Difference
        result_tuple = node.boolean(mesh_a, mesh_b, operation="DIFFERENCE", solver="FAST")
        result_mesh = result_tuple[0]
        
        self.assertIsInstance(result_mesh, trimesh.Trimesh)
        # Should be smaller volume than original box (8.0)
        print(f"Boolean Result Volume: {result_mesh.volume:.2f} (Orig: 8.0)")
        self.assertLess(result_mesh.volume, 8.0)

    def test_07_blender_uv(self):
        """Test GapBlenderSmartUV"""
        if not self.blender_path: self.skipTest("No Blender")
        print("\nTesting GapBlenderSmartUV...")

        # Create a box which needs unwrapping
        mesh = trimesh.creation.box()
        # Reset visual metadata
        mesh.visual = trimesh.visual.ColorVisuals()

        node = GapBlenderSmartUV()
        result_tuple = node.unwrap(mesh, angle_limit=66.0, island_margin=0.02)
        result_mesh = result_tuple[0]

        self.assertIsInstance(result_mesh, trimesh.Trimesh)
        print(f"UV Unwrap: {len(mesh.vertices)} -> {len(result_mesh.vertices)} vertices")
        # Smart UV projects usually split UV islands, resulting in duplicated vertices
        self.assertGreaterEqual(len(result_mesh.vertices), len(mesh.vertices))

if __name__ == '__main__':
    unittest.main()
