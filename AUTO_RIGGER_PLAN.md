# Project Name: 360Pack Studio Auto-Rigger

**Target Platform:** Visual Studio Code (Extension/Script Runner)  
**Core Technologies:** ComfyUI (Generation), Blender (Geometry Processing/Rigging), Python.

## 1. Objective
Build a "One-Click" workflow inside Visual Studio that takes a static source image (landscape or character), generates a 3D mesh, cleans the topology, and applies a functional skeleton (rig). The process must be 100% reliable, minimizing user intervention.

## 2. Architecture Overview
The workflow functions as a pipeline orchestrated by Python scripts running from VS.

### Step 1: Generation (ComfyUI)
*   **Input:** 2D Image (Landscape or Humanoid).
*   **Engine:** Tencent Hunyuan3D (Best for clean geometry) or TripoSR (Fastest).
*   **Output:** Raw .obj or .glb mesh.

### Step 2: Cleaning (Blender Background)
*   **Action:** Automated headless Blender instance.
*   **Process:** Remesh (Voxel), Decimate (Poly-reduction), Merge Vertices.

### Step 3: Auto-Rigging (Blender or Comfy)
*   **Action:** Skeleton extraction and weight binding.
*   **Engine:** ComfyUI-UniRig (Native ML rigging) OR Blender Rigify (Scripted).

## 3. Functional Requirements
### A. The Visual Studio Interface (Frontend)
A simple VS Code Extension or Task Runner panel containing:
*   **Input Slot:** Drag-and-drop image path.
*   **Settings:**
    *   Mesh Quality: (Low/Mid/High) - controls Voxel Remesh size.
    *   Rig Type: (Humanoid/Basic).
*   **Action Button:** "Generate & Rig".
*   **Viewer:** Embedded 3D viewer (using a VS Code extension like glTF Viewer) to preview the result immediately.

### B. The ComfyUI Workflow (Backend 1)
We will create a specific API-enabled workflow (`autorig_api.json`) that VS Code triggers via Websocket.

**Nodes Required:**
*   Hunyuan3D_V1 (Generates the mesh).
*   Save 3D Mesh (Exports to a temp folder VS watches).
*   (Optional) UniRig node if we want rigging inside Comfy.

### C. The Blender Script (Backend 2 - The "Cleaning" Phase)
Since AI meshes often have holes or non-manifold geometry, we use Blender for the "100% work" guarantee on cleaning.

**Python Script Logic (`clean_and_rig.py`):**
```python
import bpy
import sys

# 1. Clear Scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# 2. Import AI Mesh
input_path = sys.argv[-2] # Passed from VS
bpy.ops.import_scene.gltf(filepath=input_path)
obj = bpy.context.selected_objects[0]

# 3. Clean Geometry (The "100%" Fix)
# Voxel Remesh creates a watertight skin, removing internal geometry issues
bpy.ops.object.modifier_add(type='REMESH')
obj.modifiers["Remesh"].mode = 'VOXEL'
obj.modifiers["Remesh"].voxel_size = 0.05  # Adjustable via args
bpy.ops.object.modifier_apply(modifier="Remesh")

# Decimate to keep polycount reasonable for games/web
bpy.ops.object.modifier_add(type='DECIMATE')
obj.modifiers["Decimate"].ratio = 0.1
bpy.ops.object.modifier_apply(modifier="Decimate")

# 4. Rigging (Scripted Auto-Rig)
# If using Rigify, we align a metarig to bounding box (Complex)
# EASIER PATH: Import the skeleton from UniRig if generated in Comfy,
# OR use a basic "Voxel Heat Diffuse" binding if we have a template skeleton.

# 5. Export
output_path = sys.argv[-1]
bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
```

## 4. Implementation Steps (The "Easy" Plan)
### Phase 1: The "UniRig" Pipeline (Easiest)
This keeps the complexity inside ComfyUI, using VS only as a trigger.
1.  Install ComfyUI-UniRig (New custom node based on VAST-AI).
2.  Setup Comfy workflow: Image -> Hunyuan3D -> UniRig -> Save GLB.
3.  VS Code Script: Just sends the image to Comfy API and waits for the file.

### Phase 2: The "Hybrid" Pipeline (Quality Focus)
If UniRig fails on complex shapes, we use the Blender bridge.
1.  VS Code triggers Comfy to get the Mesh.
2.  VS Code runs: `blender.exe --background --python clean_and_rig.py -- "input.glb" "output_rigged.glb"`
3.  Blender handles the cleanup and binds it to a standard mixamo-compatible skeleton.

## 5. Recommended Tools to Install
*   **ComfyUI Node:** ComfyUI-Tencent-Hunyuan3D (Best generation).
*   **ComfyUI Node:** ComfyUI-UniRig (Best native rigging).
*   **VS Code Extension:** glTF Viewer (For previewing inside the IDE).

This workflow keeps the "Heavy Lifting" in the background. The user just sees: Image In -> Progress Bar -> Rigged Character Out.
