ComfyUI 360 HDRI Sync - Blender Addon
=======================================

This folder contains the Blender Addon to sync generated skies from ComfyUI.

Installation:
1. Open Blender.
2. Go to Edit > Preferences > Add-ons.
3. Click "Install..." at the top right.
4. Navigate to this folder:
   .../ComfyUI/custom_nodes/ComfyUI-360-HDRI-Suite/blender_scripts/
5. Select "comfyui_360_hdri_addon.py" and click "Install Add-on".
6. Enable the checkbox for "Import-Export: ComfyUI 360 HDRI Sync".

Usage:
1. In the 3D Viewport, press 'N' to open the sidebar.
2. Click the "ComfyUI 360" tab.
3. Click "Start Listener" to enable automatic syncing from ComfyUI.
   - When you run the workflow in ComfyUI, the sky will automatically update in Blender.
4. Or click "Load Sky File..." to manually pick an EXR file.
