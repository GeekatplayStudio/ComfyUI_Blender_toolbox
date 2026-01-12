# Third-Party Auto-Rigging API Integration

**Objective:** Add options to send the cleaned mesh to third-party services (AccuRig / Mixamo) for high-quality auto-rigging instead of using the local Blender method.

## 1. AccuRig (ActorCore)
*   **Status:** AccuRig is primarily a desktop application. It does not have a public web API for direct file uploads/downloads.
*   **Integration Method:** 
    *   Command Line (CLI): AccuRig does not officially support CLI automation.
    *   **Workaround:** We can automate the launch of the AccuRig app with the file path if installed locally, but full automation (save & exit) is hard without GUI scripting (PyAutoGUI).
    *   *Conclusion:* Direct API integration is not feasible. Best approach is an "Open in AccuRig" button that exports the clean FBX and launches the app.

## 2. Mixamo (Adobe)
*   **Status:** Mixamo is a web-based service. Adobe **deprecated** the public Fuse/Mixamo API years ago. There is no official REST API for uploading an OBJ and getting an FBX back.
*   **Integration Method:**
    *   **Puppeteer/Selenium Automation:** It is technically possible to script a headless browser to log in to Adobe, upload the mesh, place markers (or rely on auto), and download.
    *   **Challenges:** 
        *   Requires user credentials (Adobe ID).
        *   Captcha / 2FA issues.
        *   Fragile (breaks if Adobe changes UI).
    *   *Conclusion:* Not recommended for a stable plugin.

## 3. Alternative: ComfyUI-UniRig or 3D-Native ML
Instead of unreliable external APIs, we should focus on **Local ML Rigging** which fits the ComfyUI ethos.
*   **UniRig:** A ComfyUI node wrapper for specific ML rigging papers.
*   **Voxel Heat Diffuse Skinning:** A Blender addon (paid) that is the industry standard for "perfect" auto-weights. We can support it if installed.

## Proposed "Upload to Service" Feature (Manual Handoff)
Since fully automated APIs don't simplify existence for these specific tools, we can add a **"Prepare for AccuRig/Mixamo"** button.

### Feature: "Export for Auto-Rigging"
1.  **Format:** Convert standard `.glb` to `.obj` or `.fbx` (preferred by AccuRig/Mixamo).
2.  **T-Pose Orientation:** Ensure the model is facing +Z or -Y as expected by these tools.
3.  **Action:**
    *   Cleans topology (Voxel Remesh).
    *   Exports `_clean.fbx`.
    *   Opens the folder so user can drag-drop to Mixamo.com or AccuRig.

### Implementation Plan (Update Blender Addon)
Add a new operator `comfy360.export_for_accu_mixamo`:
1.  Apply standard cleanup (Remesh/Decimate).
2.  Force 'A-Pose' friendly rotation if detected (optional).
3.  Export FBX with settings optimized for these tools (Y-Forward, Z-Up).
4.  Show popup: "Mesh ready in [Path]. Drag into Mixamo/AccuRig."
