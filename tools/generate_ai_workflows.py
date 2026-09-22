# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - generates the AI Scene Builder workflows
"""
Builds the AI Scene Builder workflow JSON files from the real node definitions, so every
widgets_values list matches the node's INPUT_TYPES order and every link is typed correctly.

    python tools/generate_ai_workflows.py        (run from the toolbox folder)

Regenerate after changing any AI node's inputs.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import conftest  # noqa: F401,E402  - ComfyUI module mocks so nodes import outside ComfyUI

from nodes import ai_builder_nodes  # noqa: E402
from nodes import geekatplay_toolbox  # noqa: E402

NODE_CLASSES = dict(ai_builder_nodes.NODE_CLASS_MAPPINGS)
NODE_CLASSES["GapStringViewer"] = geekatplay_toolbox.GapStringViewer
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

BUILTIN = {
    "LoadImage": {"widgets": ["example.png", "image"], "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")], "inputs": []},
    "PreviewImage": {"widgets": [], "outputs": [], "inputs": [("images", "IMAGE")]},
    "Note": {"widgets": None, "outputs": [], "inputs": []},
}


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self.next_id, self.next_link = 1, 1

    def _spec(self, cls):
        it = cls.INPUT_TYPES()
        spec = []
        for section in ("required", "optional"):
            for name, definition in it.get(section, {}).items():
                typ = definition[0]
                opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
                is_widget = (isinstance(typ, list) or typ in WIDGET_TYPES) and not opts.get("forceInput")
                spec.append((name, typ, opts, is_widget))
        return spec

    def add(self, type_name, pos, size, values=None, title=None, connect=None, text=None):
        """values: {widget_name: value} overrides; connect: {input_name: (node_id, output_index)}"""
        node_id = self.next_id
        self.next_id += 1
        values, connect = values or {}, connect or {}
        inputs, outputs, widgets = [], [], []
        if type_name in BUILTIN:
            b = BUILTIN[type_name]
            widgets = [text] if type_name == "Note" else list(b["widgets"])
            if type_name == "LoadImage" and "image" in values:
                widgets[0] = values["image"]
            for name, typ in b["inputs"]:
                inputs.append({"name": name, "type": typ, "link": None})
            for name, typ in b["outputs"]:
                outputs.append({"name": name, "type": typ, "links": [], "slot_index": len(outputs)})
        else:
            cls = NODE_CLASSES[type_name]
            for name, typ, opts, is_widget in self._spec(cls):
                if is_widget:
                    if name in values:
                        val = values[name]
                    elif isinstance(typ, list):
                        val = opts.get("default", typ[0])
                    else:
                        val = opts.get("default", {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[typ])
                    widgets.append(val)
                    if name in connect:
                        inputs.append({"name": name, "type": typ if isinstance(typ, str) else "COMBO", "link": None,
                                       "widget": {"name": name}})
                else:
                    inputs.append({"name": name, "type": typ if isinstance(typ, str) else "COMBO", "link": None})
            rt = cls.RETURN_TYPES
            rn = getattr(cls, "RETURN_NAMES", rt)
            for i, (t, n) in enumerate(zip(rt, rn)):
                outputs.append({"name": n, "type": t, "links": [], "slot_index": i})
        node = {"id": node_id, "type": type_name, "pos": list(pos), "size": list(size), "flags": {}, "order": node_id,
                "mode": 0, "inputs": inputs, "outputs": outputs,
                "properties": {"Node name for S&R": type_name} if type_name != "Note" else {"text": text},
                "widgets_values": widgets}
        if title:
            node["title"] = title
        if type_name == "Note":
            node["color"] = "#432"
            node["bgcolor"] = "#653"
        self.nodes.append(node)
        for input_name, (src_id, src_slot) in connect.items():
            self.link(src_id, src_slot, node_id, input_name)
        return node_id

    def link(self, src_id, src_slot, dst_id, dst_input):
        src = next(n for n in self.nodes if n["id"] == src_id)
        dst = next(n for n in self.nodes if n["id"] == dst_id)
        out = src["outputs"][src_slot]
        inp = next(i for i in dst["inputs"] if i["name"] == dst_input)
        link_id = self.next_link
        self.next_link += 1
        out["links"].append(link_id)
        inp["link"] = link_id
        self.links.append([link_id, src_id, src_slot, dst_id, dst["inputs"].index(inp), out["type"]])

    def dump(self, path):
        data = {"last_node_id": self.next_id - 1, "last_link_id": self.next_link - 1, "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [60, 60]}},
                "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {path} ({len(self.nodes)} nodes, {len(self.links)} links)")


SECURITY = ("SECURITY: the builder executes model-written Python in Blender. No sandbox.\n"
            "Every script is saved to ComfyUI/output/ai_scene_builder/<session>/scripts/ BEFORE it runs.\n"
            "Set dry_run=true to review scripts first (then use 'AI Script Runner').")

SEEING_IT = ("WHERE YOU SEE THE RESULT\n"
             "  execution_mode=headless (default): Blender runs in the background. You get the preview\n"
             "    render and a .blend file - NOTHING appears in a Blender window you have open.\n"
             "  execution_mode=live: the model builds inside the Blender you are looking at, and the\n"
             "    viewport re-frames on each new part. Requires, in Blender:\n"
             "      1. addon v2.2.0+ installed (blender_scripts/blender_toolbox_addon.py)\n"
             "      2. N-panel > ComfyUI tab > Start Listener\n"
             "      3. AI Scene Builder (Live) > Allow AI code execution = ON")

MODELS = ("MODEL CHOICE DRIVES QUALITY MORE THAN ANY OTHER SETTING\n"
          "  The vision model decides how much of the reference is understood. A small one (4-8B)\n"
          "  reports every part as the same size and misses the fine detail, which produces\n"
          "  featureless blobs. Use the largest vision model you have; the AI Model Config node\n"
          "  prints a TIP naming the best one installed.\n"
          "  The code model decides how well that description becomes geometry: 14B works, 32B is\n"
          "  noticeably better, claude-sonnet-5 via the anthropic provider is better still.")

REPRODUCE_PROMPT = ("Reproduce the object in the reference images as a detailed 3D model: match its shapes, "
                    "proportions, materials and all of its decoration (trim, rivets, windows, motifs, paint bands).")


def wf_complete(out_dir):
    g = Graph()
    g.add("Note", (-40, -560), (760, 440), text=(
        "AI SCENE BUILDER - COMPLETE SCENE (Geekatplay Studio - Vladimir Chopine)\n\n"
        "1. Load 1-3 reference images. The Reference Analyzer runs THREE focused vision passes over each\n"
        "   (structure with real-world size / fine detail / materials) and merges them into one build\n"
        "   specification with every dimension in meters.\n"
        "2. Session: pick a name. Everything (scene.blend, scripts, results, renders) goes to\n"
        "   ComfyUI/output/ai_scene_builder/<name>/. Run again with the same name to keep building.\n"
        "3. Planner splits it into ordered steps - each step an ASSEMBLY with its trim and fittings,\n"
        "   never a lone primitive. Builder executes each: generate -> safety scan -> run in Blender ->\n"
        "   validate polygons/normals/textures/names -> retry on failure -> preview render.\n"
        "4. Read 'Reference brief' and 'Plan' first: if the brief has wrong sizes or too few parts, the\n"
        "   model is too small - see the model note below. Fixing the brief fixes the model.\n"
        "5. Outputs: preview image, blend path, full report, all scripts, validation JSON.\n\n"
        + MODELS + "\n\n" + SEEING_IT + "\n\n" + SECURITY))
    ref1 = g.add("LoadImage", (-40, -80), (315, 314), title="Reference image 1")
    ref2 = g.add("LoadImage", (-40, 280), (315, 314), title="Reference image 2 (optional)")
    sess = g.add("GapAISceneSession", (320, -80), (330, 150), values={"session_name": "my_object"})
    llm = g.add("GapAILLMConfig", (320, 120), (330, 300))
    ana = g.add("GapAIReferenceAnalyzer", (700, -80), (420, 260),
                values={"prompt": REPRODUCE_PROMPT},
                connect={"llm": (llm, 0), "images": (ref1, 0), "images_2": (ref2, 0), "session": (sess, 0)})
    plan = g.add("GapAIScenePlanner", (700, 240), (420, 260),
                 values={"prompt": REPRODUCE_PROMPT, "max_steps": 6},
                 connect={"session": (sess, 0), "llm": (llm, 0), "reference_brief": (ana, 0)})
    build = g.add("GapAISceneBuilder", (1180, -80), (460, 620),
                  values={"stop_on_failure": False},
                  connect={"session": (sess, 0), "llm": (llm, 0), "plan_json": (plan, 0), "reference_brief": (ana, 0)})
    send = g.add("GapAISendSceneToBlender", (1180, 800), (460, 300),
                 connect={"session": (sess, 0)})
    g.add("GapStringViewer", (1700, 1000), (520, 200), title="Sent to Blender",
          connect={"text": (send, 1)})
    g.add("PreviewImage", (1700, -80), (520, 360), title="Preview render", connect={"images": (build, 0)})
    g.add("GapStringViewer", (1700, 320), (520, 300), title="Build report", connect={"text": (build, 2)})
    g.add("GapStringViewer", (1700, 660), (520, 300), title="Plan", connect={"text": (plan, 1)})
    g.add("GapStringViewer", (1180, 580), (460, 200), title="Reference brief (CHECK THIS FIRST)",
          connect={"text": (ana, 0)})
    g.add("GapStringViewer", (700, 540), (420, 200), title="Model info / quality tips",
          connect={"text": (llm, 1)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Scene_Builder_Complete.json"))


def wf_single_step_with_refs(out_dir):
    g = Graph()
    g.add("Note", (-40, -380), (720, 260), text=(
        "AI SCENE BUILDER - SINGLE STEP FROM REFERENCE IMAGES\n\n"
        "One instruction, one Blender script, validated and rendered. Reference images are analyzed first\n"
        "(up to 3 image inputs; batches count as multiple references) so the model matches style, layout,\n"
        "materials and lighting. Re-run with a new instruction to keep adding to the same session scene.\n\n" + SECURITY))
    ref1 = g.add("LoadImage", (-40, -80), (315, 314), title="Reference image 1")
    ref2 = g.add("LoadImage", (-40, 280), (315, 314), title="Reference image 2 (optional)")
    sess = g.add("GapAISceneSession", (320, -80), (330, 150), values={"session_name": "single_step_scene"})
    llm = g.add("GapAILLMConfig", (320, 120), (330, 300))
    ana = g.add("GapAIReferenceAnalyzer", (700, -80), (420, 260),
                values={"prompt": REPRODUCE_PROMPT},
                connect={"llm": (llm, 0), "images": (ref1, 0), "images_2": (ref2, 0), "session": (sess, 0)})
    step = g.add("GapAIStepBuilder", (1180, -80), (460, 600),
                 values={"instruction": "Build the object described in the build specification, complete in one step: every "
                                        "part at the exact sizes and z positions it gives, every decoration it lists (trim "
                                        "rings, rivet rings, windows, motifs, paint bands) using the detail helpers, and the "
                                        "materials from its palette. Add a key light and a camera framing the object.",
                         "step_title": "Reference_Object"},
                 connect={"session": (sess, 0), "llm": (llm, 0), "reference_brief": (ana, 0)})
    g.add("PreviewImage", (1700, -80), (520, 360), title="Preview render", connect={"images": (step, 0)})
    g.add("GapStringViewer", (1700, 320), (520, 300), title="Step report", connect={"text": (step, 2)})
    g.add("GapStringViewer", (1180, 560), (460, 220), title="Reference brief", connect={"text": (ana, 0)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Single_Step_From_References.json"))


def wf_conversational(out_dir):
    g = Graph()
    g.add("Note", (-40, -360), (720, 240), text=(
        "AI SCENE BUILDER - CONVERSATIONAL PASSES\n\n"
        "Type an instruction, queue, look at the render, type the next instruction, queue again.\n"
        "The session .blend is the memory: 'add a bridge over the river', 'make the towers taller',\n"
        "'add fog lights along the path'. Each pass sees the current scene (objects, sizes, materials).\n"
        "Keep the same session_name. Use 'reset' once to start a fresh scene.\n\n" + SECURITY))
    sess = g.add("GapAISceneSession", (-40, -80), (330, 150), values={"session_name": "conversation_scene"})
    llm = g.add("GapAILLMConfig", (-40, 120), (330, 300))
    step = g.add("GapAIStepBuilder", (340, -80), (460, 600),
                 values={"instruction": "Create a 40x40 m grassy terrain with gentle hills, a winding dirt path, and a small "
                                        "stone well near the center. Warm afternoon sun and a camera framing the scene.",
                         "step_title": "Pass"},
                 connect={"session": (sess, 0), "llm": (llm, 0)})
    g.add("PreviewImage", (860, -80), (520, 360), title="Preview render", connect={"images": (step, 0)})
    g.add("GapStringViewer", (860, 320), (520, 300), title="Step report", connect={"text": (step, 2)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Step_Builder_Conversational.json"))


def wf_review(out_dir):
    g = Graph()
    g.add("Note", (-40, -400), (760, 280), text=(
        "AI SCENE BUILDER - REVIEW BEFORE EXECUTING (safest flow)\n\n"
        "The Step Builder runs with dry_run=true: it generates and SAVES the script but does not run it.\n"
        "Read it in the 'Generated script' viewer (also saved under the session's scripts/ folder),\n"
        "paste/edit it into 'AI Script Runner' and queue again to execute with the same validation,\n"
        "auto-fixes and preview render. Nothing runs in Blender that you have not seen.\n\n" + SECURITY))
    sess = g.add("GapAISceneSession", (-40, -80), (330, 150), values={"session_name": "reviewed_scene"})
    llm = g.add("GapAILLMConfig", (-40, 120), (330, 300))
    step = g.add("GapAIStepBuilder", (340, -80), (460, 600),
                 values={"instruction": "Build a wooden dock extending 12 m over a water plane, with mooring posts, a lantern at the end and a rowing boat tied to it.",
                         "step_title": "Dock", "dry_run": True, "render_preview": False},
                 connect={"session": (sess, 0), "llm": (llm, 0)})
    g.add("GapStringViewer", (860, -80), (560, 420), title="Generated script (review, then paste into the runner)",
          connect={"text": (step, 3)})
    run = g.add("GapAIScriptRunner", (860, 400), (560, 560), values={"title": "Reviewed dock script"},
                connect={"session": (sess, 0), "llm": (llm, 0)})
    g.add("PreviewImage", (1480, 400), (520, 360), title="Preview render", connect={"images": (run, 0)})
    g.add("GapStringViewer", (1480, 800), (520, 240), title="Run report", connect={"text": (run, 2)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Script_Review_Then_Run.json"))


def wf_send_to_blender(out_dir):
    g = Graph()
    g.add("Note", (-40, -430), (780, 360), text=(
        "BUILD AND SEND TO YOUR OPEN BLENDER (Geekatplay Studio - Vladimir Chopine)\n\n"
        "Headless builds are fast and safe, but they happen in a background Blender. This workflow\n"
        "builds the scene and then loads the finished .blend into the Blender you already have open,\n"
        "with materials, collections, lights and cameras intact.\n\n"
        "1. 'Blender Bridge Check' proves ComfyUI can reach your Blender BEFORE anything long runs.\n"
        "   It reports the Blender version, the addon version, what is in your scene, and whether\n"
        "   live AI execution is allowed. Queue it on its own any time to test the connection.\n"
        "2. The Step Builder builds the object headlessly (fast, cannot disturb your open file).\n"
        "3. 'Send Scene to Blender' imports the result into your session:\n"
        "     append - copies the objects into your current scene (default; your work is kept)\n"
        "     link   - references them read-only from the .blend\n"
        "     open   - REPLACES your open file; unsaved work is lost\n\n"
        "IN BLENDER (needed by both nodes): install addon v2.2.1 or newer, then press N in the 3D\n"
        "viewport > ComfyUI tab > Start Listener. 'Allow AI code execution' is NOT required to send\n"
        "a finished scene - that switch only gates execution_mode=live.\n\n" + SECURITY))
    check = g.add("GapAIBlenderBridgeCheck", (-40, -40), (360, 200))
    g.add("GapStringViewer", (-40, 200), (360, 280), title="Bridge status", connect={"text": (check, 2)})
    sess = g.add("GapAISceneSession", (400, -40), (330, 150), values={"session_name": "send_to_blender"})
    llm = g.add("GapAILLMConfig", (400, 160), (330, 300))
    step = g.add("GapAIStepBuilder", (780, -40), (460, 600),
                 values={"instruction": "Build a detailed brass desk bell: a domed body 0.09 m across and "
                                        "0.07 m tall on a stepped base, with a trim ring at the rim, a ring "
                                        "of 16 rivets above it, and a turned handle 0.05 m tall on top.",
                         "step_title": "Object"},
                 connect={"session": (sess, 0), "llm": (llm, 0)})
    send = g.add("GapAISendSceneToBlender", (1290, -40), (460, 320),
                 connect={"session": (sess, 0)})
    g.add("PreviewImage", (1290, 320), (460, 340), title="Headless preview", connect={"images": (step, 0)})
    g.add("GapStringViewer", (1800, -40), (460, 260), title="Import report", connect={"text": (send, 1)})
    g.add("GapStringViewer", (1800, 240), (460, 300), title="Build report", connect={"text": (step, 2)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Build_And_Send_To_Blender.json"))


def wf_validate(out_dir):
    g = Graph()
    g.add("Note", (-40, -320), (720, 200), text=(
        "AI SCENE VALIDATOR - QA for any session scene\n\n"
        "Checks every mesh for non-manifold edges, loose geometry, zero-area faces, flipped normals,\n"
        "duplicate vertices, n-gons; textures for missing files; names for Cube.001-style defaults;\n"
        "scene for camera/lights. Optional auto-fixes rewrite the session .blend. Renders a preview.\n"
        "Point 'blend_file' on the session node at any .blend to validate files not made by the builder."))
    sess = g.add("GapAISceneSession", (-40, -80), (330, 150), values={"session_name": "castle_scene"})
    val = g.add("GapAISceneValidator", (340, -80), (420, 360), connect={"session": (sess, 0)})
    g.add("PreviewImage", (820, -80), (520, 360), title="Preview render", connect={"images": (val, 0)})
    g.add("GapStringViewer", (820, 320), (520, 320), title="Validation report", connect={"text": (val, 1)})
    g.dump(os.path.join(out_dir, "Geekatplay_AI_Scene_Validator.json"))


if __name__ == "__main__":
    out = os.path.join(ROOT, "workflows")
    os.makedirs(out, exist_ok=True)
    wf_complete(out)
    wf_single_step_with_refs(out)
    wf_conversational(out)
    wf_review(out)
    wf_send_to_blender(out)
    wf_validate(out)
