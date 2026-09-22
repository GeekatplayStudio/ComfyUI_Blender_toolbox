# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox Test Suite - AI Scene Builder
"""Offline unit tests for the AI Scene Builder engine plus one real headless-Blender integration test
(skipped when Blender is not installed). No model calls are made."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: F401,E402  (ComfyUI mocks)

from nodes.ai_builder import config  # noqa: E402
from nodes.ai_builder.agent import SceneBuilderAgent  # noqa: E402
from nodes.ai_builder.blender_runner import BlenderRunner  # noqa: E402
from nodes.ai_builder.llm import LLMConfig, extract_code, extract_json, strip_thinking  # noqa: E402
from nodes.ai_builder.prompts import build_codegen_messages, build_planner_messages  # noqa: E402
from nodes.ai_builder.rag import ReferenceIndex, chunk_markdown, tokenize  # noqa: E402
from nodes.ai_builder.safety import scan_code  # noqa: E402
from nodes.ai_builder.session import SceneSession  # noqa: E402
from nodes.ai_builder.validation import feedback_for_retry, summarize_validation, validation_passed  # noqa: E402
from nodes.utils_blender import get_blender_path  # noqa: E402


class TestLLMHelpers(unittest.TestCase):
    def test_extract_code_prefers_fence(self):
        text = "Here you go:\n```python\nimport bpy\nprint('x')\n```\nDone."
        self.assertEqual(extract_code(text), "import bpy\nprint('x')\n")

    def test_extract_code_joins_multiple_fences_and_strips_thinking(self):
        text = "<think>plan</think>```py\nimport bpy\n```\ntext\n```python\nbpy.data\n```"
        code = extract_code(text)
        self.assertIn("import bpy", code)
        self.assertIn("bpy.data", code)
        self.assertNotIn("<think>", code)

    def test_extract_code_fallback_and_empty(self):
        self.assertTrue(extract_code("import bpy\nbpy.data.objects").startswith("import bpy"))
        self.assertEqual(extract_code("no code here"), "")

    def test_extract_json_variants(self):
        self.assertEqual(extract_json('prefix {"a": 1} suffix'), {"a": 1})
        self.assertEqual(extract_json('```json\n[1, 2]\n```'), [1, 2])
        self.assertIsNone(extract_json("nothing"))
        self.assertEqual(strip_thinking("<think>x\ny</think>ok"), "ok")

    def test_llm_config_defaults(self):
        cfg = LLMConfig(model="qwen2.5-coder:7b")
        self.assertEqual(cfg["provider"], "ollama")
        self.assertEqual(cfg["model"], "qwen2.5-coder:7b")
        self.assertIn("qwen2.5-coder:7b", cfg.describe())


class TestSafetyScan(unittest.TestCase):
    def test_blocks_dangerous_patterns(self):
        code = "import os\nimport subprocess\nos.system('rm -rf /')\nbpy.ops.wm.save_mainfile()\nopen('x','w')\neval('1')\n"
        scan = scan_code(code)
        self.assertGreaterEqual(len(scan["blocked"]), 5)

    def test_allows_normal_scene_code(self):
        code = ("from gap_helpers import *\nimport bpy, math, random\ncoll = get_or_create_collection('X')\n"
                "B = Builder()\nB.box((0,0,1),(2,2,2))\nlog_built(B.build('Cube_A', coll, make_material('Mat_A',(1,0,0))))\n"
                "# os.system in a comment is fine\n")
        scan = scan_code(code)
        self.assertEqual(scan["blocked"], [])

    def test_warns_on_destructive_helpers(self):
        scan = scan_code("from gap_helpers import *\nclear_scene()\ndelete_objects(['a'])\n")
        self.assertEqual(scan["blocked"], [])
        self.assertEqual(len(scan["warnings"]), 2)


class TestSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ai-session-")

    def test_create_add_turn_reload(self):
        s = SceneSession("My Scene!", root=os.path.join(self.tmp, "my_scene")).open()
        self.assertEqual(s.name, "My_Scene")
        self.assertTrue(os.path.isdir(s.scripts_dir))
        self.assertFalse(s.blend_exists())
        p1 = s.next_script_path(1)
        self.assertTrue(p1.endswith("step_001_attempt1.py"))
        s.add_turn({"title": "T", "instruction": "build", "ok": True, "summary": "did it",
                    "scene": {"object_count": 2, "objects": [{"name": "A", "type": "MESH"}], "collections": ["C"]}})
        s2 = SceneSession("My Scene!", root=os.path.join(self.tmp, "my_scene")).open()
        self.assertEqual(s2.turn_count, 1)
        self.assertIn("Step 1", s2.history_summary())
        self.assertIn("objects: 2", s2.scene_summary())
        self.assertTrue(s2.next_script_path(1).endswith("step_002_attempt1.py"))

    def test_reset_archives(self):
        s = SceneSession("r", root=os.path.join(self.tmp, "r")).open()
        s.add_turn({"title": "a", "instruction": "b", "ok": True})
        s = SceneSession("r", root=os.path.join(self.tmp, "r")).open(reset=True)
        self.assertEqual(s.turn_count, 0)
        self.assertTrue(any(d.startswith("archive_") for d in os.listdir(s.root)))

    def test_payload_roundtrip(self):
        s = SceneSession("p", root=os.path.join(self.tmp, "p")).open()
        s2 = SceneSession.from_payload(s.to_payload())
        self.assertEqual(s2.root, s.root)
        with self.assertRaises(ValueError):
            SceneSession.from_payload({})


class TestRAG(unittest.TestCase):
    def test_chunking_and_tokens(self):
        chunks = chunk_markdown("# A\ntext a\n## B\ntext b " * 3, "f.md")
        self.assertTrue(all(c["source"] == "f.md" for c in chunks))
        self.assertIn("bpy.data.objects", tokenize("use bpy.data.objects to LINK"))

    def test_reference_docs_retrieval(self):
        idx = ReferenceIndex()
        self.assertGreater(len(idx.chunks), 20, "reference docs missing?")
        top = idx.retrieve("scatter pine trees on the terrain surface", k=5)
        self.assertTrue(any("scatter" in (c["text"] + c["title"]).lower() for c in top))
        core_first = idx.retrieve("anything", k=3)
        self.assertTrue(any(c["source"].startswith("00_") or c["source"].startswith("06_") for c in core_first))
        ctx = ReferenceIndex.format_context(top, max_chars=2000)
        self.assertLessEqual(len(ctx), 2200)

    def test_extra_text_indexed(self):
        idx = ReferenceIndex(extra_text="## Custom\nmy custom zorblax helper builds spires")
        self.assertTrue(any("zorblax" in c["text"] for c in idx.retrieve("zorblax spire", k=2)))


class TestPromptsAndPlanning(unittest.TestCase):
    def test_codegen_messages(self):
        msgs = build_codegen_messages("Add a fence", "Scene: 3 objects", "- Step 1: OK tower", "brief", "ctx", "5.2",
                                      "Step_02_Fence", False)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Step_02_Fence", msgs[0]["content"])
        self.assertIn("EXISTING scene", msgs[0]["content"])
        self.assertIn("Add a fence", msgs[1]["content"])
        retry = build_codegen_messages("x", "", "", "", "", "5.2", "c", True, feedback="boom", previous_code="import bpy")
        self.assertEqual(len(retry), 4)
        self.assertIn("boom", retry[-1]["content"])

    def test_planner_messages_and_manual_plan(self):
        msgs = build_planner_messages("village", "", "", 5)
        self.assertIn("Maximum 5 steps", msgs[0]["content"])
        steps = SceneBuilderAgent.parse_manual_plan("1. Ground: make terrain\n- Tower: build a tower\nJust a line")
        self.assertEqual([s["title"] for s in steps], ["Ground", "Tower", "Step 3"])
        self.assertEqual(steps[2]["instruction"], "Just a line")
        js = SceneBuilderAgent.parse_manual_plan(json.dumps({"steps": [{"title": "A", "instruction": "do a"}]}))
        self.assertEqual(js[0]["title"], "A")
        self.assertEqual(SceneBuilderAgent.parse_manual_plan(""), [])


class TestValidationText(unittest.TestCase):
    def sample(self, passed=True):
        return {"passed": passed, "totals": {"mesh_objects": 2, "vertices": 100, "faces": 50, "triangles": 96,
                                            "nonmanifold_edges": 0 if passed else 3, "errors": 0 if passed else 1},
                "issues": [] if passed else [{"severity": "error", "category": "topology", "object": "Wall", "message": "3 non-manifold edges"}],
                "auto_fixes": ["Wall: merged 2 duplicate vertices"]}

    def test_summary_and_feedback(self):
        self.assertTrue(validation_passed(self.sample(True)))
        text = summarize_validation(self.sample(False))
        self.assertIn("FAILED", text)
        self.assertIn("non-manifold", text)
        fb = feedback_for_retry({"error": "ValueError: bad", "traceback": "Traceback...ValueError: bad",
                                 "validation": self.sample(False), "stdout": "BUILT Wall 10"})
        self.assertIn("PYTHON EXCEPTION", fb)
        self.assertIn("VALIDATION ERRORS", fb)
        self.assertIn("BUILT Wall", fb)


class TestRunnerJob(unittest.TestCase):
    def test_make_job_paths(self):
        tmp = tempfile.mkdtemp(prefix="ai-job-")
        s = SceneSession("j", root=os.path.join(tmp, "j")).open()
        runner = BlenderRunner(s, blender_path="C:/nonexistent/blender.exe")
        script = s.next_script_path(1)
        with open(script, "w") as f:
            f.write("print(1)\n")
        job, job_path = runner.make_job(script, render={"path": "x.png"})
        self.assertEqual(job["blend_in"], "")
        self.assertEqual(job["blend_out"], s.blend_path)
        self.assertTrue(os.path.exists(job_path))
        self.assertEqual(job["helpers_dir"], config.BLENDER_SCRIPTS_DIR)
        result = runner.run_headless(script)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])


class TestBlenderIntegration(unittest.TestCase):
    """Real headless Blender run of step_runner + gap_helpers + validator (skipped without Blender)."""

    @classmethod
    def setUpClass(cls):
        cls.blender = get_blender_path()
        if not cls.blender:
            raise unittest.SkipTest("Blender not installed")

    def test_helper_script_builds_valid_scene(self):
        tmp = tempfile.mkdtemp(prefix="ai-blender-")
        s = SceneSession("it", root=os.path.join(tmp, "it")).open()
        script = s.next_script_path(1)
        with open(script, "w", encoding="utf-8") as f:
            f.write(
                "from gap_helpers import *\n"
                "coll = get_or_create_collection('Step_01_Test')\n"
                "m = make_material('Mat_Test', (0.5, 0.5, 0.5), roughness=0.7, noise=dict(scale=4, bump=0.1))\n"
                "g = terrain('Ground', size=(10, 10), resolution=8, height=0.5, collection=coll, material=m)\n"
                "B = Builder(); B.box((0, 0, 1), (2, 2, 2)); B.cylinder((0, 0, 2), (0, 0, 4), 0.6, n=12)\n"
                "B.cone((0, 0, 4), (0, 0, 5.5), 0.8, 0.05); B.sphere((0, 0, 5.7), 0.2); B.torus((0, 0, 2.1), 1.3, 0.1)\n"
                "B.tube([(1, 0, 0.5), (2, 1, 1.5)], 0.1); B.prism([(-0.5, 0), (0.5, 0), (0, 1)], 0.3, (3, 0, 0))\n"
                "t = B.build('Tower_Test', coll, m); log_built(t); log_built(g)\n"
                "trees = scatter(t, 3, (-4, 4, -4, 4), seed=1, surface=g, collection=coll, name_prefix='Copy')\n"
                "add_sun(45, 120, 3); frame_camera_to_scene(add_camera())\n"
            )
        runner = BlenderRunner(s, blender_path=self.blender, timeout=300)
        result = runner.run_headless(script, validate=True, auto_fix_normals=True, auto_fix_doubles=True,
                                     render={"path": s.render_path_for(script), "engine": "CYCLES", "width": 128,
                                             "height": 96, "samples": 2})
        self.assertTrue(result["ok"], msg=result.get("error") or result.get("blender_output"))
        v = result["validation"]
        self.assertTrue(v["passed"], msg=json.dumps(v["issues"], indent=1))
        self.assertEqual(v["totals"]["nonmanifold_edges"], 0)
        self.assertEqual(v["totals"]["flipped_normals"], 0)
        self.assertGreaterEqual(v["totals"]["mesh_objects"], 5)
        self.assertTrue(os.path.exists(result["render_path"]))
        self.assertTrue(s.blend_exists())
        self.assertEqual({b["name"] for b in result["built"]}, {"Tower_Test", "Ground"})
        self.assertIn("Camera_Main", result["scene"]["cameras"])

    def test_failing_script_keeps_blend_and_reports(self):
        tmp = tempfile.mkdtemp(prefix="ai-blender-fail-")
        s = SceneSession("f", root=os.path.join(tmp, "f")).open()
        script = s.next_script_path(1)
        with open(script, "w", encoding="utf-8") as f:
            f.write("from gap_helpers import *\nimport bpy\nme = bpy.data.meshes.new('Bad')\n"
                    "me.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)]); me.update()\n"
                    "ob = bpy.data.objects.new('Bad_Object', me); bpy.context.scene.collection.objects.link(ob)\n"
                    "raise ValueError('intentional')\n")
        runner = BlenderRunner(s, blender_path=self.blender, timeout=300)
        result = runner.run_headless(script, validate=True, render=None)
        self.assertFalse(result["ok"])
        self.assertIn("intentional", result["error"])
        self.assertIn("Traceback", result["traceback"])
        self.assertFalse(s.blend_exists(), "a failed step must not write the session .blend")
        self.assertTrue(result["saved_blend"].endswith("_FAILED_STEP.blend"))


if __name__ == "__main__":
    unittest.main()
