# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox Test Suite - AI Scene Builder
"""Offline unit tests for the AI Scene Builder engine plus one real headless-Blender integration test
(skipped when Blender is not installed). No model calls are made."""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
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


class TestAutoModelSelection(unittest.TestCase):
    """'auto' must pick the strongest installed model for each job, offline and deterministically."""

    DETAILS = {
        "qwen2.5-coder:14b": {"params_b": 14.8, "capabilities": ["completion", "tools", "insert"]},
        "qwen2.5-coder:32b": {"params_b": 32.8, "capabilities": ["completion", "tools", "insert"]},
        "qwen2.5-coder:1.5b-base": {"params_b": 1.5, "capabilities": ["completion", "insert"]},
        "qwen2.5vl:7b": {"params_b": 8.3, "capabilities": ["completion", "vision"]},
        "qwen3.8:latest": {"params_b": 27.3, "capabilities": ["completion", "vision", "tools"]},
        "qwen3:30b": {"params_b": 30.5, "capabilities": ["completion", "tools"]},
        "nomic-embed-text:latest": {"params_b": 0.137, "capabilities": ["embedding"]},
    }

    def client(self):
        from nodes.ai_builder.llm import LLMClient
        return LLMClient(LLMConfig())

    def test_picks_largest_capable_model_for_each_job(self):
        picks = self.client().pick_best_models(self.DETAILS)
        self.assertEqual(picks["code"], "qwen2.5-coder:32b", "should prefer the biggest real coder")
        self.assertEqual(picks["vision"], "qwen3.8:latest",
                         "vision must be chosen on size, even behind a ':latest' tag")
        self.assertEqual(picks["embed"], "nomic-embed-text:latest")
        self.assertIn("32.8B", picks["reasons"]["code"])

    def test_base_models_are_never_chosen(self):
        only_base = {"qwen2.5-coder:1.5b-base": self.DETAILS["qwen2.5-coder:1.5b-base"],
                     "qwen3:30b": self.DETAILS["qwen3:30b"]}
        picks = self.client().pick_best_models(only_base)
        self.assertEqual(picks["code"], "qwen3:30b", "a -base model cannot follow instructions")

    def test_no_models_installed_is_handled(self):
        picks = self.client().pick_best_models({})
        self.assertIsNone(picks["code"])
        self.assertIsNone(picks["vision"])

    def test_no_vision_model_installed(self):
        picks = self.client().pick_best_models({"qwen3:30b": self.DETAILS["qwen3:30b"]})
        self.assertIsNone(picks["vision"])
        self.assertEqual(picks["code"], "qwen3:30b")

    def test_cloud_provider_auto_resolves_without_ollama(self):
        from nodes.ai_builder_nodes import GapAILLMConfig
        cfg, info = GapAILLMConfig().build("anthropic", "auto", "auto", "", 0.2, 16384, 4096)
        self.assertEqual(cfg["model"], config.DEFAULT_ANTHROPIC_MODEL)
        self.assertEqual(cfg["vision_model"], cfg["model"], "vision should reuse the main model")
        self.assertEqual(cfg["url"], config.DEFAULT_ANTHROPIC_URL)


class TestShippedWorkflows(unittest.TestCase):
    """Every workflow in workflows/ must stay loadable: valid links, and widget values that still
    line up with the node definitions. Nodes gain inputs over time and a stale workflow silently
    shifts every setting after the new one."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))
        import check_workflows
        cls.check = check_workflows
        cls.nodes = check_workflows.toolbox_nodes()
        cls.folder = os.path.join(ROOT_DIR, "workflows")

    def test_every_workflow_is_structurally_sound(self):
        problems = []
        for name in sorted(os.listdir(self.folder)):
            if not name.endswith(".json"):
                continue
            errors, _warnings, _notes = self.check.check_workflow(
                os.path.join(self.folder, name), self.nodes, None)
            if errors:
                problems.append(name + ":\n    " + "\n    ".join(errors))
        self.assertEqual(problems, [], msg="broken workflows:\n" + "\n".join(problems))

    def test_seed_widgets_are_accounted_for(self):
        """A seed serialises two values; forgetting the companion makes every workflow look broken."""
        spec = self.check.widget_spec(self.nodes["Geekatplay_Tripo_ModelGen"])
        self.assertIn("model_seed", spec)
        self.assertIn("model_seed:control_after_generate", spec)
        self.assertEqual(spec.index("model_seed") + 1, spec.index("model_seed:control_after_generate"))

    def test_readme_lists_every_workflow(self):
        """A workflow nobody can find is a workflow that does not exist."""
        with open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        listed = set(re.findall(r"`(Geekatplay_[\w.]+\.json|autorig_api\.json)`", readme))
        on_disk = {f for f in os.listdir(self.folder) if f.endswith(".json")}
        self.assertEqual(sorted(on_disk - listed), [], msg="workflows missing from the README")
        self.assertEqual(sorted(listed - on_disk), [], msg="README names workflows that do not exist")

    def test_templates_carry_no_machine_specific_paths(self):
        """A shipped template must not reference a file generated on someone else's computer."""
        offenders = []
        for name in sorted(os.listdir(self.folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.folder, name), encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                if node.get("type") != "Preview3D":
                    continue
                for value in node.get("widgets_values") or []:
                    if isinstance(value, str) and value.endswith((".glb", ".gltf", ".obj", ".fbx")):
                        offenders.append(f"{name} node {node.get('id')}: {value}")
        self.assertEqual(offenders, [], msg="stale asset references:\n" + "\n".join(offenders))


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


class TestBlenderBridge(unittest.TestCase):
    """Offline checks of the ComfyUI -> running Blender bridge (no Blender required)."""

    def test_ping_reports_dead_port_with_actionable_error(self):
        from nodes.ai_builder import bridge
        status = bridge.ping("127.0.0.1", 8198, timeout=2)
        self.assertFalse(status["ok"])
        self.assertEqual(status["stage"], "connect")
        self.assertIn("Start Listener", status["error"])

    def test_describe_formats_both_states(self):
        from nodes.ai_builder import bridge
        down = bridge.describe({"ok": False, "error": "nope"})
        self.assertIn("BRIDGE NOT WORKING", down)
        up = bridge.describe({
            "ok": True, "blender_version": "5.2.2", "addon_version": "2.2.1",
            "listener": {"host": "127.0.0.1", "port": 8119, "running": True},
            "blend_file": "", "scene_name": "Scene", "object_count": 3, "mesh_count": 2,
            "light_count": 1, "has_camera": True, "render_engine": "CYCLES", "ai_exec_allowed": False})
        self.assertIn("BRIDGE OK", up)
        self.assertIn("5.2.2", up)
        self.assertIn("BLOCKED", up)
        self.assertIn("Allow AI code execution", up)
        allowed = bridge.describe({
            "ok": True, "blender_version": "5.2.2", "addon_version": "2.2.1",
            "listener": {"host": "127.0.0.1", "port": 8119, "running": True},
            "blend_file": "x.blend", "scene_name": "Scene", "object_count": 1, "mesh_count": 1,
            "light_count": 0, "has_camera": False, "render_engine": "CYCLES", "ai_exec_allowed": True})
        self.assertIn("ALLOWED", allowed)

    def test_stale_addon_is_reported(self):
        from nodes.ai_builder import bridge
        shipped = bridge.repo_addon_version()
        self.assertIsNotNone(shipped, "could not read the addon version from bl_info")
        older = ".".join(str(p) for p in (shipped[0], shipped[1], max(shipped[2] - 1, 0)))
        status = {"ok": True, "addon_version": older, "blender_version": "5.2.2",
                  "listener": {"host": "127.0.0.1", "port": 8119, "running": True},
                  "blend_file": "", "scene_name": "S", "object_count": 1, "mesh_count": 1,
                  "light_count": 0, "has_camera": False, "render_engine": "CYCLES",
                  "ai_exec_allowed": False}
        self.assertIn("OUT OF DATE", bridge.addon_version_warning(status))
        self.assertIn("OUT OF DATE", bridge.describe(status))
        current = dict(status, addon_version=".".join(str(p) for p in shipped))
        self.assertEqual(bridge.addon_version_warning(current), "")
        self.assertNotIn("OUT OF DATE", bridge.describe(current))

    def test_append_rejects_missing_file(self):
        from nodes.ai_builder import bridge
        result = bridge.append_blend(os.path.join(tempfile.gettempdir(), "definitely_missing.blend"))
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_bridge_check_node_does_not_raise_by_default(self):
        from nodes.ai_builder_nodes import GapAIBlenderBridgeCheck
        out = GapAIBlenderBridgeCheck().check("127.0.0.1", 8198, 2, False)["result"]
        connected, live_allowed, text, status_json = out
        self.assertFalse(connected)
        self.assertFalse(live_allowed)
        self.assertIn("BRIDGE NOT WORKING", text)
        self.assertFalse(json.loads(status_json)["ok"])
        with self.assertRaises(RuntimeError):
            GapAIBlenderBridgeCheck().check("127.0.0.1", 8198, 2, True)

    def test_send_node_requires_a_blend(self):
        from nodes.ai_builder_nodes import GapAISendSceneToBlender
        sent, report, _ = GapAISendSceneToBlender().send(
            "append", "127.0.0.1", 8198, True, False, 5, session=None, blend_path="")["result"]
        self.assertFalse(sent)
        self.assertIn("session or an explicit blend_path", report)
        sent, report, _ = GapAISendSceneToBlender().send(
            "append", "127.0.0.1", 8198, True, False, 5, session=None,
            blend_path=os.path.join(tempfile.gettempdir(), "nope.blend"))["result"]
        self.assertFalse(sent)
        self.assertIn("No .blend", report)


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

    def test_prompt_signatures_match_the_real_helpers(self):
        """The signatures advertised to the model must be the real ones.

        Every build failure in the wild so far was the model calling a helper with the wrong
        arguments. If this list drifts from gap_helpers.py the model is being lied to, so the
        signatures are introspected inside Blender and compared.
        """
        from nodes.ai_builder.prompts import HELPER_SIGNATURES
        probe = ("import inspect, json, sys\n"
                 "sys.path.insert(0, r'{d}')\n"
                 "import gap_helpers as g\n"
                 "out = {{}}\n"
                 "for n in dir(g.Builder):\n"
                 "    f = getattr(g.Builder, n)\n"
                 "    if callable(f) and not n.startswith('_'):\n"
                 "        out['B.' + n] = str(inspect.signature(f)).replace('(self, ', '(').replace('(self)', '()')\n"
                 "for n in g.__all__:\n"
                 "    f = getattr(g, n, None)\n"
                 "    if callable(f) and not isinstance(f, type) and getattr(f, '__module__', '') == 'gap_helpers':\n"
                 "        out[n] = str(inspect.signature(f))\n"
                 "print('SIGJSON' + json.dumps(out))\n").format(d=config.BLENDER_SCRIPTS_DIR.replace("\\", "\\\\"))
        script = os.path.join(tempfile.mkdtemp(prefix="sigprobe-"), "probe.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(probe)
        proc = subprocess.run([self.blender, "--background", "--factory-startup", "--python", script],
                              capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
        line = next((l for l in (proc.stdout or "").splitlines() if l.startswith("SIGJSON")), None)
        self.assertIsNotNone(line, msg=f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")
        real = json.loads(line[len("SIGJSON"):])

        # Index the prompt by the declaration lines only: a line whose first token is "name(".
        # Prose elsewhere may mention a helper by name without giving its signature.
        declared = {}
        for line in HELPER_SIGNATURES.splitlines():
            stripped = line.strip()
            if "(" not in stripped or stripped.startswith("#"):
                continue
            head = stripped[:stripped.index("(")]
            if head and all(ch.isalnum() or ch in "._" for ch in head):
                declared.setdefault(head, stripped[len(head):])

        wrong = []
        for name, signature in real.items():
            advertised = declared.get(name)
            if advertised is None:
                continue  # not advertised to the model, nothing to keep in sync
            norm = lambda s: s.replace(" ", "").replace("'", "").replace('"', "")
            if norm(advertised) != norm(signature):
                wrong.append(f"{name}\n      prompt: {advertised}\n      real  : {signature}")
        self.assertEqual(wrong, [], msg="prompts.HELPER_SIGNATURES is out of sync with gap_helpers.py:\n"
                                        + "\n".join(wrong))

    def test_signature_mistakes_produce_a_corrective_hint(self):
        """A wrong helper call must come back with the correct signature, not just a TypeError."""
        tmp = tempfile.mkdtemp(prefix="ai-hint-")
        s = SceneSession("hint", root=os.path.join(tmp, "hint")).open()
        script = s.next_script_path(1)
        with open(script, "w", encoding="utf-8") as f:
            f.write("from gap_helpers import *\nB = Builder()\n"
                    "panel_seams(B, (0,0,0), 0.05, 0.0)\n")
        runner = BlenderRunner(s, blender_path=self.blender, timeout=300)
        result = runner.run_headless(script, validate=False, render=None)
        self.assertFalse(result["ok"])
        self.assertIn("panel_seams", result["error"])
        self.assertIn("api_hint", result, msg="no corrective signature hint was returned")
        self.assertIn("panel_seams(builder, center, radius, z0, z1", result["api_hint"])
        self.assertIn("KEYWORD-ONLY", result["api_hint"])
        feedback = feedback_for_retry(result)
        self.assertIn("panel_seams(builder", feedback, msg="the hint never reaches the retry prompt")

    def test_debug_log_records_every_stage(self):
        tmp = tempfile.mkdtemp(prefix="ai-debug-")
        s = SceneSession("dbg", root=os.path.join(tmp, "dbg")).open()
        s.debug("PLANNER - RAW MODEL REPLY", '{"steps": []}', divider=True)
        s.debug("STEP 1 ATTEMPT 1 - GENERATED SCRIPT", "from gap_helpers import *\n")
        text = s.read_debug()
        self.assertIn("PLANNER - RAW MODEL REPLY", text)
        self.assertIn("GENERATED SCRIPT", text)
        self.assertIn("from gap_helpers import *", text)
        self.assertTrue(os.path.exists(s.debug_path))

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
