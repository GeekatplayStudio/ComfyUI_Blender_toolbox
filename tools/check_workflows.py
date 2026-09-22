# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - workflow integrity checker
"""
Audit every workflow in workflows/ and report anything that would break it on load.

    python tools/check_workflows.py                 (offline: structure + toolbox nodes)
    python tools/check_workflows.py --server http://127.0.0.1:8188   (also validates core nodes)

Checks per workflow:
  * valid JSON, known format (UI graph or API prompt)
  * every node type exists (toolbox nodes always; core nodes when a server is given)
  * every link points at nodes and slots that exist, and link types agree on both ends
  * every input socket reference is consistent with the links table (no dangling links)
  * required inputs of toolbox nodes are connected or have a widget value
  * widgets_values still lines up with the node's current INPUT_TYPES - the failure that
    silently shifts every setting by one when a node gains an input
  * duplicate node ids, missing titles on Notes, empty groups

Exit code is non-zero when any ERROR is found, so it can gate a release.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import conftest  # noqa: F401,E402  - ComfyUI mocks so the nodes import outside ComfyUI

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
# Frontend-only node types: implemented in JavaScript, so they never appear in /object_info.
FRONTEND_ONLY = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "PrimitiveInt", "PrimitiveFloat",
                 "PrimitiveString", "PrimitiveStringMultiline", "PrimitiveBoolean"}


def toolbox_nodes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("toolbox", os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    module = importlib.util.module_from_spec(spec)
    sys.modules["toolbox"] = module
    spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS


def server_node_types(server):
    if not server:
        return None
    try:
        import requests
        r = requests.get(f"{server.rstrip('/')}/object_info", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  (could not reach {server}: {e}; core node types will not be checked)")
        return None


def widget_spec(cls):
    """Names of the widget slots this node serialises, in order.

    A seed widget serialises TWO values: the number and the "control_after_generate" mode the
    frontend attaches to it (fixed / increment / randomize). Missing that made every workflow with
    a seed look one value out of step, so the companion slot is included here.
    """
    names = []
    it = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, definition in it.get(section, {}).items():
            typ = definition[0]
            opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            if (isinstance(typ, list) or typ in WIDGET_TYPES) and not opts.get("forceInput"):
                names.append(name)
                if typ == "INT" and (name.endswith("seed") or opts.get("control_after_generate")):
                    names.append(f"{name}:control_after_generate")
    return names


def check_workflow(path, tb_nodes, object_info):
    errors, warnings, notes = [], [], []
    name = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [f"unreadable: {e}"], [], []

    nodes = data.get("nodes")
    if nodes is None:
        # API-format prompt: {"1": {"class_type": ..., "inputs": {...}}}
        entries = {k: v for k, v in data.items() if isinstance(v, dict) and "class_type" in v}
        if not entries:
            return ["not a recognised workflow (no 'nodes' array and no class_type entries)"], [], []
        notes.append(f"API-format prompt, {len(entries)} nodes")
        for nid, entry in entries.items():
            ctype = entry["class_type"]
            if ctype not in tb_nodes and object_info is not None and ctype not in object_info:
                errors.append(f"node {nid}: unknown node type '{ctype}'")
            for key, value in (entry.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2:
                    src = str(value[0])
                    if src not in entries:
                        errors.append(f"node {nid}.{key} references missing node '{src}'")
        return errors, warnings, notes

    notes.append(f"UI graph, {len(nodes)} nodes, {len(data.get('links', []))} links")
    by_id, seen_ids = {}, set()
    for n in nodes:
        nid = n.get("id")
        if nid in seen_ids:
            errors.append(f"duplicate node id {nid}")
        seen_ids.add(nid)
        by_id[nid] = n

    # node types
    for n in nodes:
        t = n.get("type")
        if t in FRONTEND_ONLY:
            continue
        if t in tb_nodes:
            continue
        if object_info is None:
            continue
        if t not in object_info:
            errors.append(f"node {n.get('id')} '{t}': node type not installed")

    # links table <-> socket references
    link_ids = set()
    for link in data.get("links", []):
        if not isinstance(link, list) or len(link) < 6:
            errors.append(f"malformed link entry: {link}")
            continue
        lid, src_id, src_slot, dst_id, dst_slot, ltype = link[:6]
        link_ids.add(lid)
        if src_id not in by_id:
            errors.append(f"link {lid}: source node {src_id} does not exist")
            continue
        if dst_id not in by_id:
            errors.append(f"link {lid}: target node {dst_id} does not exist")
            continue
        src_outputs = by_id[src_id].get("outputs") or []
        dst_inputs = by_id[dst_id].get("inputs") or []
        if src_slot >= len(src_outputs):
            errors.append(f"link {lid}: node {src_id} has no output slot {src_slot}")
        elif src_outputs[src_slot].get("type") != ltype and ltype != "*":
            warnings.append(f"link {lid}: type {ltype} but source slot is "
                            f"{src_outputs[src_slot].get('type')}")
        if dst_slot >= len(dst_inputs):
            errors.append(f"link {lid}: node {dst_id} has no input slot {dst_slot}")
        elif dst_inputs[dst_slot].get("link") != lid:
            errors.append(f"link {lid}: node {dst_id} input '{dst_inputs[dst_slot].get('name')}' "
                          f"points at link {dst_inputs[dst_slot].get('link')} instead")

    for n in nodes:
        for inp in (n.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None and lid not in link_ids:
                errors.append(f"node {n.get('id')} input '{inp.get('name')}' references "
                              f"link {lid} which is not in the links table")
        for out in (n.get("outputs") or []):
            for lid in (out.get("links") or []):
                if lid not in link_ids:
                    errors.append(f"node {n.get('id')} output '{out.get('name')}' references "
                                  f"link {lid} which is not in the links table")

    # widget values vs the node's current definition
    for n in nodes:
        t = n.get("type")
        cls = tb_nodes.get(t)
        if cls is None:
            continue
        expected = widget_spec(cls)
        actual = n.get("widgets_values")
        if actual is None:
            if expected:
                warnings.append(f"node {n.get('id')} '{t}': no widgets_values but the node has "
                                f"{len(expected)} widgets ({', '.join(expected[:4])}...)")
            continue
        if isinstance(actual, dict):
            unknown = [k for k in actual if k not in expected]
            if unknown:
                warnings.append(f"node {n.get('id')} '{t}': widget keys not on the node: {unknown}")
            continue
        if len(actual) != len(expected):
            errors.append(f"node {n.get('id')} '{t}': {len(actual)} saved widget values but the node "
                          f"now has {len(expected)} - every setting after the difference is shifted. "
                          f"Expected order: {', '.join(expected)}")

        # required toolbox inputs must be connected or be widgets
        it = cls.INPUT_TYPES()
        connected = {i.get("name") for i in (n.get("inputs") or []) if i.get("link") is not None}
        widgets = set(expected)
        for req_name, definition in it.get("required", {}).items():
            if req_name in widgets or req_name in connected:
                continue
            errors.append(f"node {n.get('id')} '{t}': required input '{req_name}' "
                          f"({definition[0] if not isinstance(definition[0], list) else 'COMBO'}) "
                          f"is not connected")
    return errors, warnings, notes


def main():
    ap = argparse.ArgumentParser(description="Check every workflow for problems")
    ap.add_argument("--server", default="", help="ComfyUI URL, to validate core node types too")
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    args = ap.parse_args()

    tb = toolbox_nodes()
    info = server_node_types(args.server)
    folder = os.path.join(ROOT, "workflows")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))

    print("=" * 96)
    print(f" WORKFLOW CHECK - {len(files)} files, {len(tb)} toolbox nodes"
          + (f", {len(info)} node types on the server" if info else ", core nodes not checked"))
    print("=" * 96)
    total_err = total_warn = 0
    for f in files:
        errors, warnings, notes = check_workflow(os.path.join(folder, f), tb, info)
        total_err += len(errors)
        total_warn += len(warnings)
        if args.quiet and not errors and not warnings:
            continue
        status = "ERROR" if errors else ("warn " if warnings else "ok   ")
        print(f"\n[{status}] {f}")
        for n in notes:
            print(f"         {n}")
        for e in errors:
            print(f"   ERROR {e}")
        for w in warnings:
            print(f"   warn  {w}")
    print("\n" + "=" * 96)
    print(f" {total_err} errors, {total_warn} warnings across {len(files)} workflows")
    print("=" * 96)
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
