"""Static wiring checks for paths that cannot be imported without the GPU stack.

``headless.py`` / ``VideoManager.py`` need torch, OpenCV and ONNX Runtime, so a
dataclass keyword typo or an ``args.<flag>`` mismatch would only surface at
runtime on the GPU box. These AST checks catch that class of mistake offline.

    python -m faceswapper.tests.test_wiring
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def parse(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


def class_fields(tree: ast.Module, name: str) -> set[str]:
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )
    if cls is None:
        fail(f"{name} class not found")
    return {
        n.target.id for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def test_dataclass_keywords_match_fields() -> None:
    """SwapOptions / VideoOptions 的构造 kwargs 必须都是已声明字段。"""
    headless = parse("headless.py")
    swap = class_fields(headless, "SwapOptions")
    video = swap | class_fields(headless, "VideoOptions")
    expected = {"SwapOptions": swap, "VideoOptions": video}
    found = 0
    for tree in (headless, parse("cli.py")):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else ""
            )
            if name not in expected:
                continue
            found += 1
            bad = {k.arg for k in node.keywords} - expected[name]
            if bad:
                fail(f"{name}(...) has undeclared kwargs: {sorted(bad)}")
    if found < 2:
        fail(f"expected both SwapOptions and VideoOptions call sites, got {found}")
    print("  ok  dataclass construction kwargs all match declared fields")


def test_options_attribute_reads_are_declared() -> None:
    """self.options.X 直接读取必须对应已声明字段。

    带默认值的 ``getattr(self.options, "X", default)`` 不在此列：字段缺失时
    它也有兜底，不会抛 AttributeError。
    """
    headless = parse("headless.py")
    fields = (
        class_fields(headless, "SwapOptions")
        | class_fields(headless, "VideoOptions")
    )
    undeclared = set()
    for node in ast.walk(headless):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "options"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and isinstance(node.ctx, ast.Load)
            and node.attr not in fields
        ):
            undeclared.add(node.attr)
    if undeclared:
        fail(f"self.options reads with no declared field: {sorted(undeclared)}")
    print("  ok  every direct self.options read maps to a dataclass field")


def test_cmd_video_args_reads_have_flags() -> None:
    """_cmd_video 里读取的 args.X 必须来自某个 add_argument（或 subparser）。"""
    cli = parse("cli.py")
    cmd = next(
        n for n in cli.body
        if isinstance(n, ast.FunctionDef) and n.name == "_cmd_video"
    )
    reads = {
        n.attr for n in ast.walk(cmd)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "args"
        and isinstance(n.ctx, ast.Load)
    }
    dests = {"command", "func"}
    for node in ast.walk(cli):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
        ):
            dests.add(node.args[0].value[2:].replace("-", "_"))
    missing = reads - dests
    if missing:
        fail(f"args attributes with no CLI flag: {sorted(missing)}")
    print("  ok  every _cmd_video args read has a matching CLI flag")


def test_encoder_worker_kwargs_match_signature() -> None:
    """编码线程 kwargs 的键必须都在 encoder_worker 的签名里。"""
    encoder = parse("encoder.py")
    worker = next(
        n for n in encoder.body
        if isinstance(n, ast.FunctionDef) and n.name == "encoder_worker"
    )
    sig = {a.arg for a in worker.args.args} | {
        a.arg for a in worker.args.kwonlyargs
    }
    checked = 0
    for tree in (encoder, parse("headless.py")):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if not {"silent_path", "stop_event"} <= keys:
                continue
            checked += 1
            bad = keys - sig
            if bad:
                fail(f"encoder_worker kwargs not in signature: {sorted(bad)}")
    if checked == 0:
        fail("no encoder_worker kwargs dict found")
    print("  ok  encoder thread kwargs match the worker signature")


def main() -> int:
    print("faceswapper static wiring tests")
    test_dataclass_keywords_match_fields()
    test_options_attribute_reads_are_declared()
    test_cmd_video_args_reads_have_flags()
    test_encoder_worker_kwargs_match_signature()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
