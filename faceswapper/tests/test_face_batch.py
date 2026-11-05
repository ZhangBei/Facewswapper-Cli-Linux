"""Tests for the single-frame multi-face batch eligibility policy.

stdlib-only, real behaviour tests: no GPU, no torch, no OpenCV, no PyAV.
The batch path itself is gated off; these tests cover *when* it would be safe
to turn on.

    python -m faceswapper.tests.test_face_batch
"""

from __future__ import annotations

import sys

from faceswapper.face_batch import FaceCandidate, batch_eligibility


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def eligible(**overrides):
    kwargs = {
        "candidates": [
            FaceCandidate((0, 0, 100, 100), 128),
            FaceCandidate((200, 0, 300, 100), 128),
        ],
        "swapper_type": "inswapper-128",
        "parameters": {},
    }
    kwargs.update(overrides)
    return batch_eligibility(**kwargs)


def test_two_separate_faces_are_eligible() -> None:
    ok, reason = eligible()
    if not ok:
        fail(f"two non-overlapping 128 faces should be eligible: {reason}")
    print("  ok  two separate faces with the 128 model are eligible")


def test_single_face_falls_back() -> None:
    ok, reason = eligible(candidates=[FaceCandidate((0, 0, 100, 100), 128)])
    if ok or "at least" not in reason:
        fail(f"a single face must fall back, got {(ok, reason)!r}")
    ok, reason = eligible(candidates=[])
    if ok:
        fail("an empty frame must fall back")
    print("  ok  zero or one face falls back to the per-face path")


def test_non_128_models_fall_back() -> None:
    for model in ("inswapper-256", "inswapper-512", "hyperswap-256"):
        ok, reason = eligible(swapper_type=model)
        if ok or "128" not in reason:
            fail(f"{model} must fall back, got {(ok, reason)!r}")
    print("  ok  non-128 swapper models fall back")


def test_multi_pass_switches_fall_back() -> None:
    for key in ("StrengthSwitch", "HighFidelitySwitch"):
        ok, reason = eligible(parameters={key: True})
        if ok or key not in reason:
            fail(f"{key} must fall back, got {(ok, reason)!r}")
    # 关着的开关不该挡路
    ok, _ = eligible(parameters={"StrengthSwitch": False,
                                 "HighFidelitySwitch": False})
    if not ok:
        fail("switches that are off must not block batching")
    print("  ok  multi-pass switches fall back, off switches do not")


def test_pipeline_size_must_be_uniform_128() -> None:
    cases = [
        [FaceCandidate((0, 0, 100, 100), 128),
         FaceCandidate((200, 0, 300, 100), 256)],
        [FaceCandidate((0, 0, 100, 100), 256),
         FaceCandidate((200, 0, 300, 100), 256)],
    ]
    for candidates in cases:
        ok, reason = eligible(candidates=candidates)
        if ok or "128" not in reason:
            fail(f"non-uniform/non-128 sizes must fall back: {(ok, reason)!r}")
    print("  ok  every face must have pipeline_size 128")


def test_overlapping_bboxes_fall_back() -> None:
    overlaps = [
        [FaceCandidate((0, 0, 100, 100), 128),
         FaceCandidate((50, 50, 150, 150), 128)],
        [FaceCandidate((0, 0, 100, 100), 128),
         FaceCandidate((100, 0, 200, 100), 128),   # 边界相接，不算重叠
         FaceCandidate((150, 0, 250, 100), 128)],
    ]
    ok, reason = eligible(candidates=overlaps[0])
    if ok or "overlap" not in reason:
        fail(f"overlapping bboxes must fall back: {(ok, reason)!r}")

    # 只相接（r1 == l2）应该允许
    touching = [
        FaceCandidate((0, 0, 100, 100), 128),
        FaceCandidate((100, 0, 200, 100), 128),
    ]
    ok, reason = eligible(candidates=touching)
    if not ok:
        fail(f"bboxes that only touch should be eligible: {reason}")

    ok, reason = eligible(candidates=overlaps[1])
    if ok or "overlap" not in reason:
        fail(f"a later overlapping pair must fall back: {(ok, reason)!r}")
    print("  ok  overlapping bboxes fall back, touching edges do not")


def test_batch_path_is_disabled() -> None:
    """闸门必须关闭、主路径不得挂载 batch、swap_core 的默认参数不变。

    用户明确要求「代码编写进入，但是不启用」——这里用 AST 把这条约束钉住，
    防止后续有人误把实验路径接进 _swap_video_inner。
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "VideoManager.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)

    gate_off = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "EXPERIMENTAL_FACE_BATCH"
                for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in tree.body
    )
    if not gate_off:
        fail("EXPERIMENTAL_FACE_BATCH must stay False until GPU PSNR validation")

    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "VideoManager"),
        None,
    )
    if cls is None:
        fail("VideoManager class not found")
    methods = {
        n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)
    }
    for required in ("swap_core", "swap_core_batched", "_swap_video_inner"):
        if required not in methods:
            fail(f"VideoManager is missing {required}()")

    inner_calls = {
        node.func.attr for node in ast.walk(methods["_swap_video_inner"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if "swap_core_batched" in inner_calls:
        fail("the experimental batch path must not be wired into _swap_video_inner")

    core = methods["swap_core"]
    if not (
        core.args.args[-1].arg == "batch_slot"
        and isinstance(core.args.defaults[-1], ast.Constant)
        and core.args.defaults[-1].value is None
    ):
        fail("swap_core's batch_slot must default to None")
    print("  ok  batch code is present, gated off and not wired in")


def main() -> int:
    print("faceswapper face-batch eligibility tests")
    test_two_separate_faces_are_eligible()
    test_single_face_falls_back()
    test_non_128_models_fall_back()
    test_multi_pass_switches_fall_back()
    test_pipeline_size_must_be_uniform_128()
    test_overlapping_bboxes_fall_back()
    test_batch_path_is_disabled()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
