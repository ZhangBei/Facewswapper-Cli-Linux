"""Tests for cross-frame batch eligibility, gating and grouping.

stdlib-only, real behaviour tests. The GPU-side forward is gated off; these
tests cover *when* it would be safe and how frames must stay ordered.

    python -m faceswapper.tests.test_cross_frame_batch
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from faceswapper.cross_frame_batch import (
    EXPERIMENTAL_CROSS_FRAME_BATCH,
    FrameCandidate,
    cross_frame_batch_eligibility,
    group_frames,
    resolve_cross_frame_batch,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def frames(count=3):
    return [
        FrameCandidate(index=i, pts=i * 1000, face_count=1, pipeline_size=128)
        for i in range(count)
    ]


def eligible(**overrides):
    kwargs = {
        "candidates": frames(),
        "workers": 4,
        "every_nth": 1,
        "swapper_type": "inswapper-128",
        "parameters": {},
        "batch_backend_available": True,
    }
    kwargs.update(overrides)
    return cross_frame_batch_eligibility(**kwargs)


def test_valid_batch_is_eligible() -> None:
    ok, reason = eligible()
    if not ok:
        fail(f"a contiguous single-face batch should be eligible: {reason}")
    print("  ok  contiguous single-face frames are eligible")


def test_gate_is_closed() -> None:
    if EXPERIMENTAL_CROSS_FRAME_BATCH:
        fail("the experimental gate must stay False until GPU PSNR validation")
    ok, reason = resolve_cross_frame_batch(
        frames(),
        workers=4,
        every_nth=1,
        swapper_type="inswapper-128",
        parameters={},
        batch_backend_available=True,
    )
    if ok or "gate" not in reason:
        fail(f"the gate must reject every batch: {(ok, reason)!r}")
    print("  ok  the gate rejects batching while keeping eligibility testable")


def test_hard_conditions_fall_back() -> None:
    cases = [
        ("one frame", dict(candidates=frames(1))),
        ("serial workers", dict(workers=1)),
        ("every-nth", dict(every_nth=20)),
        ("non-128 model", dict(swapper_type="hyperswap-256")),
        ("StrengthSwitch", dict(parameters={"StrengthSwitch": True})),
        ("HighFidelitySwitch", dict(parameters={"HighFidelitySwitch": True})),
        ("static batch=1", dict(batch_backend_available=False)),
    ]
    for label, overrides in cases:
        ok, reason = eligible(**overrides)
        if ok:
            fail(f"{label} must fall back, got eligible ({reason})")
    print("  ok  workers / every-nth / model / switches / backend all gate it")


def test_variable_face_counts_and_sizes_fall_back() -> None:
    varying = [
        FrameCandidate(0, 0, 1, 128),
        FrameCandidate(1, 1000, 2, 128),
    ]
    ok, reason = eligible(candidates=varying)
    if ok or "exactly one face" not in reason:
        fail(f"variable face counts must fall back: {(ok, reason)!r}")
    differing = [
        FrameCandidate(0, 0, 1, 128),
        FrameCandidate(1, 1000, 1, 256),
    ]
    ok, reason = eligible(candidates=differing)
    if ok or "128" not in reason:
        fail(f"differing pipeline_size must fall back: {(ok, reason)!r}")
    print("  ok  variable face counts and pipeline sizes fall back")


def test_pts_and_index_ordering_is_enforced() -> None:
    gaps = [
        FrameCandidate(0, 0, 1, 128),
        FrameCandidate(2, 2000, 1, 128),
    ]
    ok, reason = eligible(candidates=gaps)
    if ok or "contiguous" not in reason:
        fail(f"index gaps must fall back: {(ok, reason)!r}")
    backwards = [
        FrameCandidate(0, 0, 1, 128),
        FrameCandidate(1, 500, 1, 128),
        FrameCandidate(2, 500, 1, 128),
    ]
    ok, reason = eligible(candidates=backwards)
    if ok or "PTS" not in reason:
        fail(f"non-increasing PTS must fall back: {(ok, reason)!r}")
    print("  ok  contiguous indices and strictly increasing PTS are enforced")


def test_grouping_preserves_decode_order() -> None:
    items = frames(7)
    groups = group_frames(items, 3)
    if [len(g) for g in groups] != [3, 3, 1]:
        fail(f"unexpected group sizes: {[len(g) for g in groups]}")
    flat = [c for group in groups for c in group]
    if flat != items:
        fail("grouping reordered frames")
    try:
        group_frames(items, 1)
    except ValueError:
        pass
    else:
        fail("batch_size < 2 must be rejected")
    print("  ok  grouping slices in order and keeps a short tail")


def test_not_wired_into_the_pipeline() -> None:
    """闸门之外，主流水线也不得调用跨帧 batch 入口。"""
    root = Path(__file__).resolve().parents[1]
    for name in ("headless.py", "frame_order.py", "VideoManager.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "resolve_cross_frame_batch" in calls:
            fail(f"{name} must not wire in the experimental cross-frame batch")
    print("  ok  no production frame path calls the gated entry point")


def main() -> int:
    print("faceswapper cross-frame batch tests")
    test_valid_batch_is_eligible()
    test_gate_is_closed()
    test_hard_conditions_fall_back()
    test_variable_face_counts_and_sizes_fall_back()
    test_pts_and_index_ordering_is_enforced()
    test_grouping_preserves_decode_order()
    test_not_wired_into_the_pipeline()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
