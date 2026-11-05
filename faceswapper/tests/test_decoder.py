"""Tests for GPU-decode probing and frame field normalization.

stdlib-only, real behaviour tests: no GPU, no torch, no torchcodec, no PyAV.

    python -m faceswapper.tests.test_decoder
"""

from __future__ import annotations

import sys

from faceswapper.decoder import frame_fields, resolve_decoder
from faceswapper.errors import HeadlessError


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


class ExplodingProbe:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise RuntimeError("should not be called")


def test_pyav_is_the_untouched_default() -> None:
    probe = ExplodingProbe()
    plan = resolve_decoder("pyav", probe)
    if probe.calls:
        fail(f"the default path must not probe GPU decode: {probe.calls}")
    if plan.backend != "pyav":
        fail(f"wrong backend: {plan.backend!r}")
    print("  ok  --decoder pyav never probes torchcodec or CUDA")


def test_gpu_available_is_selected() -> None:
    plan = resolve_decoder("gpu", lambda: (True, "ok"))
    if plan.backend != "gpu" or not plan.note:
        fail(f"GPU plan should carry the API caveat: {plan!r}")
    print("  ok  available GPU decode is selected with a caveat")


def test_gpu_unavailable_fails_loudly() -> None:
    cases = [
        (lambda: (False, "torchcodec is not installed"), "torchcodec"),
        (lambda: (False, "CUDA is not available"), "CUDA"),
        (lambda: (_ for _ in ()).throw(LookupError("bad import")), "probe failed"),
    ]
    for probe, expected in cases:
        try:
            resolve_decoder("gpu", probe)
        except HeadlessError as exc:
            if expected.lower() not in str(exc).lower():
                fail(f"unhelpful error for {expected!r}: {exc!r}")
            if "pyav" not in str(exc):
                fail(f"error should point at the pyav fallback: {exc!r}")
        else:
            fail("unavailable GPU decode must raise, not silently fall back")
    print("  ok  unavailable GPU decode raises with a reason and a fallback hint")


def test_unknown_decoder_is_rejected() -> None:
    try:
        resolve_decoder("nvdec", lambda: (True, "ok"))
    except HeadlessError as exc:
        if "pyav" not in str(exc):
            fail(f"unhelpful error: {exc!r}")
    else:
        fail("an unknown --decoder value must be rejected")
    print("  ok  unknown decoder values are rejected")


class FakeFrame:
    def __init__(self, data=None, pts=None, pts_seconds=None):
        if data is not None:
            self.data = data
        if pts is not None:
            self.pts = pts
        if pts_seconds is not None:
            self.pts_seconds = pts_seconds


class AmbiguousBool:
    """模拟多元素 tensor：对它求 bool 会抛错。"""

    def __bool__(self):
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")


def test_frame_fields_handles_both_shapes() -> None:
    tensor = AmbiguousBool()
    cases = [
        ({"data": tensor, "pts": 1234}, (tensor, 1234)),
        ({"frame": tensor, "pts_seconds": 0.5}, (tensor, 0.5)),
        (FakeFrame(data=tensor, pts=4321), (tensor, 4321)),
        (FakeFrame(data=tensor, pts_seconds=1.25), (tensor, 1.25)),
        (FakeFrame(data=tensor), (tensor, None)),
    ]
    for raw, expected in cases:
        got = frame_fields(raw)
        if got[0] is not expected[0] or got[1] != expected[1]:
            fail(f"frame_fields({raw!r}) = {got!r}, expected {expected!r}")
    print("  ok  frame_fields handles dict/attr layouts without bool(tensor)")


def test_frame_fields_rejects_unknown_layout() -> None:
    for raw in ({}, {"pts": 10}, FakeFrame(pts=10), object()):
        try:
            frame_fields(raw)
        except HeadlessError as exc:
            if "torchcodec" not in str(exc):
                fail(f"error should mention the adapter assumption: {exc!r}")
        else:
            fail(f"missing data must raise: {raw!r}")
    print("  ok  unknown frame layouts raise instead of yielding a bad frame")


def main() -> int:
    print("faceswapper GPU decoder tests")
    test_pyav_is_the_untouched_default()
    test_gpu_available_is_selected()
    test_gpu_unavailable_fails_loudly()
    test_unknown_decoder_is_rejected()
    test_frame_fields_handles_both_shapes()
    test_frame_fields_rejects_unknown_layout()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
