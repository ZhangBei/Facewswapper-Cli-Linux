"""Tests for the frame dtype/layout normalization policy.

The subject is stdlib-only, so these are real behaviour tests: no GPU, no
torch, no OpenCV, no ONNX Runtime, and no numpy either.

    python -m faceswapper.tests.test_frame_upload
"""

from __future__ import annotations

import sys

from faceswapper.frame_upload import (
    FramePlan,
    estimate_saved_bytes,
    format_saved_bytes,
    frame_bytes,
    normalize_plan,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_frame_bytes() -> None:
    """1080p RGB uint8 是 6220800 字节，文档与估算都基于这个数。"""
    if frame_bytes(1080, 1920) != 1920 * 1080 * 3:
        fail("1080p frame byte count is wrong")
    if frame_bytes(4, 5, channels=1) != 20:
        fail("single-channel frame byte count is wrong")
    for args in ((0, 10), (10, 0), (-1, 10), (10, 10, 0)):
        try:
            frame_bytes(*args)
        except ValueError:
            continue
        fail(f"frame_bytes{args!r} should raise ValueError")
    print("  ok  frame byte budget is computed correctly")


def test_gate_closed_always_keeps_the_old_path() -> None:
    """默认关闭时，任何 dtype/布局都必须照旧复制，一字节都不省。"""
    for dtype, contiguous in (
        ("uint8", True), ("uint8", False),
        ("float32", True), ("float32", False), ("int16", False),
    ):
        plan = normalize_plan(dtype, contiguous, 1080, 1920, zero_copy=False)
        if plan != FramePlan(True, 0, "gate-closed"):
            fail(f"gate-closed plan for {(dtype, contiguous)!r} is {plan!r}")
    print("  ok  a closed gate always keeps the original astype path")


def test_reusable_frame_skips_the_copy() -> None:
    """uint8 + C contiguous 才能零拷贝复用，省下整帧字节数。"""
    plan = normalize_plan("uint8", True, 1080, 1920, zero_copy=True)
    if plan.needs_copy or plan.bytes_saved != frame_bytes(1080, 1920):
        fail(f"reusable frame plan is {plan!r}")
    if plan.reason != "already-uint8-contiguous":
        fail(f"unexpected reason {plan.reason!r}")
    print("  ok  uint8 contiguous frames are reused without a memcpy")


def test_frames_that_need_conversion_still_copy() -> None:
    """dtype 不符或布局不连续时必须复制，不能为了零拷贝改像素语义。"""
    cases = [
        ("float32", True, "dtype-float32-needs-conversion"),
        ("int16", True, "dtype-int16-needs-conversion"),
        ("uint16", True, "dtype-uint16-needs-conversion"),
        ("uint8", False, "not-c-contiguous"),
    ]
    for dtype, contiguous, reason in cases:
        plan = normalize_plan(dtype, contiguous, 720, 1280, zero_copy=True)
        if not plan.needs_copy or plan.bytes_saved != 0:
            fail(f"{(dtype, contiguous)!r} must still copy: {plan!r}")
        if plan.reason != reason:
            fail(f"{(dtype, contiguous)!r} reason is {plan.reason!r}")
    print("  ok  dtype conversion and non-contiguous frames still copy")


def test_estimate_saved_bytes() -> None:
    """累计一串 plan 省下的字节数，可接生成器；不能改动入参。"""
    plans = [
        normalize_plan("uint8", True, 1080, 1920, zero_copy=True),
        normalize_plan("float32", True, 1080, 1920, zero_copy=True),
        normalize_plan("uint8", True, 1080, 1920, zero_copy=False),
        normalize_plan("uint8", True, 1080, 1920, zero_copy=True),
    ]
    expected = frame_bytes(1080, 1920) * 2
    if estimate_saved_bytes(plans) != expected:
        fail("saved-byte total does not count only reusable frames")
    if estimate_saved_bytes(iter(plans)) != expected:
        fail("estimate_saved_bytes should accept an iterator")
    if plans[0].bytes_saved != frame_bytes(1080, 1920):
        fail("estimate_saved_bytes mutated its input")

    # 文档引用的量级：1080p、H2D + D2H 两侧、30fps、60 秒。
    one_frame = frame_bytes(1080, 1920) * 2
    minute = estimate_saved_bytes(
        normalize_plan("uint8", True, 1080, 1920, zero_copy=True)
        for _ in range(30 * 60 * 2)
    )
    # 生成器里每个 plan 代表「一侧的一帧」，共 60s × 30fps × 2 侧。
    if minute != one_frame * 30 * 60:
        fail("60s budget arithmetic is wrong")
    if minute < 20_000_000_000:
        fail(f"60s budget looks too small: {minute}")
    print("  ok  saved-byte accounting matches the documented 1080p budget")


def test_format_saved_bytes() -> None:
    cases = {
        0: "0 B",
        1023: "1023 B",
        1024: "1.0 KiB",
        6220800: "5.9 MiB",
    }
    for raw, expected in cases.items():
        got = format_saved_bytes(raw)
        if got != expected:
            fail(f"format_saved_bytes({raw}) = {got!r}, expected {expected!r}")
    print("  ok  byte formatting is readable and stable")


def main() -> int:
    print("faceswapper frame upload policy tests")
    test_frame_bytes()
    test_gate_closed_always_keeps_the_old_path()
    test_reusable_frame_skips_the_copy()
    test_frames_that_need_conversion_still_copy()
    test_estimate_saved_bytes()
    test_format_saved_bytes()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
