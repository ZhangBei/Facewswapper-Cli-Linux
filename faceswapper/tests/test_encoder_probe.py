"""Tests for encoder selection and NVENC probing.

stdlib-only, real behaviour tests: no GPU, no PyAV, no torch, no OpenCV.
``codec_factory`` is injected, so both the available and unavailable NVENC
cases are exercised without any hardware.

    python -m faceswapper.tests.test_encoder_probe
"""

from __future__ import annotations

import sys

from faceswapper.encoder_probe import (
    X264_TO_NVENC_PRESET,
    EncoderPlan,
    resolve_encoder,
)
from faceswapper.errors import HeadlessError


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


class ExplodingFactory:
    """被调用就炸——用来证明 libx264 分支不做任何 NVENC 探测。"""

    def __init__(self):
        self.calls = []

    def __call__(self, name, mode=None):
        self.calls.append((name, mode))
        raise RuntimeError("should not be called")


class FakeFactory:
    def __init__(self, available=True):
        self.available = available
        self.calls = []

    def __call__(self, name, mode=None):
        self.calls.append((name, mode))
        if not self.available:
            raise LookupError(f"unknown codec {name}")
        return object()


def test_libx264_is_the_untouched_default() -> None:
    factory = ExplodingFactory()
    plan = resolve_encoder("libx264", factory, crf=18, preset="medium")
    if factory.calls:
        fail(f"the default path must not probe NVENC: {factory.calls!r}")
    if plan.codec != "libx264":
        fail(f"wrong codec: {plan.codec!r}")
    if plan.stream_options != {"crf": "18", "preset": "medium"}:
        fail(f"libx264 options changed: {plan.stream_options!r}")
    print("  ok  libx264 keeps CRF/preset and never probes NVENC")


def test_nvenc_available_is_selected() -> None:
    factory = FakeFactory(available=True)
    plan = resolve_encoder("nvenc", factory, crf=23, preset="fast")
    if factory.calls != [("h264_nvenc", "w")]:
        fail(f"NVENC must be probed as an encoder: {factory.calls!r}")
    if plan.codec != "h264_nvenc":
        fail(f"wrong codec: {plan.codec!r}")
    opts = plan.stream_options
    if opts.get("rc") != "vbr" or opts.get("cq") != "23":
        fail(f"CRF must map to CQ with rc=vbr: {opts!r}")
    if opts.get("preset") != X264_TO_NVENC_PRESET["fast"]:
        fail(f"preset was not mapped to NVENC: {opts!r}")
    if "not equivalent" not in plan.note:
        fail("the plan must warn that CQ/preset are not equivalent to CRF")
    print("  ok  available NVENC maps CRF->CQ and x264->p1..p7")


def test_nvenc_unavailable_fails_loudly() -> None:
    factory = FakeFactory(available=False)
    try:
        resolve_encoder("nvenc", factory, crf=18, preset="medium")
    except HeadlessError as exc:
        if "h264_nvenc" not in str(exc) or "libx264" not in str(exc):
            fail(f"unhelpful error message: {exc!r}")
        if exc.__cause__ is None:
            fail("the original codec error should be chained")
    else:
        fail("unavailable NVENC must raise instead of silently falling back")
    print("  ok  unavailable NVENC raises HeadlessError with a fix hint")


def test_all_documented_presets_map() -> None:
    documented = ("ultrafast", "veryfast", "fast", "medium", "slow", "veryslow")
    for preset in documented:
        if preset not in X264_TO_NVENC_PRESET:
            fail(f"missing NVENC mapping for {preset!r}")
        plan = resolve_encoder(
            "nvenc", FakeFactory(), crf=20, preset=preset,
        )
        if plan.stream_options["preset"] != X264_TO_NVENC_PRESET[preset]:
            fail(f"preset {preset!r} mapped incorrectly: {plan!r}")
    # 未知档位退回 p4，不抛错
    plan = resolve_encoder("nvenc", FakeFactory(), crf=20, preset="custom")
    if plan.stream_options["preset"] != "p4":
        fail(f"unknown preset should fall back to p4: {plan!r}")
    print("  ok  every documented x264 preset has an NVENC counterpart")


def test_unknown_encoder_is_rejected() -> None:
    try:
        resolve_encoder("h264", FakeFactory(), crf=18, preset="medium")
    except HeadlessError as exc:
        if "libx264" not in str(exc):
            fail(f"unhelpful error message: {exc!r}")
    else:
        fail("an unknown --encoder value must be rejected")
    print("  ok  unknown encoder values are rejected")


def main() -> int:
    print("faceswapper encoder probe tests")
    test_libx264_is_the_untouched_default()
    test_nvenc_available_is_selected()
    test_nvenc_unavailable_fails_loudly()
    test_all_documented_presets_map()
    test_unknown_encoder_is_rejected()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
