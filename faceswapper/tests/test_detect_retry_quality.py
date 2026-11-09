"""--detect-retry-quality 的离线测试（无 GPU / torch / numpy / OpenCV）。

侧脸方案 P1：把 --detect-retry 从「0 检出」放宽到「检出了但不可靠」的
一正一侧混合帧。这里覆盖三件可离线验证的事：

1. kps 质量代理的旋转不变性（不把 roll 歪头当成坏 kps）与 yaw 不惩罚。
2. retry_plan 的默认行为逐字不变，以及 quality 模式的门控真值表。
3. 接线：重检结果只追加不替换、默认零行为变化、CLI 阈值校验。

    python -m faceswapper.tests.test_detect_retry_quality
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from faceswapper.detect_retry import (
    DEFAULT_RETRY_MIN_CONF,
    DEFAULT_RETRY_MIN_KPS_QUALITY,
    DEFAULT_RETRY_QUALITY_SPEC,
    RETRY_INPUT_SIZE,
    kps_quality,
    parse_retry_quality,
    retry_plan,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTAL = [(100.0, 100.0), (140.0, 100.0), (120.0, 125.0),
           (105.0, 150.0), (135.0, 150.0)]


def rolled(face, degrees, cx=120.0, cy=125.0):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return [(cx + c * (x - cx) - s * (y - cy),
             cy + s * (x - cx) + c * (y - cy)) for x, y in face]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_parse_retry_quality() -> None:
    if parse_retry_quality(None) is not None:
        fail("None must mean disabled")
    if parse_retry_quality("  ") is not None:
        fail("blank spec must mean disabled")
    if parse_retry_quality("0.7:0.4") != (0.7, 0.4):
        fail("spec must parse into (min_conf, min_kps_quality)")
    if parse_retry_quality(DEFAULT_RETRY_QUALITY_SPEC) != (
            DEFAULT_RETRY_MIN_CONF, DEFAULT_RETRY_MIN_KPS_QUALITY):
        fail("default spec must match the two constants")
    for bad in ("0.7", "0.7:0.4:0.2", "abc:0.4", "1.5:0.4",
                "-0.1:0.4", ":0.4", "nan:0.4"):
        try:
            parse_retry_quality(bad)
        except ValueError:
            continue
        fail(f"invalid spec {bad!r} was accepted")
    print("  ok  parse_retry_quality: None / valid / invalid all pinned")


def test_kps_quality_rotation_invariant() -> None:
    base = kps_quality(FRONTAL)
    if abs(base - 1.0) > 1e-9:
        fail(f"a well-formed frontal face must score 1.0, got {base}")
    for degrees in (0, 15, 30, 45, 90, 135, 180, 270):
        got = kps_quality(rolled(FRONTAL, degrees))
        if abs(got - base) > 1e-9:
            fail(f"kps_quality must be roll-invariant, {degrees} deg -> {got}")
    print("  ok  kps_quality: roll-invariant across 0-270 deg")


def test_kps_quality_yaw_and_degenerate() -> None:
    # 鼻尖沿眼轴偏移是 yaw 信号，不该被当成几何错误扣分。
    yawed = [FRONTAL[0], FRONTAL[1], (132.0, 125.0),
             FRONTAL[3], FRONTAL[4]]
    if abs(kps_quality(yawed) - 1.0) > 1e-9:
        fail("yaw nose offset must not lower the geometry quality")
    for bad, label in (
        ([(10.0, 10.0)] * 5, "degenerate"),
        ([(0.0, 0.0), (10.0, 90.0), (300.0, 5.0),
          (2.0, 300.0), (400.0, 400.0)], "scrambled"),
        ([(1.0, 2.0), (3.0, 4.0)], "too short"),
        ([], "empty"),
    ):
        if kps_quality(bad) != 0.0:
            fail(f"{label} kps must score 0.0")
    print("  ok  kps_quality: yaw not penalised, degenerate inputs safe")


def test_retry_plan_backwards_compatible() -> None:
    # 默认（quality_enabled=False）逐字保持旧行为。
    if retry_plan(detected=0, input_size=640, detect_score=0.5) is not None:
        fail("disabled retry must return None")
    if retry_plan(detected=0, input_size=640, detect_score=0.5,
                  enabled=True) != (RETRY_INPUT_SIZE, 0.7):
        fail("empty frame must retry at 960 / +0.2")
    if retry_plan(detected=2, input_size=640, detect_score=0.5,
                  enabled=True) is not None:
        fail("detected>0 must never retry by default")
    if retry_plan(detected=0, input_size=640, detect_score=0.5,
                  auto_orient=True, enabled=True) is not None:
        fail("auto-orient gate must still win")
    print("  ok  retry_plan: legacy behaviour pinned when quality is off")


def test_retry_plan_quality_gate() -> None:
    base = dict(detected=2, input_size=640, detect_score=0.5,
                enabled=True, quality_enabled=True)
    # 高置信度 + 好 kps：不重试。
    if retry_plan(**base, max_conf=0.95, frame_kps_quality=0.9) is not None:
        fail("reliable frame must not be retried")
    # 两级门控各自独立成立。
    for kw in (dict(max_conf=0.55, frame_kps_quality=0.9),
               dict(max_conf=0.95, frame_kps_quality=0.2)):
        plan = retry_plan(**base, **kw)
        if plan != (RETRY_INPUT_SIZE, 0.7):
            fail(f"unreliable frame must retry, got {plan!r} for {kw!r}")
    # 缺信号时不能凭空触发。
    if retry_plan(**base) is not None:
        fail("missing signals must not trigger a retry")
    # 两道防误检闸仍然生效。
    if retry_plan(**base, max_conf=0.1, frame_kps_quality=0.0,
                  auto_orient=True) is not None:
        fail("auto-orient gate must apply in quality mode too")
    at_max = dict(base, input_size=960)
    if retry_plan(**at_max, max_conf=0.1, frame_kps_quality=0.0) is not None:
        fail("already-max input size must not retry")
    # 自定义阈值。
    if retry_plan(**base, max_conf=0.65, frame_kps_quality=0.9,
                  min_conf=0.7) is None:
        fail("custom min_conf must be honoured")
    print("  ok  retry_plan: conf / kps gates, legacy safeguards, custom thresholds")


def test_cli_validation() -> None:
    import argparse

    from faceswapper.cli import _retry_quality_spec, build_parser

    if _retry_quality_spec("0.7:0.4") != "0.7:0.4":
        fail("_retry_quality_spec must return the original string")
    try:
        _retry_quality_spec("0.7")
    except argparse.ArgumentTypeError:
        pass
    else:
        fail("invalid spec must be rejected at argparse time")
    parser = build_parser()
    argv = ["video", "--source", "a.jpg", "--target", "a.mp4",
            "--output", "o.mp4", "--detect-retry"]
    plain = parser.parse_args(argv)
    if plain.detect_retry_quality is not None:
        fail("flag must default to None (disabled)")
    with_const = parser.parse_args(argv + ["--detect-retry-quality"])
    if with_const.detect_retry_quality != DEFAULT_RETRY_QUALITY_SPEC:
        fail(f"bare flag must use the default spec, got "
             f"{with_const.detect_retry_quality!r}")
    custom = parser.parse_args(argv + ["--detect-retry-quality", "0.7:0.4"])
    if custom.detect_retry_quality != "0.7:0.4":
        fail("explicit spec must round-trip")
    print("  ok  CLI: default off, bare flag uses defaults, custom spec parses")


def test_wiring_and_default_off() -> None:
    """接线 + 默认零行为变化（源码断言，本机无法 import GPU 栈）。"""
    cli = (ROOT / "faceswapper" / "cli.py").read_text(encoding="utf-8")
    headless = (ROOT / "faceswapper" / "headless.py").read_text(encoding="utf-8")
    vm = (ROOT / "faceswapper" / "VideoManager.py").read_text(encoding="utf-8")
    retry = (ROOT / "faceswapper" / "detect_retry.py").read_text(
        encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if '"--detect-retry-quality"' not in cli:
        fail("cli.py missing --detect-retry-quality")
    if cli.count("detect_retry_quality=args.detect_retry_quality") != 2:
        fail("flag must reach both SwapOptions and VideoOptions")
    if "--detect-retry-quality" not in readme:
        fail("README must document the flag (test_docs_stay_in_sync)")
    if "detect_retry_quality: str | None = None" not in headless:
        fail("SwapOptions missing the field")
    if "self.vm.detect_retry_quality = parse_retry_quality(" not in headless:
        fail("headless must parse and inject the spec")
    if "self.detect_retry_quality: tuple[float, float] | None = None" not in vm:
        fail("VideoManager must default the spec to None")
    if "with_scores=need_scores" not in vm:
        fail("first-pass detect must opt into scores only when enabled")
    if "quality_ok = quality_limits is not None and applied_angle == 0.0" not in vm:
        fail("quality retry must be excluded on rotated frames")
    # 重检结果只追加、不替换：必须走合并 + concatenate。
    block = vm.split("if len(kpss) == 0 or quality_ok:", 1)[-1].split(
        "自动朝向反馈", 1)[0]
    if "merge_probe_landmarks(" not in block or "np.concatenate(" not in block:
        fail("quality retry must merge new faces instead of overwriting them")
    if "kpss = retry_kpss" not in block:
        fail("empty-frame retry must still use the retry result directly")
    # stdlib 侧确实实现了质量代理与解析。
    for needle in ("def kps_quality(", "def parse_retry_quality(",
                   "DEFAULT_RETRY_MIN_CONF", "quality_enabled: bool = False"):
        if needle not in retry:
            fail(f"detect_retry.py missing {needle!r}")
    print("  ok  wiring: flag threaded, default off, merge-not-overwrite")


def main() -> int:
    print("faceswapper detect-retry-quality tests")
    test_parse_retry_quality()
    test_kps_quality_rotation_invariant()
    test_kps_quality_yaw_and_degenerate()
    test_retry_plan_backwards_compatible()
    test_retry_plan_quality_gate()
    test_cli_validation()
    test_wiring_and_default_off()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
