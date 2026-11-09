"""--align-for-swap 的离线测试（无 GPU / torch / numpy / OpenCV）。

侧脸方案 P2：把识别阶段已算出的 68 点派生 5 点复用到 ``swap_core`` 的
替换几何（L3）。可离线验证的部分：

1. ``yaw_sideness`` 的 roll 旋转不变性（修 frontalness 的歪头污染）。
2. ``align_kps_for_swap`` 的三道门（开关 / 侧度 / 68 点质量）。
3. 接线：``ret`` 仍是 2 元素、68 点走旁路映射、默认零行为变化、
   两个前置条件在 headless 里强制校验。

    python -m faceswapper.tests.test_align_for_swap
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from faceswapper.detect_retry import (
    DEFAULT_ALIGN_FOR_SWAP_SIDENESS,
    align_kps_for_swap,
    kps_quality,
    local_frame,
    project,
    yaw_sideness,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTAL = [(100.0, 100.0), (140.0, 100.0), (120.0, 125.0),
           (105.0, 150.0), (135.0, 150.0)]


def yawed(nose_dx):
    return [FRONTAL[0], FRONTAL[1],
            (120.0 + nose_dx, 125.0), FRONTAL[3], FRONTAL[4]]


def shifted(face, dx, dy):
    return [(x + dx, y + dy) for x, y in face]


def rolled(face, degrees, cx=120.0, cy=125.0):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return [(cx + c * (x - cx) - s * (y - cy),
             cy + s * (x - cx) + c * (y - cy)) for x, y in face]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_local_frame() -> None:
    frame = local_frame(FRONTAL)
    if frame is None:
        fail("a well-formed face must have a local frame")
    pts, ux, uy, vx, vy, L, mid_x, mid_y = frame
    if (abs(ux - 1.0) > 1e-9 or abs(uy) > 1e-9
            or abs(vx) > 1e-9 or abs(vy - 1.0) > 1e-9
            or abs(L - 40.0) > 1e-9
            or abs(mid_x - 120.0) > 1e-9 or abs(mid_y - 100.0) > 1e-9):
        fail(f"unexpected local frame: {frame[1:]!r}")
    # 鼻尖在眼中点正下方 25px -> (u=0, v=0.625)。
    nose_u, nose_v = project(frame, pts[2])
    if abs(nose_u) > 1e-9 or abs(nose_v - 0.625) > 1e-9:
        fail(f"unexpected nose projection: {(nose_u, nose_v)!r}")
    for bad in ([], [(1.0, 2.0)], [(10.0, 10.0)] * 5,
                [(float("nan"), 1.0)] + FRONTAL[1:]):
        if local_frame(bad) is not None:
            fail(f"degenerate input must not produce a frame: {bad!r}")
    print("  ok  local_frame / project: axes, scale, degenerate inputs")


def test_yaw_sideness() -> None:
    if yaw_sideness(FRONTAL) != 0.0:
        fail("a frontal face must have zero sideness")
    for degrees in (0, 15, 30, 45, 90, 135, 180, 270):
        if abs(yaw_sideness(rolled(FRONTAL, degrees))) > 1e-9:
            fail(f"roll {degrees} deg must not look like yaw")
    # 鼻尖沿眼轴偏移越大越侧，且旋转后不失真。
    previous = -1.0
    for dx in (0, 4, 8, 12, 16, 20, 24, 30):
        value = yaw_sideness(yawed(dx))
        if value < previous - 1e-9:
            fail(f"sideness must be monotonic, dx={dx}")
        if value < 0.0 or value > 1.0:
            fail("sideness must be clipped to 0-1")
        previous = value
    if abs(yaw_sideness(yawed(8)) - 8 / 40 / 0.6) > 1e-9:
        fail("sideness must use the 0.6 normalisation of frontalness")
    if yaw_sideness(yawed(30)) != 1.0:
        fail("sideness must saturate at large yaw")
    if abs(yaw_sideness(rolled(yawed(16), 45)) - yaw_sideness(yawed(16))) > 1e-9:
        fail("sideness must be roll-invariant on yawed faces too")
    if yaw_sideness([(10.0, 10.0)] * 5) != 0.0:
        fail("degenerate kps must not trigger the swap-align gate")
    print("  ok  yaw_sideness: roll-invariant, monotonic, saturating")


def test_align_kps_gate() -> None:
    frontal = FRONTAL
    profile = yawed(20)                 # sideness 0.833
    good_align = shifted(profile, 2, 1)  # 几何自洽，代表 68 点结果
    bad_align = [(10.0, 10.0)] * 5     # 遮挡导致的退化 68 点
    thr = DEFAULT_ALIGN_FOR_SWAP_SIDENESS

    # 门 1：开关关闭 / 没有 68 点结果。
    if align_kps_for_swap(profile, good_align, None) is not profile:
        fail("disabled gate must keep the detector kps")
    if align_kps_for_swap(profile, None, thr) is not profile:
        fail("missing align_kps must keep the detector kps")
    # 门 2：正脸帧不折腾。
    if align_kps_for_swap(frontal, good_align, thr) is not frontal:
        fail("frontal frames must keep the detector kps (pixel regression)")
    # 门 3：68 点几何不自洽时回退。
    if align_kps_for_swap(profile, bad_align, thr) is not profile:
        fail("degenerate 68-point result must be rejected")
    # 大 yaw + 自洽 68 点：启用，返回的就是那个对象。
    chosen = align_kps_for_swap(profile, good_align, thr)
    if chosen is not good_align:
        fail("large-yaw frame with good align_kps must use it")
    # 阈值可配：侧度高于阈值时启用，抬高阈值后同一帧回退。
    near = yawed(0.35 * 0.6 * 40)
    align_obj = shifted(near, 1, 1)
    if align_kps_for_swap(near, align_obj, 0.34) is not align_obj:
        fail("sideness above the threshold must enable align_kps")
    if align_kps_for_swap(profile, good_align, 0.95) is not profile:
        fail("a very high threshold must disable the gate")
    if kps_quality(good_align) < kps_quality(profile):
        fail("test fixture itself is inconsistent")
    print("  ok  align_kps_for_swap: off / frontal / quality / large-yaw gates")


def test_cli_and_defaults() -> None:
    import argparse

    from faceswapper.cli import _sideness_spec, build_parser

    if _sideness_spec("0.5") != "0.5":
        fail("_sideness_spec must return the original string")
    for bad in ("abc", "1.5", "-0.1", "nan"):
        try:
            _sideness_spec(bad)
        except argparse.ArgumentTypeError:
            continue
        fail(f"invalid sideness {bad!r} was accepted")
    parser = build_parser()
    argv = ["video", "--source", "a.jpg", "--target", "a.mp4",
            "--output", "o.mp4", "--align", "68-point"]
    plain = parser.parse_args(argv)
    if plain.align_for_swap is not None:
        fail("--align-for-swap must default to None")
    bare = parser.parse_args(argv + ["--align-for-swap"])
    if bare.align_for_swap != str(DEFAULT_ALIGN_FOR_SWAP_SIDENESS):
        fail(f"bare flag must use the default, got {bare.align_for_swap!r}")
    custom = parser.parse_args(argv + ["--align-for-swap", "0.5"])
    if custom.align_for_swap != "0.5":
        fail("explicit threshold must round-trip")
    print("  ok  CLI: default off, bare flag uses 0.35, custom/illegal values")


def test_wiring_and_default_off() -> None:
    """接线 + 默认零行为变化（源码断言，本机无法 import GPU 栈）。"""
    cli = (ROOT / "faceswapper" / "cli.py").read_text(encoding="utf-8")
    headless = (ROOT / "faceswapper" / "headless.py").read_text(encoding="utf-8")
    vm = (ROOT / "faceswapper" / "VideoManager.py").read_text(encoding="utf-8")
    models = (ROOT / "faceswapper" / "Models.py").read_text(encoding="utf-8")
    retry = (ROOT / "faceswapper" / "detect_retry.py").read_text(
        encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if '"--align-for-swap"' not in cli or "--align-for-swap" not in readme:
        fail("CLI flag and README entry must both exist")
    if cli.count("align_for_swap=args.align_for_swap") != 2:
        fail("flag must reach both SwapOptions and VideoOptions")
    if "align_for_swap: str | None = None" not in headless:
        fail("SwapOptions missing the field")
    # 两个前置条件必须明确报错，不能静默不生效。
    for needle in (
        '--align-for-swap 需要 --align 68-point',
        '--align-for-swap 需要 --target-face',
        'self.vm.align_for_swap = float(align_swap_spec)',
    ):
        if needle not in headless:
            fail(f"headless missing prerequisite check: {needle!r}")
    # Models：可选第三返回值，默认调用方仍只看到两项。
    if "def run_recognize(self, img, kps, dim=1, with_align_kps=False):" not in models:
        fail("run_recognize must opt into align_kps")
    if "return out, cropped_image, kps_for_align" not in models:
        fail("recognize must expose kps_for_align")
    # VideoManager：ret 结构不变、旁路映射、两处 swap_core 都接上。
    if "self.align_for_swap: float | None = None" not in vm:
        fail("VideoManager must default the gate to None")
    if "ret.append([face_kps, face_emb])" not in vm:
        fail("ret must stay a 2-element list")
    if "for face_kps, face_emb in ret:" not in vm:
        fail("the 2-element unpacking must stay intact")
    if "align_kps_map[id(face_kps)] = align_kps" not in vm:
        fail("align_kps must travel through the side-channel map")
    if "with_align_kps=want_align_kps" not in vm:
        fail("first-pass recognize must opt in only when enabled")
    if vm.count("self._kps_for_swap(") < 2:
        fail("both main swap_core call sites must use the gate")
    # stdlib 侧的纯决策函数与姿态代理。
    for needle in ("def align_kps_for_swap(", "def yaw_sideness(",
                   "def local_frame(", "DEFAULT_ALIGN_FOR_SWAP_SIDENESS"):
        if needle not in retry:
            fail(f"detect_retry.py missing {needle!r}")
    print("  ok  wiring: prereqs, optional return, intact ret, default off")


def main() -> int:
    print("faceswapper align-for-swap tests")
    test_local_frame()
    test_yaw_sideness()
    test_align_kps_gate()
    test_cli_and_defaults()
    test_wiring_and_default_off()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
