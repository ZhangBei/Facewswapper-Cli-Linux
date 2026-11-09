"""--probe-angles 的离线行为测试（无 GPU / torch / numpy / OpenCV）。

覆盖三件事：

1. 步长解析与 ``choose_angle`` 的向后兼容（不传 angles 时旧断言仍成立）。
2. 仿射几何：0-345 度逐 15 度，A 与 inv(A) 的 5 点往返、以及
   ``normalized_grid_matrix`` 的像素往返。这是「只旋转检测副本」路径
   正确性的全部可离线验证部分——A 由本模块自己构造，不需要现场标定。
3. 接线：默认 ``probe_angles=None`` 时旧整帧旋转路径逐字保留。

    python -m faceswapper.tests.test_probe_angles
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from faceswapper.orientation import (
    PROBE_ANGLES,
    apply_affine,
    choose_angle,
    invert_affine,
    map_kps_back,
    normalized_grid_matrix,
    parse_probe_angles,
    rotation_affine,
)

ROOT = Path(__file__).resolve().parents[2]
PTS = ((10.0, 20.0), (100.0, 50.0), (640.0, 360.0),
       (0.0, 0.0), (1919.0, 1079.0))


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_parse_probe_angles() -> None:
    # None / 空串 = 旧路径，是逐像素回归底线的开关。
    if parse_probe_angles(None) is not None:
        fail("None must map to None (legacy path)")
    if parse_probe_angles("  ") is not None:
        fail("blank spec must map to None")
    if parse_probe_angles("90") != (90.0, 180.0, 270.0):
        fail("step 90 must reproduce the three quarter turns")
    eight = parse_probe_angles("45")
    if len(eight) != 7 or eight[0] != 45.0 or eight[-1] != 315.0:
        fail(f"step 45 must give 7 non-zero axes (8-way), got {eight!r}")
    sixteen = parse_probe_angles("22.5")
    if len(sixteen) != 15 or sixteen[0] != 22.5 or sixteen[-1] != 337.5:
        fail(f"step 22.5 must give 15 non-zero axes, got {sixteen[:3]!r}")
    # 0 度永远由第一遍检测覆盖，不进候选轴。
    if 0.0 in eight or 0.0 in sixteen:
        fail("0 degree must not be probed again")
    for bad in ("7", "0", "-45", "abc", "361", "720", "1e309"):
        try:
            parse_probe_angles(bad)
        except ValueError:
            continue
        fail(f"invalid step {bad!r} was accepted")
    print("  ok  parse_probe_angles: None / 4 / 8 / 16 way, invalid steps rejected")


def test_choose_angle_backwards_compatible() -> None:
    # 旧断言逐字保留：非候选角度静默忽略、平手取最小角度。
    if choose_angle({0: 99, 45: 99, 360: 99, 90: 1}) != 90:
        fail("default choose_angle must ignore foreign keys")
    if choose_angle({90: 2, 180: 2, 270: 2}) != 90:
        fail("tie must pick the smallest qualifying angle")
    if choose_angle({}) != 0:
        fail("empty counts must keep the original frame")
    # 显式传 angles 后，45 度可以胜出。
    axes = parse_probe_angles("45")
    if choose_angle({45.0: 3, 90.0: 1, 135.0: 0}, axes) != 45.0:
        fail("explicit axes must allow a 45-degree winner")
    if choose_angle({45.0: 0, 90.0: 0}, axes) != 0:
        fail("all-zero explicit counts must keep the original frame")
    print("  ok  choose_angle: legacy behaviour pinned, explicit axes opt-in")


def test_affine_roundtrip() -> None:
    worst = 0.0
    for angle in range(0, 360, 15):
        A, dst_hw = rotation_affine(angle, (1080, 1920))
        inv_A = invert_affine(A)
        M = normalized_grid_matrix(inv_A, (1080, 1920), dst_hw)
        if len(M) != 2 or any(len(row) != 3 for row in M):
            fail(f"grid matrix must be 2x3 at {angle}")
        for x, y in PTS:
            u, v = apply_affine(A, x, y)
            bx, by = apply_affine(inv_A, u, v)
            worst = max(worst, abs(bx - x), abs(by - y))
            # grid_sample 的 theta 作用在归一化坐标上；反解回应落在 x, y。
            gx = (u + 0.5) * 2.0 / dst_hw[1] - 1.0
            gy = (v + 0.5) * 2.0 / dst_hw[0] - 1.0
            ix = M[0][0] * gx + M[0][1] * gy + M[0][2]
            iy = M[1][0] * gx + M[1][1] * gy + M[1][2]
            px = (ix + 1.0) * 1920 / 2.0 - 0.5
            py = (iy + 1.0) * 1080 / 2.0 - 0.5
            worst = max(worst, abs(px - x), abs(py - y))
    if worst > 1e-8:
        fail(f"affine/grid roundtrip error {worst:.3e} px exceeds 1e-8")
    # 45 度外接矩形：16:9 帧与方图的膨胀倍率都应与解析值一致。
    _, dst = rotation_affine(45.0, (1080, 1920))
    if dst != (2122, 2122):
        fail(f"45-degree 16:9 canvas should be 2122x2122, got {dst!r}")
    _, sq = rotation_affine(45.0, (1024, 1024))
    if sq[0] < 1447 or sq[0] > 1449:
        fail(f"45-degree square canvas should be ~1448, got {sq!r}")
    print(f"  ok  affine/grid roundtrip 24 angles, worst {worst:.2e} px")


def test_map_kps_back() -> None:
    A, dst_hw = rotation_affine(45.0, (1080, 1920))
    inv_A = invert_affine(A)
    rotated = []
    for face in (PTS, tuple((x + 3.0, y - 2.0) for x, y in PTS)):
        rotated.append([apply_affine(A, x, y) for x, y in face])
    back = map_kps_back(rotated, inv_A)
    if len(back) != 2 or any(len(face) != 5 for face in back):
        fail("map_kps_back must preserve (N,5,2)")
    worst = 0.0
    for i, face in enumerate(rotated):
        for j, (x, y) in enumerate(face):
            bx, by = back[i][j]
            ox, oy = apply_affine(inv_A, x, y)
            worst = max(worst, abs(bx - ox), abs(by - oy))
    if worst > 1e-9:
        fail(f"map_kps_back disagrees with inv(A): {worst:.3e}")
    if map_kps_back([], inv_A) != []:
        fail("empty detections must map to an empty list")
    print("  ok  map_kps_back: preserves (N,5,2), matches inv(A), handles empty")


def test_rotation_matrix_shape() -> None:
    # 纯旋转 + 平移：det(A) 恒为 1，逆向矩阵不带缩放/剪切。
    for angle in (22.5, 45.0, 90.0, 135.0, 225.0, 315.0):
        A, _ = rotation_affine(angle, (1080, 1920))
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if not math.isclose(det, 1.0, abs_tol=1e-9):
            fail(f"det(A) must be 1 at {angle}, got {det}")
        if A[2] != (0.0, 0.0, 1.0):
            fail(f"last row must be homogeneous at {angle}")
    print("  ok  rotation_affine: pure rotation (det=1) at all probe axes")


def test_cli_step_validation() -> None:
    """非法步长在 argparse 阶段就被拒，不拖到模型下载 / 逐帧路径。"""
    import argparse

    from faceswapper.cli import _probe_step, build_parser

    if _probe_step("45") != "45":
        fail("_probe_step must return the original string")
    for bad in ("7", "0", "-45", "abc"):
        try:
            _probe_step(bad)
        except argparse.ArgumentTypeError:
            continue
        fail(f"invalid step {bad!r} was not rejected by argparse")
    parser = build_parser()
    args = parser.parse_args([
        "video", "--source", "a.jpg", "--target", "a.mp4",
        "--output", "o.mp4", "--probe-angles", "45",
    ])
    if args.probe_angles != "45":
        fail(f"--probe-angles did not round-trip, got {args.probe_angles!r}")
    print("  ok  CLI validates the step up front and round-trips the value")


def test_wiring_and_legacy_path() -> None:
    """接线 + 默认零行为变化（源码断言，本机无法 import GPU 栈）。"""
    cli = (ROOT / "faceswapper" / "cli.py").read_text(encoding="utf-8")
    headless = (ROOT / "faceswapper" / "headless.py").read_text(encoding="utf-8")
    vm = (ROOT / "faceswapper" / "VideoManager.py").read_text(encoding="utf-8")
    if '"--probe-angles"' not in cli:
        fail("cli.py missing --probe-angles")
    if cli.count("probe_angles=args.probe_angles") != 2:
        fail("--probe-angles must reach both SwapOptions and VideoOptions")
    if "probe_angles: str | None = None" not in headless:
        fail("SwapOptions missing probe_angles field")
    if "parse_probe_angles(" not in headless:
        fail("headless must parse probe_angles before the frame loop")
    if "self.probe_angles: tuple[float, ...] | None = None" not in vm:
        fail("VideoManager must default probe_angles to None")
    if "def _rotate_detect_copy(self, img, angle):" not in vm:
        fail("VideoManager missing _rotate_detect_copy")
    if "choose_angle(counts, self.probe_angles)" not in vm:
        fail("new path must pass explicit axes to choose_angle")
    # 旧路径必须逐字留在默认分支里，否则 workers=1 逐像素回归不成立。
    probe_block = vm.split("if should_probe(", 1)[-1].split("漏检重试", 1)[0]
    branches = probe_block.split("if self.probe_angles is None:", 1)[-1]
    legacy = branches.split("else:", 1)[0]
    for needle in (
        "counts = {}",
        "for angle in PROBE_ANGLES:",
        "v2.functional.rotate(",
        "fallback_angle = choose_angle(counts)",
        "applied_angle = float(fallback_angle)",
    ):
        if needle not in legacy:
            fail(f"legacy probe branch lost {needle!r}")
    # 新路径不得碰整帧 img，也不得设置 applied_angle。
    fresh = branches.split("else:", 1)[-1]
    if "applied_angle = " in fresh:
        fail("new path must not rotate the frame (applied_angle stays 0)")
    if "self._collect_angle_candidates(" not in fresh:
        fail("new path must collect candidates via the shared helper")
    collector = vm.split("def _collect_angle_candidates(", 1)[-1].split(
        "def _probe_on_target_miss(", 1)[0]
    if "map_kps_back(" not in collector:
        fail("candidate collection must map kps back to frame coordinates")
    if "with_scores=True" not in collector:
        fail("candidate collection must opt into scores for merging")
    if PROBE_ANGLES != (90, 180, 270):
        fail("PROBE_ANGLES must stay unchanged for the legacy path")
    print("  ok  wiring: CLI/headless/VM connected, legacy branch preserved")


def main() -> int:
    print("faceswapper probe-angles tests")
    test_parse_probe_angles()
    test_choose_angle_backwards_compatible()
    test_cli_step_validation()
    test_affine_roundtrip()
    test_map_kps_back()
    test_rotation_matrix_shape()
    test_wiring_and_legacy_path()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
