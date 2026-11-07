"""frontal_match 的离线行为测试（无 GPU / torch / OpenCV）。

frontal_match.effective_threshold 是 --target-face 匹配判定的姿态补偿：
正脸不补偿，越侧判定线越松，compensation=0 时与旧判定逐字等价。

    python -m faceswapper.tests.test_frontal_match
"""

from __future__ import annotations

import sys

from faceswapper.frontal_match import (
    effective_threshold,
    effective_threshold_pair,
    frontalness,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


# 合成 5 点 kps：[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]。
# 眼距 100，眼 y=50；正脸鼻 x=50（眼中点），侧脸鼻越偏 x 偏差越大。
def kps_with_nose(nose_x: float) -> list[list[float]]:
    return [[0.0, 50.0], [100.0, 50.0], [nose_x, 60.0],
            [30.0, 90.0], [70.0, 90.0]]


def test_frontalness() -> None:
    if frontalness(kps_with_nose(50.0)) != 1.0:
        fail("frontal nose should give frontalness 1.0")
    # 鼻尖偏 30（0.3 眼距）→ offset/0.6 = 0.5 → frontalness 0.5。
    got = frontalness(kps_with_nose(20.0))
    if abs(got - 0.5) > 1e-9:
        fail(f"nose offset 0.3 eye-dists should give 0.5, got {got}")
    # 偏到界外（0.6 眼距以上）钉在 0，不允许负数漏进阈值里。
    if frontalness(kps_with_nose(-10.0)) != 0.0:
        fail("extreme nose offset should clamp to 0.0")
    if frontalness(kps_with_nose(110.0)) != 0.0:
        fail("nose beyond the right eye should clamp to 0.0")
    # 退化输入（空 kps、坏形状、眼距非正）一律 0——最侧档，不会崩溃。
    for bad in ([], [[1.0]], [[100.0, 50.0], [0.0, 50.0], [50.0, 60.0]],
                None):
        if frontalness(bad) != 0.0:
            fail(f"degenerate kps {bad!r} should give frontalness 0.0")
    print("  ok  frontalness: frontal=1, linear in offset, clamped, "
          "degenerate-safe")


def test_effective_threshold() -> None:
    # compensation=0：任何姿态、任何 kps（含 None）都逐字等于原阈值。
    for k in (kps_with_nose(50.0), kps_with_nose(0.0), None):
        if effective_threshold(k, 55.0, 0.0) != 55.0:
            fail(f"compensation 0 must equal base for kps {k!r}")
    # 正脸全补偿档：不加分。
    if effective_threshold(kps_with_nose(50.0), 55.0, 30.0) != 55.0:
        fail("frontal face must not be compensated")
    # 全侧脸全补偿档：55 - 30。
    if effective_threshold(kps_with_nose(-10.0), 55.0, 30.0) != 25.0:
        fail("full-side face at compensation 30 must drop 30 points")
    # 中间姿态线性插值：offset 0.3 → 一半补偿。
    got = effective_threshold(kps_with_nose(20.0), 55.0, 30.0)
    if abs(got - 40.0) > 1e-9:
        fail(f"half-side at compensation 30 should give 40.0, got {got}")
    print("  ok  effective_threshold: off=identity, linear, clamped")


def test_effective_threshold_pair() -> None:
    frontal = kps_with_nose(50.0)
    side = kps_with_nose(-10.0)
    # kps_ref=None 退化为单侧版。
    if effective_threshold_pair(side, None, 55.0, 30.0) != \
            effective_threshold(side, 55.0, 30.0):
        fail("pair with kps_ref=None must equal single-side version")
    # 补偿 0 恒等（含两侧都全侧）。
    if effective_threshold_pair(side, side, 55.0, 0.0) != 55.0:
        fail("compensation 0 must equal base even for side/side pair")
    # 正脸对正脸：不补偿。
    if effective_threshold_pair(frontal, frontal, 55.0, 30.0) != 55.0:
        fail("frontal/frontal pair must not be compensated")
    # 正脸对全侧与全侧对正脸：对称，补偿满额。
    for a, b in ((frontal, side), (side, frontal)):
        if effective_threshold_pair(a, b, 55.0, 30.0) != 25.0:
            fail(f"cross-pair {a is frontal}/{b is frontal} must drop full 30")
    # 两张都全侧：侧度相加封顶 1，不叠加成 -35。
    if effective_threshold_pair(side, side, 55.0, 30.0) != 25.0:
        fail("side/side pair must clamp at full compensation (25), not -35")
    print("  ok  effective_threshold_pair: None fallback, symmetric, capped")


def test_default_off_wiring() -> None:
    """参数默认 0 且 CLI 折叠链不改动匹配判定。"""
    from faceswapper.parameters import default_values

    dv = default_values()
    if dv.get("PoseCompensationSlider") != 0:
        fail(f"PoseCompensationSlider default must be 0, got "
             f"{dv.get('PoseCompensationSlider')!r}")
    print("  ok  PoseCompensationSlider defaults to 0 (off)")


def main() -> int:
    print("faceswapper frontal_match tests")
    test_frontalness()
    test_effective_threshold()
    test_effective_threshold_pair()
    test_default_off_wiring()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
