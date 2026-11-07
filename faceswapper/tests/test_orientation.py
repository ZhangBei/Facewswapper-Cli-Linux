"""orientation 的离线行为测试（无 GPU / torch / OpenCV）。

逐帧无状态的整帧朝向兜底：正常帧零额外推理，只有 0 检出帧才试 90/180/270，
且只用 90 度倍数以保证 expand=True 的正/逆旋转可逆。

    python -m faceswapper.tests.test_orientation
"""

from __future__ import annotations

import sys

from faceswapper.orientation import PROBE_ANGLES, choose_angle, should_probe


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_probe_angles_are_quarter_turns() -> None:
    # 非 90 倍数的角度会让 rotate(expand=True) 的正/逆两次调用把画布越滚
    # 越大；0 度由第一遍检测覆盖，不重复推理。这两条是几何与开销的底线。
    if PROBE_ANGLES != (90, 180, 270):
        fail(f"probe angles must be exactly (90, 180, 270), got {PROBE_ANGLES!r}")
    for angle in PROBE_ANGLES:
        if angle % 90 != 0:
            fail(f"{angle} is not a multiple of 90; frame size would drift")
    print("  ok  probe angles are exactly the three quarter turns")


def test_should_probe_only_on_empty_frames() -> None:
    # 默认开启，0 检出才探测。
    assert should_probe(detected=0, enabled=True) is True
    # 正常帧：零额外推理、零旋转——默认路径逐像素不变的关键。
    for n in (1, 2, 10):
        assert should_probe(detected=n, enabled=True) is False
    # 回滚开关。
    assert should_probe(detected=0, enabled=False) is False
    # GUI 时代的 OrientAutoSwitch / OrientSwitch 已转过一次，不叠加。
    assert should_probe(detected=0, enabled=True, already_rotated=True) is False
    # parameter dict 里可能传来字符串。
    assert should_probe(detected="0", enabled=True) is True
    assert should_probe(detected="2", enabled=True) is False
    print("  ok  should_probe: only empty frames, gated by flag and prior rotation")


def test_choose_angle() -> None:
    # 全 0 或空 dict：保持原帧，不冒误检风险。
    assert choose_angle({}) == 0
    assert choose_angle({90: 0, 180: 0, 270: 0}) == 0
    # 各角度胜出。
    assert choose_angle({90: 3, 180: 1, 270: 0}) == 90
    assert choose_angle({90: 1, 180: 4, 270: 2}) == 180
    assert choose_angle({90: 0, 180: 1, 270: 5}) == 270
    # 平手取最小角度（严格 > 才更新），与"证据不足时倾向不折腾"一致。
    assert choose_angle({90: 2, 180: 2, 270: 2}) == 90
    assert choose_angle({90: 0, 180: 2, 270: 2}) == 180
    # 非候选角度（0 度、任意角度）静默忽略，不让调用方混进来。
    assert choose_angle({0: 99, 45: 99, 360: 99, 90: 1}) == 90
    assert choose_angle({0: 99, 45: 99, 360: 99}) == 0
    # 负数计数也不能把 best_count 拉偏。
    assert choose_angle({90: -1, 180: 0, 270: 0}) == 0
    print("  ok  choose_angle: winner / tie / all-zero / foreign keys pinned")


def main() -> int:
    print("faceswapper orientation tests")
    test_probe_angles_are_quarter_turns()
    test_should_probe_only_on_empty_frames()
    test_choose_angle()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
