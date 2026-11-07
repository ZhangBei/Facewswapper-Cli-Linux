"""detect_retry 的离线行为测试（无 GPU / torch / OpenCV）。

detect_retry.retry_plan 是 0 检出帧的重试判定：只在漏检帧追加一次升档
检测，两道防误检闸（更高分数阈值、auto-orient 让位）都在这里钉住。

    python -m faceswapper.tests.test_detect_retry
"""

from __future__ import annotations

import sys

from faceswapper.detect_retry import RETRY_INPUT_SIZE, retry_plan


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_retry_plan() -> None:
    # 默认关闭：任何输入都不重试。
    assert retry_plan(detected=0, input_size=640, detect_score=0.5) is None
    # 帧上有脸：绝不重试（误检防线之一）。
    assert retry_plan(detected=2, input_size=640, detect_score=0.5,
                      enabled=True) is None
    # auto-orient 让位：4 向重探已由 OrientAutoSwitch 负责。
    assert retry_plan(detected=0, input_size=640, detect_score=0.5,
                      auto_orient=True, enabled=True) is None
    # 0 检出 + 开启：升到 960、分数 +0.2。
    plan = retry_plan(detected=0, input_size=640, detect_score=0.5,
                      enabled=True)
    if plan != (960, 0.7):
        fail(f"expected (960, 0.7), got {plan!r}")
    # 阈值抬升封顶在 1.0。
    plan = retry_plan(detected=0, input_size=640, detect_score=0.95,
                      enabled=True)
    if plan is None or plan[1] != 1.0:
        fail(f"score bump must clamp at 1.0, got {plan!r}")
    # 已在最大档（960/800 不算——800 < 960 会升；960 及以上放弃）。
    assert retry_plan(detected=0, input_size=960, detect_score=0.5,
                      enabled=True) is None
    assert retry_plan(detected=0, input_size=1120, detect_score=0.5,
                      enabled=True) is None
    plan = retry_plan(detected=0, input_size=800, detect_score=0.5,
                      enabled=True)
    if plan is None or plan[0] != RETRY_INPUT_SIZE:
        fail(f"input_size 800 should retry at {RETRY_INPUT_SIZE}, got {plan!r}")
    # 字符串输入尺寸（parameter dict 里存的是 str）也要能比较。
    plan = retry_plan(detected=0, input_size="640", detect_score=0.5,
                      enabled=True)
    if plan != (960, 0.7):
        fail(f"string input_size must work, got {plan!r}")
    print("  ok  retry_plan: off/gated/score-bump/max-size/string all pinned")


def test_default_off_wiring() -> None:
    from faceswapper.parameters import default_values

    dv = default_values()
    if dv.get("DetectRetrySwitch") is not False:
        fail(f"DetectRetrySwitch default must be False, got "
             f"{dv.get('DetectRetrySwitch')!r}")
    print("  ok  DetectRetrySwitch defaults to False (off)")


def main() -> int:
    print("faceswapper detect_retry tests")
    test_retry_plan()
    test_default_off_wiring()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
