"""漏检帧的检测重试计划。

stdlib-only：``VideoManager`` 需要 torch / OpenCV 才能 import，而「这一帧
要不要重试、用什么参数重试」是纯逻辑，可以脱离 GPU 栈做行为测试。

背景：``run_detect`` 返回空 ``kpss`` 的帧直接透传（不换脸）。侧脸/小脸
漏检常是「这个输入尺寸下过不了置信度」，升一档输入尺寸就能检出。只在
**0 检出**的帧上追加推理——已有检出的帧绝不重试，正常路径行为逐字不变，
开销为零（一次布尔判断）。

防误检的两道闸：
1. 重试轮用更高的分数阈值（+0.2，即 20 分）：背景误检通常是低置信度，
   第一遍用用户阈值没检出的帧，第二遍用更高阈值检出，误检增量严格小于
   正检。
2. auto-orient（OrientAutoSwitch）开启时不重试——它自己的
   ``orient_miss_streak`` 已在检测失败后做 4 向重探，横竖颠倒类漏检由它
   负责，双开会在同一帧跑两套试探。
"""

from __future__ import annotations

from typing import Any


# 重试轮在用户阈值上抬升的分数（0-100 刻度）。
RETRY_SCORE_BUMP = 20.0
# 重试用的检测输入尺寸。要求是 32 的倍数（检测器动态输入约束），取
# DetectInputSizeTextSelModes 的最大档。
RETRY_INPUT_SIZE = 960


def retry_plan(
    *,
    detected: int,
    input_size: Any,
    detect_score: float,
    auto_orient: bool = False,
    enabled: bool = False,
) -> tuple[int, float] | None:
    """0 检出帧的重试参数 ``(input_size, score)``；其余情况返回 None。

    - detected > 0：帧上有脸，绝不重试。
    - auto_orient：4 向重探已由 OrientAutoSwitch 负责，双开会在同一帧
      跑两套试探。
    - 输入尺寸已升到最大档（>= RETRY_INPUT_SIZE）时再升没有意义，放弃。
    """
    if not enabled or detected > 0 or auto_orient:
        return None
    if int(input_size) >= RETRY_INPUT_SIZE:
        return None
    score = min(1.0, float(detect_score) + RETRY_SCORE_BUMP / 100.0)
    return RETRY_INPUT_SIZE, score
