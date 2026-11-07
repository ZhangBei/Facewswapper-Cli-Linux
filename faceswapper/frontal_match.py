"""姿态自适应的目标匹配阈值。

stdlib-only：``VideoManager`` 需要 torch / OpenCV 才能 import，而「这一张
检测脸的匹配线该下调多少」是 kps 的纯逻辑，可以脱离 GPU 栈做行为测试。

背景：ArcFace 相似度随偏航角系统性下降——同一个正脸参考图，侧脸帧的
cosine 分数天然低 10-20 分。单一 ``--target-threshold`` 只能二选一：调低
（误换到相似的人）或调高（侧脸帧漏换）。这里用检测已输出的 5 点关键点算
一个「正脸度」，对阈值做线性补偿：正脸帧判定线不变，越侧越松。

compensation 是「全侧脸时阈值下调的百分数」，0（默认）= 关闭，判定与旧
代码逐字等价。设计上无跨帧状态——frame_order.py 的 2N 有序窗口下帧会跨
worker 乱序完成，任何 EMA 都会让输出依赖调度顺序，破坏 workers=1 回归
底线。
"""

from __future__ import annotations

from typing import Any


def frontalness(kps: Any) -> float:
    """5 点关键点的正脸度，1 = 正脸，0 = 完全侧脸（也含退化输入）。

    kps 顺序与检测器一致：[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]。
    正脸时左右眼中点的 x 与鼻尖的 x 基本重合；偏航越大，鼻尖相对
    眼中点偏得越远。用眼距做分母归一化，平移缩放不变。
    """
    try:
        e_l, e_r, nose = kps[0], kps[1], kps[2]
        eye_dist = float(e_r[0]) - float(e_l[0])
        if eye_dist <= 0.0:
            return 0.0
        eye_mid_x = (float(e_l[0]) + float(e_r[0])) / 2.0
        offset = abs(float(nose[0]) - eye_mid_x) / eye_dist
    except (IndexError, TypeError, ValueError):
        return 0.0
    # 正脸 offset≈0，全侧脸经验上 ≈0.5-0.8。归一到 0-1：offset >= 0.6
    # 视为最侧。clip 掉极端姿态（大倾角、俯仰）造成的超界值。
    return max(0.0, min(1.0, 1.0 - offset / 0.6))


def effective_threshold(kps: Any, base_threshold: float, compensation: float) -> float:
    """这一张脸应该用的匹配线（单侧姿态：只有检测脸的 kps）。

    compensation=0 恒等于 base_threshold——保证默认关闭时与旧判定逐字
    等价；compensation=30 表示全侧脸（frontalness=0）的判定线比正脸低
    30 分，线性插值。
    """
    return effective_threshold_pair(kps, None, base_threshold, compensation)


def effective_threshold_pair(
    kps_face: Any, kps_ref: Any, base_threshold: float, compensation: float,
) -> float:
    """配对版：参考图自身的姿态也参与补偿。

    ArcFace 相似度随**两张脸的姿态差**下降——侧面参考图匹配正脸帧时
    分数同样系统性偏低，单一阈值对参考图姿态误校准。把两张脸的"侧度"
    相加（各占 0-1，封顶 1）作为补偿比例：正脸对正脸不补偿，正脸对全侧
    与全侧对正脸补偿满额，两张都全侧不叠加超过满额。

    kps_ref 为 None（参考图姿态未知）时退化为只看检测脸，与
    `effective_threshold` 一致。compensation=0 恒等于 base_threshold。
    """
    if compensation <= 0.0:
        return base_threshold
    sideness = 1.0 - frontalness(kps_face)
    if kps_ref is not None:
        sideness += 1.0 - frontalness(kps_ref)
    drop = float(compensation) * min(1.0, sideness)
    return base_threshold - drop
