"""单帧多脸 batch 的适用性判定（阶段 4 的 7.1，实验性，未启用）。

stdlib-only：这条判定只依赖整数矩形和参数 dict，可以脱离 GPU 栈做行为测试。
真正的 batch 前向在 ``VideoManager.swap_core_batched``，由
``EXPERIMENTAL_FACE_BATCH`` 闸门挡住，当前不会被执行。

判据见 docs/gpu-utilization-plan.md 7.1：只有 128 inswapper、所有脸
pipeline_size 都是 128、没有多 pass 开关、且 bbox 两两不重叠时，把 N 张脸
合成一次 swapper 前向才与逐脸路径等价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FaceCandidate:
    """一张待换脸的候选脸。

    ``bbox`` 是 paste_back 会写回的矩形 ``(left, top, right, bottom)``；
    ``pipeline_size`` 由该脸的 scale 算出，必须是 128 才能塞进统一 batch。
    """

    bbox: tuple[int, int, int, int]
    pipeline_size: int


# 这些开关会让 swap_core 跑不止一次 swapper 前向（强度迭代 / High Fidelity
# 第二道 pass）。单次 batch 结果无法表达迭代语义，必须回退逐脸路径。
INCOMPATIBLE_SWITCHES = ("StrengthSwitch", "HighFidelitySwitch")


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """两个矩形是否有正面积交集；边界相接不算重叠。"""
    l1, t1, r1, b1 = a
    l2, t2, r2, b2 = b
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def batch_eligibility(
    candidates: Sequence[FaceCandidate],
    *,
    swapper_type: str,
    parameters: Mapping[str, Any],
    target_face_filtering: bool = False,
    min_faces: int = 2,
) -> tuple[bool, str]:
    """这一帧能不能把 N 张脸合成一次 ``run_swapper_batched``。

    返回 ``(是否可以, 原因)``；原因用于日志与回退说明，不抛异常——不适用时
    调用方应静默回退到逐脸路径。

    最硬的一条是 bbox 不重叠：batch 的 prepare 阶段全部从**原始 img** warp，
    而 finish 阶段按原顺序写回。重叠时后一张脸的 ``pipeline_input_face``
    会与逐脸路径（读到的已是前一张写回后的像素）不同，回填补不回来。
    """
    n = len(candidates)
    if n < min_faces:
        return False, f"only {n} face(s), batch needs at least {min_faces}"
    if target_face_filtering:
        # --target-face 会按相似度只换选中的脸；batch 路径把所有脸都换成
        # 同一个 source，语义不同，必须回退逐脸。
        return False, "--target-face similarity filtering is not batched"
    if swapper_type != "inswapper-128":
        return False, (
            f"swapper {swapper_type!r} is not the 128 inswapper ONNX graph"
        )
    for key in INCOMPATIBLE_SWITCHES:
        if parameters.get(key):
            return False, f"{key} needs more than one swapper pass"

    sizes = {c.pipeline_size for c in candidates}
    if sizes != {128}:
        return False, f"pipeline_size must all be 128, got {sorted(sizes)}"

    boxes = [c.bbox for c in candidates]
    for i in range(n):
        for j in range(i + 1, n):
            if _overlaps(boxes[i], boxes[j]):
                return False, f"face {i} and face {j} have overlapping bboxes"
    return True, "eligible"
