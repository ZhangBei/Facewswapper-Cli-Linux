"""单帧多脸 batch 的适用性判定（阶段 4 的 7.1，实验性，未启用）。

stdlib-only：这条判定只依赖整数矩形和参数 dict，可以脱离 GPU 栈做行为测试。
真正的 batch 前向在 ``VideoManager.swap_core_batched``，由
``EXPERIMENTAL_FACE_BATCH`` 闸门挡住，当前不会被执行。

判据见 docs/gpu-utilization-plan.md 7.1：只有 128 inswapper、模型支持
动态 batch、所有脸 pipeline_size 都是 128、没有多 pass 开关、且 bbox 两两
不重叠时，把 N 张脸合成一次 swapper 前向才与逐脸路径等价。

``batch_backend_available`` 与 7.2 的 ``cross_frame_batch_eligibility``
同名同义，且刻意**不设默认值**：调用方必须显式给出模型能力，避免把
「session 尚未懒加载、探测标志还是默认 False」误读成支持动态 batch。
11-06 的受控开闸实验正是漏了这条，N=3 的 batch 被 ONNX 拒绝后触发
``CUDA failure 700``。
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
    batch_backend_available: bool,
    target_face_filtering: bool = False,
    min_faces: int = 2,
) -> tuple[bool, str]:
    """这一帧能不能把 N 张脸合成一次 ``run_swapper_batched``。

    返回 ``(是否可以, 原因)``；原因用于日志与回退说明，不抛异常——不适用时
    调用方应静默回退到逐脸路径。

    最硬的一条是 bbox 不重叠：batch 的 prepare 阶段全部从**原始 img** warp，
    而 finish 阶段按原顺序写回。重叠时后一张脸的 ``pipeline_input_face``
    会与逐脸路径（读到的已是前一张写回后的像素）不同，回填补不回来。

    ``batch_backend_available`` 是 11-06 受控实验后补的硬条件：某些
    inswapper 导出把 batch 维锁死为 1（日志里的 ``static-batch=1``），
    此时 N>1 的 io_binding 会被 ONNX 拒绝，绝不能走到 ``run_swapper_batched``。
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
    if not batch_backend_available:
        return False, "swapper export locks the batch dimension to 1"
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
