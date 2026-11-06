"""跨帧 batch 的适用性判定与分组（阶段 4 的 7.2，实验性，未启用）。

stdlib-only：判定只看帧元数据与参数 dict，分组只做有序切片，因此整条逻辑
能脱离 GPU 栈做行为测试。

跨帧 batch 想把 N 个 worker 的帧拼成一次 ORT Run，但会与 PTS 保序、
``--every-nth``、可变人脸数冲突（见 docs/gpu-utilization-plan.md 7.2）。
本模块只负责「什么时候允许做」与「怎么有序分组」；推理侧的回填与时序
验证必须在 GPU 环境用逐帧 PSNR 基线解冻，当前闸门恒为 False。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

# 闸门：关闭时 resolve_cross_frame_batch 一律拒绝。主流水线（frame_order /
# headless / VideoManager._swap_video_inner）不调用本模块，改动它不会改变
# 任何现有输出。解冻步骤见方案文档 7.2。
EXPERIMENTAL_CROSS_FRAME_BATCH = False


@dataclass(frozen=True)
class FrameCandidate:
    """一帧参与跨帧 batch 所需的元数据。"""

    index: int          # 解码顺序，必须连续
    pts: int            # 相对时间戳，必须严格递增
    face_count: int     # 该帧检测到的人脸数
    pipeline_size: int  # 由该脸 scale 算出，必须恒为 128


# 与 7.1 同理：这些开关会让单帧跑不止一次 swapper 前向，单次 batch 结果
# 无法表达迭代语义。
INCOMPATIBLE_SWITCHES = ("StrengthSwitch", "HighFidelitySwitch")


def cross_frame_batch_eligibility(
    candidates: Sequence[FrameCandidate],
    *,
    workers: int,
    every_nth: int,
    swapper_type: str,
    parameters: Mapping[str, Any],
    batch_backend_available: bool,
    min_batch: int = 2,
) -> tuple[bool, str]:
    """纯判定：这一批帧能否合成一次跨帧 ORT Run。

    不检查闸门，方便在闸门关闭时仍然测试判定逻辑本身。所有条件都是硬性的：

    - ``workers >= 2``：串行路径没有可合并的并发帧。
    - ``every_nth == 1``：跳帧会让 batch 里的帧不对齐。
    - 仅 ``inswapper-128``、``pipeline_size`` 恒为 128。
    - ``batch_backend_available``：某些 inswapper 导出把 batch 维锁死为 1，
      ``Models._swapper_batch_unsupported`` 探测到后必须回退。
    - 每帧人脸数恒为 1：可变人脸数会让 batch 维变化，且逐脸 paste-back
      的顺序语义无法用一次前向表达。
    - ``index`` 连续、``pts`` 严格递增：VFR 时序不能被 batch 重排。
    """
    n = len(candidates)
    if n < min_batch:
        return False, f"only {n} frame(s), cross-frame batch needs {min_batch}"
    if workers < 2:
        return False, "cross-frame batch needs at least 2 workers"
    if every_nth != 1:
        return False, "--every-nth skips frames and breaks batch alignment"
    if swapper_type != "inswapper-128":
        return False, f"swapper {swapper_type!r} is not the 128 inswapper graph"
    for key in INCOMPATIBLE_SWITCHES:
        if parameters.get(key):
            return False, f"{key} needs more than one swapper pass"
    if not batch_backend_available:
        return False, "swapper export locks the batch dimension to 1"

    if {c.face_count for c in candidates} != {1}:
        return False, "every frame must contain exactly one face"
    if {c.pipeline_size for c in candidates} != {128}:
        return False, "pipeline_size must all be 128"

    indices = [c.index for c in candidates]
    if indices != list(range(indices[0], indices[0] + n)):
        return False, "frame indices are not contiguous"
    pts = [c.pts for c in candidates]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        return False, "PTS is not strictly increasing (VFR ordering)"
    return True, "eligible"


def resolve_cross_frame_batch(
    candidates: Sequence[FrameCandidate], **kwargs: Any
) -> tuple[bool, str]:
    """闸门 + 判定。主路径解冻后应调用这个入口，而不是直接调判定函数。"""
    if not EXPERIMENTAL_CROSS_FRAME_BATCH:
        return False, "experimental cross-frame batch is disabled by the gate"
    return cross_frame_batch_eligibility(candidates, **kwargs)


def group_frames(
    candidates: Sequence[FrameCandidate], batch_size: int
) -> list[list[FrameCandidate]]:
    """按固定大小切成有序分组，组内顺序与解码顺序一致。

    最后一组允许不足 ``batch_size``（由调用方决定是否丢弃或逐帧处理）。
    PTS 保序依赖这个顺序：分组只能切，不能排序。
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    return [
        list(candidates[start:start + batch_size])
        for start in range(0, len(candidates), batch_size)
    ]


@dataclass(frozen=True)
class BatchRunResult:
    """一次跨帧 batch 编排的结果。"""

    status: str  # "gated" | "fallback" | "batched"
    reason: str
    frames: int = 0


def run_cross_frame_batch(
    *,
    gate_open: bool,
    eligible: tuple[bool, str],
    prepare: Callable[[int, Any], None],
    infer: Callable[[int], None],
    finish: Callable[[int, Any], None],
    fallback: Callable[[], None],
    frame_items: Sequence[Any],
) -> BatchRunResult:
    """编排「prepare 全部帧 → 一次 infer → 按序 finish」，回调全部注入。

    抽成纯函数是为了让顺序契约能在没有 GPU / torch 的环境下做行为测试：
    ``VideoManager.cross_frame_batch_swap`` 只负责把真实回调绑进来。

    顺序是硬要求：prepare 必须全部完成（收集完各帧输入）才能 infer，finish
    必须按原顺序执行（paste_back 与 PTS 保序）。闸门关闭或判定不通过时
    **一次 prepare 都不调用**，直接走 fallback，保证输入帧未被改动。
    """
    if not gate_open:
        return BatchRunResult("gated", "experimental gate is closed")
    ok, reason = eligible
    if not ok:
        fallback()
        return BatchRunResult("fallback", reason)

    for index, item in enumerate(frame_items):
        prepare(index, item)
    infer(len(frame_items))
    for index, item in enumerate(frame_items):
        finish(index, item)
    return BatchRunResult("batched", reason, len(frame_items))
