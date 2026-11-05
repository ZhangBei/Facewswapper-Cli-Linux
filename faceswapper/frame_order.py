"""有序并行的换脸帧调度器。

stdlib-only：``VideoManager`` / ``Models`` 的重运行时一概不 import，这样
「多线程乱序完成、但输出严格按解码顺序」这条硬约束可以在没有 GPU 栈的
机器上用假 ``swap_fn`` 做**行为测试**，不必靠读源码字符串。

窗口最多 ``2*workers`` 帧：主线程编码队首时，后续帧已并行在 GPU 上推进。
每个 worker 各持一份 ORT session / CUDA stream（由调用方配置），显存占用
随 workers 上升。
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator

Frame = tuple[int, Any, Any]  # (index, rgb, pts)


def ordered_swap_frames(
    frames: Iterable[Frame],
    workers: int,
    step: int,
    swap_fn: Callable[[Any], Any],
) -> Iterator[Frame]:
    """把换脸预提交给 worker 线程池，仍严格按解码顺序 yield。

    ``frames`` 产出 ``(index, rgb, pts)``。只有 ``index % step == 0`` 的帧
    交给 ``swap_fn``，其余原样透传（``--every-nth`` 语义）；``pts`` 原封
    不动，因此 VFR 时序不受影响。

    只有队首 future 完成才交给消费方：即使后续帧先算完，也不会提前 yield，
    每帧拿到的仍是它自己的结果而不是邻居的。``swap_fn`` 抛出的异常在队首
    ``future.result()`` 处如实传播。
    """
    worker_count = max(1, int(workers))
    nth = max(1, int(step))
    pending: deque[tuple[int, Any, Any, Any]] = deque()
    limit = worker_count * 2
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for index, rgb, pts in frames:
            future = (
                pool.submit(swap_fn, rgb) if index % nth == 0 else None
            )
            pending.append((index, rgb, pts, future))
            if len(pending) >= limit:
                i, original, p, f = pending.popleft()
                yield i, (f.result() if f is not None else original), p
        while pending:
            i, original, p, f = pending.popleft()
            yield i, (f.result() if f is not None else original), p
