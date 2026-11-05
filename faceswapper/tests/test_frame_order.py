"""Tests for the ordered parallel frame scheduler.

These exercise the real scheduler with a fake ``swap_fn``: no GPU, no ONNX
models, no torch, no OpenCV, no PyAV. The point is to prove the hard contract
— workers may finish out of order, but the consumer always sees decode order —
rather than checking that certain strings appear in a source file.

    python -m faceswapper.tests.test_frame_order
"""

from __future__ import annotations

import sys
import threading
import time

from faceswapper.frame_order import ordered_swap_frames


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def frames(n: int):
    """产出 (index, rgb, pts)；rgb 用 index 本身，方便验证结果不串味。"""
    return [(i, i, i * 1000) for i in range(n)]


def test_output_order_with_shuffled_completion() -> None:
    """完成顺序真的被打乱时，输出仍严格 0..N-1，且每帧拿到自己的结果。"""
    n = 9
    completions: list[int] = []
    lock = threading.Lock()

    def slow_swap(rgb):
        # 小 index 睡得更久，制造明显的乱序完成。
        time.sleep(0.002 * (n - rgb))
        with lock:
            completions.append(rgb)
        return ("swapped", rgb)

    got = list(ordered_swap_frames(iter(frames(n)), 4, 1, slow_swap))
    expected = [(i, ("swapped", i), i * 1000) for i in range(n)]

    if sorted(completions) != list(range(n)):
        fail(f"not every frame was swapped: {completions!r}")
    if completions == list(range(n)):
        fail("completion order was not shuffled; this test would be vacuous")
    if got != expected:
        fail(f"output lost decode order or mixed up frame results: {got!r}")
    print("  ok  shuffled completions still yield frames in decode order")


def test_every_nth_passes_through_untouched() -> None:
    """--every-nth 语义：只换第 N 帧，其余原样透传，PTS 不变。"""
    calls: list[int] = []
    lock = threading.Lock()

    def swap(rgb):
        with lock:
            calls.append(rgb)
        return ("swapped", rgb)

    got = list(ordered_swap_frames(iter(frames(8)), 2, 3, swap))
    if [i for i, _, _ in got] != list(range(8)):
        fail("every-nth changed the output order")
    if sorted(calls) != [0, 3, 6]:
        fail(f"wrong frames were submitted to swap_fn: {sorted(calls)!r}")
    for index, rgb, pts in got:
        expected_rgb = ("swapped", index) if index % 3 == 0 else index
        if rgb != expected_rgb or pts != index * 1000:
            fail(f"frame {index} was corrupted: rgb={rgb!r} pts={pts!r}")
    print("  ok  every-nth skips frames without touching their rgb or PTS")


def test_worker_counts_produce_identical_output() -> None:
    """workers=1/2/4/8 的输出逐项一致（并发度不改变结果）。"""
    def swap(rgb):
        return ("s", rgb)

    baseline = list(ordered_swap_frames(iter(frames(12)), 1, 1, swap))
    for workers in (2, 4, 8):
        got = list(ordered_swap_frames(iter(frames(12)), workers, 1, swap))
        if got != baseline:
            fail(f"workers={workers} produced different output")
    print("  ok  worker count does not change the ordered result")


def test_swap_exception_propagates() -> None:
    """swap_fn 抛错时异常如实传播，不能被 future 悄悄吞掉。"""
    def swap(rgb):
        if rgb == 3:
            raise RuntimeError("boom")
        return rgb

    try:
        list(ordered_swap_frames(iter(frames(10)), 2, 1, swap))
    except RuntimeError as exc:
        if "boom" not in str(exc):
            fail(f"wrong exception surfaced: {exc!r}")
    else:
        fail("a swap_fn exception was swallowed")
    print("  ok  swap failures propagate through the ordered window")


def test_empty_input_and_degenerate_args() -> None:
    """空输入返回空；workers/step 的非法小值不会让线程池炸掉。"""
    if list(ordered_swap_frames(iter([]), 4, 1, lambda x: x)) != []:
        fail("empty input should yield nothing")
    got = list(ordered_swap_frames(iter(frames(3)), 0, 0, lambda x: ("s", x)))
    if [i for i, _, _ in got] != [0, 1, 2]:
        fail("degenerate workers/step broke the output")
    print("  ok  empty input and clamped workers/step are safe")


def test_generator_close_does_not_hang() -> None:
    """消费者中途退出时，线程池仍能收尾，不会挂死进程。"""
    def swap(rgb):
        time.sleep(0.01)
        return rgb

    gen = ordered_swap_frames(iter(frames(20)), 2, 1, swap)
    if next(gen)[0] != 0:
        fail("first yielded frame is not index 0")
    started = time.monotonic()
    gen.close()
    if time.monotonic() - started > 3.0:
        fail("closing the generator blocked on outstanding futures")
    print("  ok  closing the generator drains the pool without hanging")


def main() -> int:
    print("faceswapper ordered frame scheduler tests")
    test_output_order_with_shuffled_completion()
    test_every_nth_passes_through_untouched()
    test_worker_counts_produce_identical_output()
    test_swap_exception_propagates()
    test_empty_input_and_degenerate_args()
    test_generator_close_does_not_hang()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
