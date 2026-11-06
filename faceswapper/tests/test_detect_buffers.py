"""Tests for the RetinaFace output-buffer reuse policy.

The subject is stdlib-only, so these are real behaviour tests: no GPU, no
torch, no OpenCV, no ONNX Runtime. ``alloc`` is injected as a fake factory.

    python -m faceswapper.tests.test_detect_buffers
"""

from __future__ import annotations

import sys

from faceswapper.detect_buffers import (
    KINDS,
    STRIDES,
    OutputBufferCache,
    output_buffer_specs,
    resolve_reuse_buffers,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


class FakeBuffer:
    def __init__(self, shape, spec, serial):
        self.shape = shape
        self.spec = spec
        self.serial = serial


def make_alloc():
    calls = []

    def alloc(shape, spec):
        calls.append((shape, spec))
        return FakeBuffer(shape, spec, len(calls))

    return alloc, calls


def test_output_shapes_match_per_frame_path() -> None:
    """shape 与 Models 的逐帧 torch.empty 公式逐字一致，顺序也要对上。"""
    for h, w in ((640, 640), (320, 480), (512, 512)):
        specs = output_buffer_specs(h, w)
        if len(specs) != len(KINDS) * len(STRIDES):
            fail(f"{(h, w)} produced {len(specs)} specs, expected 9")
        # 按 kind 分组，每组内 stride 顺序为 8/16/32。
        for group, kind in enumerate(KINDS):
            chunk = specs[group * 3:(group + 1) * 3]
            if tuple(s.kind for s in chunk) != (kind,) * 3:
                fail(f"{(h, w)} group {group} kinds are wrong: {chunk!r}")
            if tuple(s.stride for s in chunk) != STRIDES:
                fail(f"{(h, w)} group {group} strides are wrong: {chunk!r}")
            for spec in chunk:
                expected_n = (h // spec.stride) * (w // spec.stride) * 2
                width = {"score": 1, "bbox": 4, "kps": 10}[kind]
                if spec.shape != (expected_n, width):
                    fail(f"{(h, w)} {kind}/{spec.stride}: "
                         f"{spec.shape} != {(expected_n, width)}")

    # 抽查一个具体值：640 输入、stride 8 -> 80*80*2 个 anchor。
    first = output_buffer_specs(640, 640)[0]
    if first.shape != (12800, 1):
        fail(f"stride-8 score shape is {first.shape}, expected (12800, 1)")
    print("  ok  output shapes and bind_output ordering match the per-frame path")


def test_invalid_sizes_are_rejected() -> None:
    for h, w in ((0, 640), (640, 0), (-32, 640), (640, -1)):
        try:
            output_buffer_specs(h, w)
        except ValueError:
            continue
        fail(f"output_buffer_specs({h}, {w}) should raise ValueError")
    for bad in ((), (8, 0), (8, -16)):
        try:
            output_buffer_specs(640, 640, bad)
        except ValueError:
            continue
        fail(f"strides {bad!r} should raise ValueError")
    print("  ok  invalid detector sizes and strides are rejected")


def test_resolve_reuse_buffers_gate() -> None:
    """默认关闭；TRT-EP 与未知 backend 不猜、直接关闭。"""
    cases = [
        (False, "cuda", False),
        (False, "trt", False),
        (True, "cuda", True),
        (True, "trt", False),
        (True, "onnx", False),
        (True, "", False),
        (1, "cuda", True),
    ]
    for enabled, backend, expected in cases:
        got = resolve_reuse_buffers(enabled, backend)
        if got is not expected:
            fail(f"resolve_reuse_buffers({enabled!r}, {backend!r}) = "
                 f"{got!r}, expected {expected!r}")
    if resolve_reuse_buffers(False) is not False:
        fail("default backend must still leave the switch off")
    print("  ok  reuse is off by default and never guessed for TRT/unknown backends")


def test_cache_reuses_buffers_for_same_key() -> None:
    alloc, calls = make_alloc()
    cache = OutputBufferCache(alloc)
    first = cache.get(640, 640)
    if len(calls) != 9:
        fail(f"first get allocated {len(calls)} buffers, expected 9")
    for _ in range(5):
        again = cache.get(640, 640)
        if again is not first:
            fail("same key returned a different tuple of buffers")
    if len(calls) != 9:
        fail(f"same key allocated {len(calls)} buffers in total, expected 9")
    if cache.key != (640, 640, STRIDES):
        fail(f"cache key is {cache.key!r}")
    print("  ok  identical detector size allocates once and reuses the buffers")


def test_cache_rebuilds_when_key_changes() -> None:
    alloc, calls = make_alloc()
    cache = OutputBufferCache(alloc)
    first = cache.get(640, 640)
    # DetectInputSizeTextSel 变了：整组 shape 都变，必须重建。
    changed_size = cache.get(320, 320)
    if changed_size is first or len(calls) != 18:
        fail("changing input size did not rebuild the buffer set")
    if any(buf.shape[0] == first[0].shape[0] for buf in changed_size):
        fail("rebuilt buffers still have the old anchor count")
    # 同一尺寸但 stride 组合不同（例如换了 detector 导出）也要重建。
    changed_strides = cache.get(320, 320, (8, 16))
    if changed_strides is changed_size or len(calls) != 24:
        fail("changing strides did not rebuild the buffer set")
    if len(changed_strides) != 6:
        fail(f"2 strides should produce 6 buffers, got {len(changed_strides)}")
    # 回到最初的 key 会再建一组（不做 LRU，刻意只留一组）。
    back = cache.get(640, 640)
    if back is first or len(calls) != 33:
        fail("returning to an earlier key should rebuild, not resurrect stale buffers")
    print("  ok  size/stride changes rebuild the set and never reuse stale buffers")


def test_separate_caches_are_isolated() -> None:
    """两个 cache 实例模拟两个 worker：不能共享同一批缓冲。"""
    alloc_a, calls_a = make_alloc()
    alloc_b, calls_b = make_alloc()
    worker_a = OutputBufferCache(alloc_a)
    worker_b = OutputBufferCache(alloc_b)
    buffers_a = worker_a.get(640, 640)
    buffers_b = worker_b.get(640, 640)
    if len(calls_a) != 9 or len(calls_b) != 9:
        fail("each worker should allocate its own 9 buffers")
    if any(x is y for x, y in zip(buffers_a, buffers_b)):
        fail("two workers received the same buffer object")
    if worker_a.get(640, 640) is not buffers_a:
        fail("worker A lost its cached buffers after worker B ran")
    if len(calls_a) != 9:
        fail("worker B's call caused extra allocations in worker A")
    print("  ok  per-worker caches stay isolated under interleaved calls")


def test_invalid_input_does_not_dirty_cache() -> None:
    alloc, calls = make_alloc()
    cache = OutputBufferCache(alloc)
    good = cache.get(640, 640)
    key = cache.key
    try:
        cache.get(0, 640)
    except ValueError:
        pass
    else:
        fail("cache.get(0, 640) should raise ValueError")
    if cache.key != key or cache.get(640, 640) is not good:
        fail("a rejected size changed the cache state")
    if len(calls) != 9:
        fail("a rejected size triggered allocations")
    print("  ok  rejected sizes leave the cache untouched")


def main() -> int:
    print("faceswapper detect buffer reuse tests")
    test_output_shapes_match_per_frame_path()
    test_invalid_sizes_are_rejected()
    test_resolve_reuse_buffers_gate()
    test_cache_reuses_buffers_for_same_key()
    test_cache_rebuilds_when_key_changes()
    test_separate_caches_are_isolated()
    test_invalid_input_does_not_dirty_cache()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
