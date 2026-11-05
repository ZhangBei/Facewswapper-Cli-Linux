"""Tests for the standalone video encoder thread.

These run without a GPU, without ONNX models, without PyAV and without numpy:
``faceswapper.encoder`` is stdlib-only and takes the ``av`` module as an
argument, so a small fake container is enough to exercise the real queue
ordering, sentinel flush and error propagation.

    python -m faceswapper.tests.test_encoder
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from fractions import Fraction

from faceswapper.encoder import (
    EncoderFailure,
    encoder_worker,
    submit_frames,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


class _FakeFrame:
    def __init__(self, rgb):
        self.rgb = rgb
        self.pts = None
        self.time_base = None


class _FakeVideoFrame:
    def __init__(self, owner):
        self._owner = owner

    def from_ndarray(self, rgb, format=None):
        frame = _FakeFrame(rgb)
        self._owner.frames.append(frame)
        return frame


class _FakeStream:
    def __init__(self, log, fail_on=None):
        self._log = log
        self._fail_on = fail_on
        self.time_base = Fraction(1, 90000)
        self.width = None
        self.height = None
        self.pix_fmt = None
        self.options = None

    def encode(self, frame):
        if frame is None:
            self._log.append("flush")
            return []
        if self._fail_on is not None and frame.rgb == self._fail_on:
            raise RuntimeError("simulated encoder failure")
        self._log.append(frame.rgb)
        return []


class _FakeOutput:
    def __init__(self, log, fail_on=None):
        self._log = log
        self.stream = _FakeStream(log, fail_on)
        self.closed = False

    def add_stream(self, codec, rate=None):
        self._log.append(("add_stream", codec, rate))
        return self.stream

    def mux(self, packet):
        self._log.append(("mux", packet))

    def close(self):
        self.closed = True
        self._log.append("close")


class _FakeAV:
    def __init__(self, fail_on=None):
        self.log: list = []
        self.frames: list[_FakeFrame] = []
        self.output = _FakeOutput(self.log, fail_on)
        self.VideoFrame = _FakeVideoFrame(self)

    def open(self, path, mode=None, format=None, options=None):
        self.log.append(("open", path))
        return self.output


def _worker(fake, items, stop, errors, **extra):
    """用一组固定参数启动（在调用线程里同步跑完）编码 worker。"""
    kwargs = {
        "av": fake,
        "silent_path": "out.mp4",
        "width": 720,
        "height": 1280,
        "fps": 24.0,
        "in_tb": Fraction(1, 90000),
        "pix_fmt": "yuv420p",
        "crf": 18,
        "preset": "medium",
        "items": items,
        "stop_event": stop,
        "errors": errors,
        "on_progress": None,
        "total": 7,
    }
    kwargs.update(extra)
    encoder_worker(**kwargs)


def test_encoder_preserves_queue_order() -> None:
    """7 帧 + sentinel：编码顺序必须是 0..6，且 flush/close 都发生。"""
    fake = _FakeAV()
    items: queue.Queue = queue.Queue()
    stop = threading.Event()
    errors: list[BaseException] = []
    progress: list[tuple[int, int]] = []
    encoded: list[int] = []
    timed: list[tuple[str, float]] = []

    for i in range(7):
        items.put((i + 1, i, i * 1000))
    items.put(None)

    _worker(
        fake, items, stop, errors,
        on_progress=lambda idx, total: progress.append((idx, total)),
        on_encoded=lambda: encoded.append(1),
        add_time=lambda key, sec: timed.append((key, sec)),
    )

    order = [x for x in fake.log if isinstance(x, int)]
    if order != list(range(7)):
        fail(f"frames encoded out of order: {order!r}")
    if "flush" not in fake.log or "close" not in fake.log:
        fail(f"missing flush/close: {fake.log!r}")
    if not fake.output.closed:
        fail("output container was not closed")
    if len(encoded) != 7:
        fail(f"on_encoded called {len(encoded)} times, expected 7")
    if progress != [(7, 7)]:
        fail(f"progress callback out of order: {progress!r}")
    if [key for key, _ in timed] != ["encode"] * 7:
        fail(f"encode segment timing not recorded: {timed!r}")
    if errors:
        fail(f"unexpected encoder errors: {errors!r}")
    if [f.pts for f in fake.frames] != [i * 1000 for i in range(7)]:
        fail("PTS values were not preserved by the encoder thread")
    if any(f.time_base != Fraction(1, 90000) for f in fake.frames):
        fail("frame time_base does not match the source time base")
    print("  ok  encoder drains the queue in FIFO order with flush and close")


def test_encoder_reports_failure() -> None:
    """编码中途抛错：异常回传、stop_event 置位、容器仍然关闭。"""
    fake = _FakeAV(fail_on=2)
    items: queue.Queue = queue.Queue()
    stop = threading.Event()
    errors: list[BaseException] = []
    for i in range(5):
        items.put((i + 1, i, i * 1000))
    items.put(None)

    _worker(fake, items, stop, errors)

    if not errors or not isinstance(errors[0], RuntimeError):
        fail(f"encoder exception was not handed back: {errors!r}")
    if not stop.is_set():
        fail("stop_event was not set after an encoder failure")
    if not fake.output.closed:
        fail("output container was not closed on the failure path")
    print("  ok  encoder failures propagate and still close the container")


def test_encoder_accepts_alternative_codec() -> None:
    """阶段 5：codec / stream_options 由 encoder_probe 传入，worker 原样使用。"""
    fake = _FakeAV()
    items: queue.Queue = queue.Queue()
    stop = threading.Event()
    errors: list[BaseException] = []
    items.put((1, 0, 0))
    items.put(None)
    opts = {"rc": "vbr", "cq": "23", "preset": "p3"}

    _worker(
        fake, items, stop, errors,
        codec="h264_nvenc", stream_options=opts,
    )

    if not any(
        isinstance(entry, tuple) and entry[0] == "add_stream"
        and entry[1] == "h264_nvenc"
        for entry in fake.log
    ):
        fail(f"alternative codec was not used: {fake.log!r}")
    if fake.output.stream.options != opts:
        fail(f"stream options were not applied: {fake.output.stream.options!r}")
    if fake.output.stream.options is opts:
        fail("worker must copy stream_options, not hand PyAV the caller's dict")
    if errors:
        fail(f"unexpected encoder errors: {errors!r}")
    print("  ok  worker honours the probed codec and copies stream options")


def test_submit_frames_retries_a_full_queue() -> None:
    """有界队列满时重试，腾出空间后成功投递，不永久阻塞。"""
    items: queue.Queue = queue.Queue(maxsize=1)
    items.put("occupied")
    stop = threading.Event()
    errors: list[BaseException] = []

    def drain() -> None:
        time.sleep(0.05)
        items.get()

    helper = threading.Thread(target=drain)
    helper.start()
    submit_frames(items, "new", stop, errors, timeout=0.01)
    helper.join(timeout=2.0)
    if helper.is_alive():
        fail("drain helper did not finish")
    if items.get(timeout=1.0) != "new":
        fail("submit_frames did not enqueue after the queue freed up")
    print("  ok  submit_frames retries instead of blocking on a full queue")


def test_submit_frames_stops_on_error() -> None:
    """编码线程已报错或要求停止时，主线程立刻抛错而不是继续投递。"""
    stop = threading.Event()
    stop.set()
    try:
        submit_frames(queue.Queue(), "x", stop, [])
    except EncoderFailure:
        pass
    else:
        fail("submit_frames must raise when stop_event is set")

    errors: list[BaseException] = [RuntimeError("boom")]
    try:
        submit_frames(queue.Queue(), "x", threading.Event(), errors)
    except EncoderFailure as exc:
        if exc.cause is not errors[0]:
            fail("EncoderFailure lost its original cause")
    else:
        fail("submit_frames must raise when the encoder already failed")
    print("  ok  submit_frames aborts promptly on stop or encoder error")


def test_stop_event_exits_without_sentinel() -> None:
    """主线程异常退出时不投 sentinel，编码线程靠 stop_event 自行收尾。"""
    fake = _FakeAV()
    items: queue.Queue = queue.Queue()
    stop = threading.Event()
    stop.set()
    errors: list[BaseException] = []

    started = time.monotonic()
    _worker(fake, items, stop, errors)
    elapsed = time.monotonic() - started

    if elapsed > 3.0:
        fail(f"encoder did not exit promptly without a sentinel: {elapsed:.1f}s")
    if not fake.output.closed:
        fail("output container was not closed on the stop path")
    if errors:
        fail(f"stop path should not report errors: {errors!r}")
    print("  ok  encoder exits on stop_event when no sentinel arrives")


def main() -> int:
    print("faceswapper encoder thread tests")
    test_encoder_preserves_queue_order()
    test_encoder_reports_failure()
    test_encoder_accepts_alternative_codec()
    test_submit_frames_retries_a_full_queue()
    test_submit_frames_stops_on_error()
    test_stop_event_exits_without_sentinel()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
