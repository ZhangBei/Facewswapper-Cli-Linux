"""视频编码线程：把 libx264 编码从主帧循环里挪到独立线程。

本模块刻意只依赖标准库，PyAV 对象通过 ``av`` 参数注入——这样队列保序、
sentinel 收尾和异常传播都能在没有 GPU 栈、没有 PyAV 的机器上做离线测试。
理由与 ``model_store.py`` 相同：把可独立验证的逻辑从重运行时里剥出来。

``--encode-thread`` 默认关闭。关闭时 ``headless._encode_video`` 走原来的
主线程编码路径，一行都不经过这里，输出与基线逐字一致。
"""

from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction
from typing import Any, Callable

from faceswapper.errors import HeadlessError


class EncoderFailure(HeadlessError):
    """编码线程出错，由主线程重抛；保留原始异常供排查。"""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"video encoder thread failed: {cause}")
        self.cause = cause


def submit_frames(
    items: "queue.Queue",
    item: Any,
    stop_event: threading.Event,
    errors: list[BaseException],
    timeout: float = 1.0,
) -> None:
    """带超时地把一帧（或 sentinel）投进有界队列。

    队列满时重试而不是死等：编码线程一旦失败会设置 ``stop_event`` 并把
    异常写进 ``errors``，下一轮循环就能立刻抛错，不会永久阻塞在 ``put``。
    """
    while True:
        if errors:
            raise EncoderFailure(errors[0])
        if stop_event.is_set():
            raise EncoderFailure(
                errors[0] if errors else RuntimeError("encoder stopped")
            )
        try:
            items.put(item, timeout=timeout)
            return
        except queue.Full:
            continue


def encoder_worker(
    av: Any,
    silent_path: str,
    *,
    width: int,
    height: int,
    fps: float,
    in_tb: Fraction,
    pix_fmt: str,
    crf: int,
    preset: str,
    items: "queue.Queue",
    stop_event: threading.Event,
    errors: list[BaseException],
    on_progress: Callable[[int, int], None] | None,
    total: int,
    on_encoded: Callable[[], None] | None = None,
    add_time: Callable[[str, float], None] | None = None,
    codec: str = "libx264",
    stream_options: dict[str, str] | None = None,
) -> None:
    """编码线程主体：按队列的 FIFO 顺序编码，遇到 ``None`` 收尾。

    ``av`` 相关对象全部在本线程创建、使用、销毁，绝不跨线程传递。队列项
    形如 ``(display_index, rgb, pts)``：``rgb`` 已在主线程转成 contiguous
    ndarray；``pts`` 是源相对时间戳；``pts_scale`` 用本线程的
    ``out_stream.time_base`` 计算，公式与串行路径一致。

    ``codec`` / ``stream_options`` 来自 ``encoder_probe.resolve_encoder()``；
    不传时保持原来的 libx264 + CRF/preset，因此既有离线测试不受影响。
    """
    output = None
    try:
        output = av.open(
            silent_path, mode="w", format="mp4",
            options={"movflags": "+faststart"},
        )
        out_stream = output.add_stream(
            codec, rate=Fraction(fps).limit_denominator(1000000)
        )
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = pix_fmt
        # 复制一份：PyAV / FFmpeg 可能会改这个 dict，不该污染调用方的 plan。
        out_stream.options = (
            dict(stream_options) if stream_options is not None
            else {"crf": str(int(crf)), "preset": str(preset)}
        )
        out_stream.time_base = in_tb
        out_tb = out_stream.time_base or in_tb
        pts_scale = float(in_tb / out_tb)

        while True:
            try:
                item = items.get(timeout=1.0)
            except queue.Empty:
                # 主线程异常退出时不会投 sentinel，靠 stop_event 收尾。
                if stop_event.is_set():
                    break
                continue
            if item is None:
                # FIFO 保证 sentinel 之前的帧都已编码完，这里再 flush 编码器。
                for packet in out_stream.encode(None):
                    output.mux(packet)
                break

            display_index, rgb, pts = item
            t0 = time.monotonic() if add_time is not None else 0.0
            new_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            new_frame.pts = int(round(pts * pts_scale))
            new_frame.time_base = out_tb
            for packet in out_stream.encode(new_frame):
                output.mux(packet)
            if add_time is not None:
                add_time("encode", time.monotonic() - t0)
            if on_encoded is not None:
                on_encoded()
            # 进度回调移到编码线程按序触发，语义与串行路径一致。
            if on_progress is not None and (
                display_index % 10 == 0 or display_index == total
            ):
                on_progress(display_index, total or display_index)
    except BaseException as exc:  # 线程里的任何异常都要交回主线程重抛
        errors.append(exc)
        stop_event.set()
    finally:
        if output is not None:
            try:
                output.close()
            except BaseException as exc:
                errors.append(exc)
                stop_event.set()
