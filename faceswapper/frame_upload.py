"""帧 dtype/布局归一化时「要不要复制整帧」的判定（stdlib-only）。

生产路径上有多处 ``frame.astype("uint8")``：``numpy.astype`` 默认
``copy=True``，哪怕输入已经是 uint8 且 C contiguous 也会**无条件**复制一整帧。
PyAV 的 ``to_ndarray(format="rgb24")`` 和 ``_finalize_frame`` 的 D2H 结果恰好
常常已经满足这两个条件，于是每帧白白多一次全帧 memcpy；1080p RGB 约
6.2 MB，H2D 与 D2H 两侧合计约 12.4 MB/帧，60 秒 30fps 约 22 GB 纯内存带宽。

这里把「已经满足 → 直接复用原数组」抽成不依赖 numpy 的纯判定，并给出可量化
的字节预算：生产代码只消费 ``FramePlan``，真正的
``np.ascontiguousarray(..., dtype=np.uint8)`` 仍在各自模块里，默认路径逐字
保留。术语沿用 CONTEXT.md：中文行文，领域术语保留英文原词。
"""

from __future__ import annotations

from dataclasses import dataclass

FRAME_DTYPE = "uint8"
CHANNELS = 3
BYTES_PER_UINT8 = 1


@dataclass(frozen=True)
class FramePlan:
    """一帧的归一化决策。"""

    needs_copy: bool
    bytes_saved: int
    reason: str


def frame_bytes(height: int, width: int, channels: int = CHANNELS) -> int:
    """一帧 uint8 HWC 图像的字节数，用于估算省下的 memcpy 量。"""
    h, w, c = int(height), int(width), int(channels)
    if h <= 0 or w <= 0 or c <= 0:
        raise ValueError(
            f"frame dimensions must be positive, got {(h, w, c)!r}"
        )
    return h * w * c * BYTES_PER_UINT8


def normalize_plan(
    dtype: str,
    c_contiguous: bool,
    height: int,
    width: int,
    zero_copy: bool = False,
    channels: int = CHANNELS,
) -> FramePlan:
    """决定这一帧是否真的需要被复制/转换。

    - ``zero_copy=False``：闸门关闭，一律走原来的 ``astype`` 路径，不省字节。
    - ``zero_copy=True`` 且 ``dtype`` 已是 ``uint8`` 且 C contiguous：不需要
      任何复制，``torch.from_numpy`` / ``VideoFrame.from_ndarray`` 可以直接
      借用这块内存。
    - ``zero_copy=True`` 但 dtype 或布局不满足：仍然必须复制，这时用
      ``np.ascontiguousarray(frame, dtype=np.uint8)`` 一次完成转换 + 整理，
      与原来的 ``astype`` 数值等价，不省字节。

    注意别名：跳过复制时返回的帧与输入共享内存。调用方只把它交给同步的
    ``.to('cuda')`` / ``from_ndarray``，不会原地写它；PyAV 每帧
    ``to_ndarray`` 都产出新数组，不存在跨帧复用。
    """
    size = frame_bytes(height, width, channels)
    if not zero_copy:
        return FramePlan(True, 0, "gate-closed")
    if str(dtype) != FRAME_DTYPE:
        return FramePlan(True, 0, f"dtype-{dtype}-needs-conversion")
    if not bool(c_contiguous):
        return FramePlan(True, 0, "not-c-contiguous")
    return FramePlan(False, size, "already-uint8-contiguous")


def estimate_saved_bytes(plans) -> int:
    """累计一串 ``FramePlan`` 省下的 memcpy 字节数（可注入生成器）。"""
    return sum(int(plan.bytes_saved) for plan in plans)


def format_saved_bytes(n: int) -> str:
    """把字节数格式化成人类可读的 MB / GB，供诊断输出与文档引用。"""
    value = float(int(n))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"
