"""GPU 常驻解码的探测与帧字段归一（阶段 5b，默认关闭）。

stdlib-only：torchcodec 是否可用由注入的 ``probe`` 决定；frame 的字段提取
只依赖 Python 属性访问，因此这两条逻辑能脱离 GPU 栈做行为测试。

默认 ``--decoder pyav``。显式 ``--decoder gpu`` 且探测通过时，headless 用
torchcodec 的 ``VideoDecoder(..., device="cuda")`` 产出 CUDA tensor，
``VideoManager._swap_video_inner`` 本来就能吃 CUDA tensor，于是省掉
``torch.from_numpy(...).to('cuda')`` 这次上传。

**实验性**：torchcodec 各版本的 frame 字段与 PTS 单位不统一，适配层对拿不到
的字段一律明确报错，不静默猜一个值出来（那会产出坏帧或错时序）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from faceswapper.errors import HeadlessError


@dataclass(frozen=True)
class DecoderPlan:
    """一次运行实际使用的解码后端。"""

    backend: str  # "pyav" | "gpu"
    note: str = ""


def resolve_decoder(
    requested: str,
    probe: Callable[[], tuple[bool, str]],
) -> DecoderPlan:
    """把 ``--decoder`` 解析成 ``DecoderPlan``。

    ``pyav`` 分支**完全不调用** ``probe``，默认路径既不 import torchcodec
    也不碰 CUDA。``gpu`` 探测失败时抛 ``HeadlessError`` 并带上原因，不静默
    回退——否则用户会以为正在走 GPU 常驻路径。
    """
    if requested not in ("pyav", "gpu"):
        raise HeadlessError(
            f"unknown --decoder {requested!r}; expected pyav or gpu"
        )
    if requested == "pyav":
        return DecoderPlan("pyav")

    try:
        available, reason = probe()
    except Exception as exc:
        raise HeadlessError(
            f"GPU decode probe failed: {exc}; use the default --decoder pyav"
        ) from exc
    if not available:
        raise HeadlessError(
            f"GPU decode unavailable: {reason}; "
            "use the default --decoder pyav, or install torchcodec>=0.4 "
            "with a CUDA-capable build"
        )
    return DecoderPlan(
        "gpu",
        note=(
            "torchcodec frame fields / PTS units vary by version; the adapter "
            "raises instead of guessing when a field is missing"
        ),
    )


def frame_fields(raw: Any) -> tuple[Any, Any]:
    """从 torchcodec frame 里取出 ``(data, pts)``，不触碰 tensor 内容。

    兼容 Mapping（dict-like）与带属性的 Frame 对象两种形态。刻意不用
    ``a or b`` 串联——多元素 tensor 的布尔值有歧义，会抛 RuntimeError。
    ``pts`` 允许为 None，由调用方决定是报错还是用兜底步长。
    """
    if isinstance(raw, Mapping):
        data = raw.get("data")
        if data is None:
            data = raw.get("frame")
        pts = raw.get("pts")
        if pts is None:
            pts = raw.get("pts_seconds")
    else:
        data = getattr(raw, "data", None)
        if data is None:
            data = getattr(raw, "frame", None)
        pts = getattr(raw, "pts", None)
        if pts is None:
            pts = getattr(raw, "pts_seconds", None)
    if data is None:
        raise HeadlessError(
            "GPU decoder frame has neither 'data' nor 'frame'; "
            "torchcodec's frame layout differs from what this adapter expects"
        )
    return data, pts
