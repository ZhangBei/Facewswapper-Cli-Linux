"""视频编码器选择与 NVENC 可用性探测（阶段 5 的编码侧）。

stdlib-only：``codec_factory`` 由调用方注入（真实环境传 ``av.codec.Codec``），
所以「探测 → 选 codec → 映射参数」这条链路能脱离 PyAV / GPU 栈做行为测试。
理由与 ``model_store.py`` / ``encoder.py`` 相同。

默认始终是 ``libx264``；只有显式 ``--encoder nvenc`` **且**探测通过才会切到
``h264_nvenc``。NVENC 是独立硬件单元，不提高 CUDA SM 利用率，只解除 CPU
编码瓶颈；CRF 与 CQ、x264 preset 与 NVENC p1..p7 的语义都不相同，不是等价
替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from faceswapper.errors import HeadlessError


# x264 的速度档位 -> h264_nvenc 的 p1(最快)..p7(最慢)。只是速度倾向的近似
# 对应，压缩效率与画质并不一一等价；p1..p7 需要较新的 FFmpeg/NVENC SDK，
# 旧构建可能只认 fast/medium/slow，真机上若报未知 preset 应回退 libx264。
X264_TO_NVENC_PRESET = {
    "ultrafast": "p1",
    "veryfast": "p2",
    "fast": "p3",
    "medium": "p4",
    "slow": "p6",
    "veryslow": "p7",
}


@dataclass(frozen=True)
class EncoderPlan:
    """一次运行实际使用的编码器与 ``out_stream.options``。"""

    codec: str
    stream_options: dict[str, str]
    note: str = ""


def resolve_encoder(
    requested: str,
    codec_factory: Callable[..., Any],
    *,
    crf: int,
    preset: str,
) -> EncoderPlan:
    """把 ``--encoder`` 解析成 ``EncoderPlan``。

    ``libx264`` 分支**完全不调用** ``codec_factory``，因此默认路径不会因为
    环境里没有 NVENC 而多做任何探测、也不会改变既有行为。

    ``nvenc`` 不可用时明确报错而不是静默回退——静默回退会让人以为正在用
    硬件编码，实际仍是 libx264。
    """
    if requested not in ("libx264", "nvenc"):
        raise HeadlessError(
            f"unknown --encoder {requested!r}; expected libx264 or nvenc"
        )

    if requested == "libx264":
        return EncoderPlan(
            codec="libx264",
            stream_options={"crf": str(int(crf)), "preset": str(preset)},
        )

    try:
        codec_factory("h264_nvenc", "w")
    except Exception as exc:
        raise HeadlessError(
            "h264_nvenc is not available in this PyAV/FFmpeg build; "
            "use the default --encoder libx264, or install an NVENC-capable "
            "FFmpeg/PyAV build plus a driver with NVENC support"
        ) from exc

    return EncoderPlan(
        codec="h264_nvenc",
        stream_options={
            "rc": "vbr",
            "cq": str(int(crf)),
            "preset": X264_TO_NVENC_PRESET.get(str(preset), "p4"),
        },
        note=(
            "NVENC uses CQ (rc=vbr) instead of CRF and p1..p7 instead of "
            "x264 presets; these are not equivalent settings"
        ),
    )
