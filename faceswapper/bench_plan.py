"""GPU 基线矩阵的命令构造与结果解析（方案 3.4）。

stdlib-only：本模块**不执行**任何子进程，只构造命令与解析文本，因此能在
没有 GPU、没有 ffmpeg、没有模型的机器上做行为测试。真正的执行在
``tools/bench_gpu.py``（默认 dry-run）。

对应 docs/gpu-utilization-plan.md 3.4：同一段短视频跑 ``workers=1/2/4``，
记录 elapsed / avg fps / 分段耗时、GPU 利用率曲线、VRAM 峰值、ffprobe 帧数，
并用 ffmpeg psnr 比对 workers=1 基线与其它的像素一致性。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchConfig:
    """一次基线矩阵的配置。"""

    source: str
    target: str
    models: str = "./models"
    output_dir: str = "bench-out"
    workers: tuple[int, ...] = (1, 2, 4)
    preset: str = "ultrafast"
    crf: int = 23
    profile: bool = True
    extra_flags: tuple[str, ...] = ()
    python: str = "python"


@dataclass(frozen=True)
class BenchCase:
    """矩阵里的一条用例。"""

    label: str
    workers: int
    output: str
    command: tuple[str, ...]


def validate_config(config: BenchConfig) -> None:
    """输入不合法时尽早报错，避免 dry-run 打印出跑不了的命令。"""
    if not config.source or not config.target:
        raise ValueError("--source and --target are required")
    if not config.workers:
        raise ValueError("at least one worker count is required")
    seen: set[int] = set()
    for workers in config.workers:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if workers > 8:
            raise ValueError(f"CLI clamps --workers to 8, got {workers}")
        if workers in seen:
            raise ValueError(f"duplicate worker count: {workers}")
        seen.add(workers)


def build_swap_command(
    config: BenchConfig, workers: int, output: str
) -> list[str]:
    """构造一条可直接交给 subprocess.run 的换脸命令（不用 shell）。"""
    cmd = [
        config.python, "-m", "faceswapper.cli",
        "--models", config.models,
        "video",
        "--source", config.source,
        "--target", config.target,
        "--output", output,
        "--workers", str(workers),
        "--preset", config.preset,
        "--crf", str(config.crf),
    ]
    if config.profile:
        cmd.append("--profile")
    cmd.extend(config.extra_flags)
    return cmd


def build_cases(config: BenchConfig) -> list[BenchCase]:
    """生成 workers 矩阵；输出文件名按 workers 区分，互不覆盖。"""
    validate_config(config)
    cases = []
    for workers in config.workers:
        label = f"w{workers}"
        output = f"{config.output_dir.rstrip('/')}/out-{label}.mp4"
        cases.append(BenchCase(
            label=label,
            workers=workers,
            output=output,
            command=tuple(build_swap_command(config, workers, output)),
        ))
    return cases


_ELAPSED = re.compile(
    r"elapsed:\s*(\d{2}:\d{2}:\d{2}\.\d)"
    r"(?:\s*\(avg\s+([\d.]+)\s+fps\))?"
)
_DONE = re.compile(
    r"done:\s*(\d+)\s+frame\(s\)\s+at\s+([\d.]+)\s+fps"
    r".*?audio=(yes|no)"
)
_PROFILE = re.compile(
    r"decode=([\d.]+)s\s+swap=([\d.]+)s\s+encode=([\d.]+)s"
)


def elapsed_to_seconds(text: str) -> float | None:
    """``HH:MM:SS.s`` -> 秒。格式不符返回 None，不抛异常。"""
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def parse_metrics(stdout: str) -> dict[str, Any]:
    """从一次 CLI 运行的 stdout 里抽取指标；缺失字段为 None。"""
    result: dict[str, Any] = {
        "elapsed": None,
        "elapsed_seconds": None,
        "avg_fps": None,
        "frames": None,
        "source_fps": None,
        "audio": None,
        "decode_s": None,
        "swap_s": None,
        "encode_s": None,
    }
    match = _ELAPSED.search(stdout)
    if match:
        result["elapsed"] = match.group(1)
        result["elapsed_seconds"] = elapsed_to_seconds(match.group(1))
        if match.group(2):
            result["avg_fps"] = float(match.group(2))
    match = _DONE.search(stdout)
    if match:
        result["frames"] = int(match.group(1))
        result["source_fps"] = float(match.group(2))
        result["audio"] = match.group(3) == "yes"
    match = _PROFILE.search(stdout)
    if match:
        result["decode_s"] = float(match.group(1))
        result["swap_s"] = float(match.group(2))
        result["encode_s"] = float(match.group(3))
    return result


def build_gpu_monitor_command(interval: int = 1) -> list[str]:
    """nvidia-smi dmon：利用率与显存曲线，重定向由调用方处理。

    ``-s`` 的参数是连续的字母串（``um`` = utilization + memory），写成
    ``u,m`` 会被 dmon 判为「Failed to find any metric to display」并直接退出。
    """
    return ["nvidia-smi", "dmon", "-s", "um", "-d", str(max(1, interval))]


def build_ffprobe_command(path: str) -> list[str]:
    """数帧与时长，用于确认输出没有丢帧 / 时长漂移。"""
    return [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,duration,r_frame_rate",
        "-of", "json", path,
    ]


def build_psnr_command(baseline: str, candidate: str) -> list[str]:
    """比对 workers=1 基线与其它的逐帧 PSNR（输出为纯文本，不落文件）。

    PSNR 统计行是 ffmpeg 的 info 级日志，用 ``-loglevel error`` 会把它整个
    吞掉，只剩返回码 0 却没有数值；``-nostats`` 压掉进度噪声，让统计行留在
    stderr 末尾便于解析。
    """
    return [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
        "-i", baseline, "-i", candidate,
        "-filter_complex", "psnr", "-f", "null", "-",
    ]


def parse_psnr(text: str) -> str | None:
    """从 ffmpeg psnr 输出里取 average 值；逐像素一致时是 ``inf``。"""
    match = re.search(r"\bPSNR\b[^\n]*?\baverage:([^\s]+)", text or "")
    return match.group(1) if match else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总：以 workers=1 为基线算加速比，不修改传入的 rows。"""
    baseline = next(
        (r for r in rows if r.get("workers") == 1 and r.get("returncode") == 0),
        None,
    )
    base_seconds = baseline.get("elapsed_seconds") if baseline else None
    cases = []
    for row in rows:
        item = dict(row)
        seconds = item.get("elapsed_seconds")
        item["speedup"] = (
            round(base_seconds / seconds, 3)
            if base_seconds and seconds else None
        )
        cases.append(item)
    return {
        "baseline_label": baseline.get("label") if baseline else None,
        "cases": cases,
        "all_succeeded": all(r.get("returncode") == 0 for r in rows),
    }
