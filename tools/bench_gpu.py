"""GPU 基线验收（docs/gpu-utilization-plan.md 3.4）。

默认 **dry-run**：只打印将要执行的命令，不碰 GPU、不跑换脸、不写文件。
加 ``--run`` 才真正执行。命令构造与结果解析都在
``faceswapper.bench_plan``（stdlib-only，有离线行为测试），本脚本只负责
编排子进程与汇总报告。

    # 先看要跑什么（无 GPU 的机器上也能跑）
    python tools/bench_gpu.py --source face.jpg --target in.mp4

    # 在 GPU 机器上真正执行
    python tools/bench_gpu.py --source face.jpg --target in.mp4 --run \
        --workers 1,2,4 --monitor-gpu --check-frames --psnr

    # 透传额外开关（每个 token 单独给一次）
    python tools/bench_gpu.py --source face.jpg --target in.mp4 \
        --extra-flag=--encode-thread --extra-flag=--encoder \
        --extra-flag=nvenc

退出 0 表示所有用例成功，1 表示至少一条失败。报告写到
``<output-dir>/bench_report.json``。workers=1 是回归基线，其它用例会与它
算加速比并（可选）做 PSNR 比对。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from faceswapper.bench_plan import (  # noqa: E402
    BenchConfig,
    build_cases,
    build_ffprobe_command,
    build_gpu_monitor_command,
    build_psnr_command,
    parse_metrics,
    summarize,
)


def _parse_workers(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--workers must be comma-separated integers, got {text!r}"
        ) from exc


def _print_dry_run(cases, args) -> None:
    print("dry-run: no command will be executed. Add --run to execute.")
    if args.monitor_gpu:
        print("  monitor: " + shlex.join(
            build_gpu_monitor_command(args.monitor_interval)))
    for case in cases:
        print(f"  [{case.label}] " + shlex.join(list(case.command)))
        if args.check_frames:
            print("    verify: " + shlex.join(build_ffprobe_command(case.output)))
    if args.psnr:
        baseline = next((c for c in cases if c.workers == 1), None)
        if baseline:
            for case in cases:
                if case.workers != 1:
                    print("    psnr:   " + shlex.join(
                        build_psnr_command(baseline.output, case.output)))


def _run_probe(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=600,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out: {shlex.join(cmd)}"


def _execute(cases, args, config: BenchConfig) -> int:
    os.makedirs(config.output_dir, exist_ok=True)
    monitor = None
    monitor_log = None
    if args.monitor_gpu:
        log_path = os.path.join(config.output_dir, "gpu-baseline.log")
        monitor_log = open(log_path, "w", encoding="utf-8")
        try:
            monitor = subprocess.Popen(
                build_gpu_monitor_command(args.monitor_interval),
                stdout=monitor_log, stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            print("warn: nvidia-smi not found; skipping GPU sampling")
            monitor_log.close()
            monitor_log = None

    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            print(f"[{case.label}] running ...", flush=True)
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    list(case.command), capture_output=True,
                    text=True, check=False,
                )
                rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            except FileNotFoundError as exc:
                rc, stdout, stderr = 127, "", str(exc)
            wall = round(time.monotonic() - started, 2)

            row: dict[str, Any] = {
                "label": case.label,
                "workers": case.workers,
                "output": case.output,
                "returncode": rc,
                "wall_seconds": wall,
            }
            row.update(parse_metrics(stdout))
            if rc != 0:
                print(f"  failed (rc={rc}); last output:")
                print((stdout + stderr)[-2000:])
            else:
                print(
                    f"  elapsed={row['elapsed']} avg_fps={row['avg_fps']} "
                    f"frames={row['frames']}"
                )
                if args.check_frames:
                    vrc, vout, verr = _run_probe(
                        build_ffprobe_command(case.output))
                    row["ffprobe_returncode"] = vrc
                    try:
                        row["ffprobe"] = json.loads(vout) if vout else None
                    except json.JSONDecodeError:
                        row["ffprobe"] = None
                    if vrc != 0:
                        row["ffprobe_error"] = verr.strip()[-500:]
            rows.append(row)

        if args.psnr:
            baseline = next(
                (r for r in rows
                 if r["workers"] == 1 and r["returncode"] == 0), None,
            )
            if baseline:
                for row in rows:
                    if row["workers"] == 1 or row["returncode"] != 0:
                        continue
                    prc, pout, perr = _run_probe(build_psnr_command(
                        baseline["output"], row["output"]))
                    row["psnr_returncode"] = prc
                    row["psnr_output"] = (perr or pout).strip()[-1000:]
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
        if monitor_log is not None:
            monitor_log.close()

    report = summarize(rows)
    report["config"] = {
        "source": config.source,
        "target": config.target,
        "models": config.models,
        "workers": list(config.workers),
        "preset": config.preset,
        "crf": config.crf,
        "profile": config.profile,
        "extra_flags": list(config.extra_flags),
        "monitor_gpu": bool(args.monitor_gpu and monitor is not None),
        "psnr": args.psnr,
        "check_frames": args.check_frames,
    }
    report_path = os.path.join(config.output_dir, "bench_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    print(f"{'case':<8} {'workers':>8} {'elapsed':>14} {'avg fps':>10} {'speedup':>9} {'rc':>4}")
    for row in report["cases"]:
        print(
            f"{row['label']:<8} {row['workers']:>8} "
            f"{str(row.get('elapsed') or '-'):>14} "
            f"{str(row.get('avg_fps') or '-'):>10} "
            f"{str(row.get('speedup') or '-'):>9} "
            f"{row.get('returncode'):>4}"
        )
    print(f"report: {report_path}")
    return 0 if report["all_succeeded"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GPU 基线矩阵验收；默认 dry-run，--run 才执行",
    )
    parser.add_argument("--source", required=True, help="source face image")
    parser.add_argument("--target", required=True, help="input video")
    parser.add_argument("--models", default="./models")
    parser.add_argument("--output-dir", default="bench-out")
    parser.add_argument("--workers", type=_parse_workers, default=(1, 2, 4))
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--no-profile", action="store_true",
                        help="不采集 decode/swap/encode 分段耗时")
    parser.add_argument("--extra-flag", action="append", default=[],
                        metavar="TOKEN",
                        help="透传给每次换脸命令的额外 token（可重复）")
    parser.add_argument("--python", default="python",
                        help="解释器（默认 python）")
    parser.add_argument("--run", action="store_true",
                        help="真正执行命令（默认只打印）")
    parser.add_argument("--monitor-gpu", action="store_true",
                        help="用 nvidia-smi dmon 采样利用率/显存")
    parser.add_argument("--monitor-interval", type=int, default=1)
    parser.add_argument("--check-frames", action="store_true",
                        help="用 ffprobe -count_frames 核对帧数与时长")
    parser.add_argument("--psnr", action="store_true",
                        help="与 workers=1 基线做逐帧 PSNR 比对")
    args = parser.parse_args(argv)

    config = BenchConfig(
        source=args.source,
        target=args.target,
        models=args.models,
        output_dir=args.output_dir,
        workers=tuple(args.workers),
        preset=args.preset,
        crf=args.crf,
        profile=not args.no_profile,
        extra_flags=tuple(args.extra_flag),
        python=args.python,
    )
    try:
        cases = build_cases(config)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.run:
        _print_dry_run(cases, args)
        return 0
    return _execute(cases, args, config)


if __name__ == "__main__":
    sys.exit(main())
