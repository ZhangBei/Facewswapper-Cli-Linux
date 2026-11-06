"""Tests for the GPU baseline plan (command building + metric parsing).

stdlib-only, real behaviour tests: no GPU, no ffmpeg, no models. The runner
script is exercised in dry-run mode, which never executes a swap command.

    python -m faceswapper.tests.test_bench_plan
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from faceswapper.bench_plan import (
    BenchConfig,
    build_cases,
    build_ffprobe_command,
    build_gpu_monitor_command,
    build_psnr_command,
    build_swap_command,
    elapsed_to_seconds,
    parse_metrics,
    summarize,
    validate_config,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def config(**overrides):
    kwargs = {"source": "face.jpg", "target": "in.mp4"}
    kwargs.update(overrides)
    return BenchConfig(**kwargs)


def test_validate_config_rejects_bad_input() -> None:
    bad = [
        config(source=""),
        config(workers=()),
        config(workers=(0,)),
        config(workers=(-1,)),
        config(workers=(9,)),
        config(workers=(2, 2)),
    ]
    for cfg in bad:
        try:
            validate_config(cfg)
        except ValueError:
            continue
        fail(f"invalid config was accepted: {cfg!r}")
    validate_config(config(workers=(1, 8)))
    print("  ok  empty paths / bad worker lists are rejected")


def test_build_cases_are_distinct_and_ordered() -> None:
    cases = build_cases(config(workers=(1, 2, 4)))
    if [c.label for c in cases] != ["w1", "w2", "w4"]:
        fail(f"wrong case labels: {[c.label for c in cases]}")
    if len({c.output for c in cases}) != 3:
        fail("outputs would overwrite each other")
    for case in cases:
        cmd = list(case.command)
        if "--workers" not in cmd or str(case.workers) not in cmd:
            fail(f"missing --workers in {cmd!r}")
        if "--profile" not in cmd:
            fail(f"profile should be on by default: {cmd!r}")
    print("  ok  worker matrix produces distinct, ordered commands")


def test_swap_command_shape_and_extra_flags() -> None:
    cfg = config(profile=False, extra_flags=("--encode-thread", "--encoder", "nvenc"))
    cmd = build_swap_command(cfg, 4, "out.mp4")
    head = cmd[:3]
    if head != ["python", "-m", "faceswapper.cli"]:
        fail(f"unexpected command head: {head!r}")
    for token in ("--source", "face.jpg", "--target", "in.mp4",
                  "--output", "out.mp4", "--workers", "4",
                  "--preset", "ultrafast", "--crf", "23",
                  "--encode-thread", "--encoder", "nvenc"):
        if token not in cmd:
            fail(f"missing {token!r} in {cmd!r}")
    if "--profile" in cmd:
        fail("--no-profile was ignored")
    print("  ok  swap command carries all flags and honours --no-profile")


SAMPLE_OUTPUT = """\
frame 4731/4731 (100.0%)
done: 4731 frame(s) at 24.000 fps, 720x1280, audio=yes -> out-w1.mp4
profile: decode=12.40s swap=131.20s encode=18.30s
elapsed: 00:02:34.6 (avg 30.6 fps)
"""


def test_parse_metrics_extracts_all_fields() -> None:
    got = parse_metrics(SAMPLE_OUTPUT)
    expected = {
        "elapsed": "00:02:34.6",
        "elapsed_seconds": 154.6,
        "avg_fps": 30.6,
        "frames": 4731,
        "source_fps": 24.0,
        "audio": True,
        "decode_s": 12.40,
        "swap_s": 131.20,
        "encode_s": 18.30,
    }
    for key, value in expected.items():
        if got[key] != value:
            fail(f"{key} = {got[key]!r}, expected {value!r}")
    print("  ok  CLI stdout parses into elapsed / fps / frames / segments")


def test_parse_metrics_tolerates_missing_lines() -> None:
    got = parse_metrics("elapsed: 00:00:05.3\n")
    if got["elapsed_seconds"] != 5.3 or got["avg_fps"] is not None:
        fail(f"partial output parsed wrong: {got!r}")
    empty = parse_metrics("no metrics here")
    if any(value is not None for value in empty.values()):
        fail(f"missing metrics must be None: {empty!r}")
    if elapsed_to_seconds("bad") is not None:
        fail("malformed elapsed must return None")
    print("  ok  partial / missing output degrades to None, never raises")


def test_auxiliary_commands() -> None:
    if build_gpu_monitor_command(2) != [
        "nvidia-smi", "dmon", "-s", "u,m", "-d", "2",
    ]:
        fail("wrong nvidia-smi command")
    probe = build_ffprobe_command("out.mp4")
    if "ffprobe" not in probe or "-count_frames" not in probe or "out.mp4" not in probe:
        fail(f"wrong ffprobe command: {probe!r}")
    psnr = build_psnr_command("a.mp4", "b.mp4")
    if "-filter_complex" not in psnr or "psnr" not in psnr:
        fail(f"wrong psnr command: {psnr!r}")
    print("  ok  monitor / ffprobe / psnr commands are well formed")


def test_summarize_speedup_does_not_mutate_rows() -> None:
    rows = [
        {"label": "w1", "workers": 1, "returncode": 0, "elapsed_seconds": 100.0},
        {"label": "w4", "workers": 4, "returncode": 0, "elapsed_seconds": 50.0},
    ]
    original = [dict(r) for r in rows]
    report = summarize(rows)
    if rows != original:
        fail("summarize mutated its input")
    if report["cases"][1]["speedup"] != 2.0:
        fail(f"wrong speedup: {report['cases'][1]!r}")
    if report["baseline_label"] != "w1" or not report["all_succeeded"]:
        fail(f"wrong baseline / status: {report!r}")

    broken = [
        {"label": "w1", "workers": 1, "returncode": 1, "elapsed_seconds": 100.0},
        {"label": "w2", "workers": 2, "returncode": 0, "elapsed_seconds": 50.0},
    ]
    report = summarize(broken)
    if report["baseline_label"] is not None or report["all_succeeded"]:
        fail(f"a failed baseline must not be used: {report!r}")
    if report["cases"][1]["speedup"] is not None:
        fail("no valid baseline means no speedup")
    print("  ok  speedup uses a successful baseline and is mutation-free")


def test_runner_dry_run_is_safe_offline() -> None:
    """默认 dry-run：只打印命令，不执行换脸、不写文件。"""
    script = Path(__file__).resolve().parents[2] / "tools" / "bench_gpu.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--source", "face.jpg",
         "--target", "in.mp4", "--workers", "1,2,4",
         "--monitor-gpu", "--check-frames", "--psnr"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if proc.returncode != 0:
        fail(f"dry-run failed: rc={proc.returncode} stderr={proc.stderr!r}")
    out = proc.stdout
    for expected in ("dry-run", "out-w1.mp4", "out-w4.mp4",
                     "nvidia-smi", "ffprobe", "psnr", "--workers"):
        if expected not in out:
            fail(f"dry-run output missing {expected!r}: {out!r}")
    print("  ok  tools/bench_gpu.py dry-run works without a GPU stack")


def main() -> int:
    print("faceswapper GPU baseline plan tests")
    test_validate_config_rejects_bad_input()
    test_build_cases_are_distinct_and_ordered()
    test_swap_command_shape_and_extra_flags()
    test_parse_metrics_extracts_all_fields()
    test_parse_metrics_tolerates_missing_lines()
    test_auxiliary_commands()
    test_summarize_speedup_does_not_mutate_rows()
    test_runner_dry_run_is_safe_offline()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
