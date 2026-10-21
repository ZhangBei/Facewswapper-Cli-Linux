#!/usr/bin/env python3
"""Install the Rope CUDA runtime without relying on GUI dependencies."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CUDA_WHEELS = {
    "cu128": {
        "index": "https://download.pytorch.org/whl/cu128",
        "driver": "570+",
    },
    "cu126": {
        "index": "https://download.pytorch.org/whl/cu126",
        "driver": "560+",
    },
    "cu124": {
        "index": "https://download.pytorch.org/whl/cu124",
        "driver": "550+",
    },
}

ONNXRUNTIME_PACKAGES = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-qnn",
)


def run(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install CUDA PyTorch, ONNX Runtime GPU, and the headless Rope "
            "dependencies."
        )
    )
    parser.add_argument(
        "cuda",
        choices=sorted(CUDA_WHEELS),
        help="PyTorch CUDA wheel channel",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="do not install or replace torch and torchvision",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="force reinstall Python packages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    return parser


def detect_conda() -> tuple[bool, str | None]:
    """Return (in_conda_env, env name)."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return False, None
    name = os.environ.get("CONDA_DEFAULT_ENV")
    return True, name or os.path.basename(prefix)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.version_info < (3, 11):
        print(
            f"error: Python {sys.version.split()[0]} is too old; use 3.11+",
            file=sys.stderr,
        )
        return 2

    cuda = CUDA_WHEELS[args.cuda]
    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "requirements.txt"
    if not requirements.is_file():
        print(f"error: requirements file not found: {requirements}", file=sys.stderr)
        return 2

    pip = [sys.executable, "-m", "pip"]
    print(
        f"Rope CUDA installer: {args.cuda} "
        f"(NVIDIA driver {cuda['driver']} or newer)"
    )

    in_conda, conda_env = detect_conda()
    if in_conda:
        print(f"conda environment: {conda_env}")
        print(
            "note: conda owns the native libraries (CUDA runtime, ffmpeg, "
            "libGL);\n      Python packages still come from pip so there is "
            "exactly one onnxruntime build."
        )
    elif args.cuda == "cu128":
        print(
            "note: no conda environment detected. On a system without a "
            "CUDA toolkit,\n      the cu128 pip wheels bundle their own "
            "CUDA runtime libraries."
        )

    if not args.skip_torch and in_conda:
        print(
            "warning: replacing torch inside a conda env can conflict with "
            "conda packages.\n"
            "         If torch was installed from conda-forge, re-run with "
            "--skip-torch."
        )

    # Mixing CPU and GPU ONNX Runtime wheels causes confusing provider
    # errors, so remove every known variant before installing one.
    for package in ONNXRUNTIME_PACKAGES:
        run(
            pip + ["uninstall", "-y", package],
            dry_run=args.dry_run,
        )

    if not args.skip_torch:
        torch_command = pip + [
            "install",
            "torch",
            "torchvision",
            "--index-url",
            cuda["index"],
        ]
        if args.force_reinstall:
            torch_command.insert(4, "--force-reinstall")
        run(torch_command, dry_run=args.dry_run)

    requirements_command = pip + ["install", "-r", str(requirements)]
    if args.force_reinstall:
        requirements_command.insert(4, "--force-reinstall")
    run(requirements_command, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("Verify the environment with:")
        print("  python tools/check_env.py --models ./models")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
