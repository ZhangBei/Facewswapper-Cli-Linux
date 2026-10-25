#!/usr/bin/env python3
"""Install the Faceswapper CUDA runtime without GUI dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# channel -> minimum NVIDIA driver major. torch itself always comes from
# pip's own index (PIP_INDEX_URL or PyPI) — no download source is
# hardcoded here. Use --index-url for a CUDA-specific wheel channel.
CUDA_WHEELS = {
    "cu128": 570,
    "cu126": 560,
    "cu124": 550,
}

# "default" is not a channel — it asks for detection.
AUTO = "default"

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


def _capture(command: list[str]) -> tuple[int, str]:
    """Run a command and return (exit code, combined output)."""
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def cuda_channel_for_driver(driver_major: int) -> str | None:
    """Newest wheel channel whose driver floor `driver_major` satisfies."""
    for channel in sorted(CUDA_WHEELS, reverse=True):
        if driver_major >= CUDA_WHEELS[channel]:
            return channel
    return None


def detect_driver_major() -> int | None:
    """NVIDIA driver major version, or None when there is no usable GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    code, out = _capture(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if code != 0 or not out:
        return None
    try:
        return int(out.splitlines()[0].strip().split(".")[0])
    except (ValueError, IndexError):
        return None


def torch_cuda_ready() -> bool:
    """True when torch is importable and can actually see a CUDA device."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def onnxruntime_cuda_ready() -> bool:
    """True when onnxruntime exposes CUDAExecutionProvider."""
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def missing_requirements(requirements: Path) -> list[str]:
    """Distribution names from requirements.txt that are not installed.

    Checks distributions (not import names) so the names match the file
    verbatim — no opencv-python-headless -> cv2 mapping to keep in sync.
    """
    from importlib.metadata import PackageNotFoundError, version

    missing: list[str] = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~;\s]", line, maxsplit=1)[0]
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install CUDA PyTorch, ONNX Runtime GPU, and the headless Faceswapper "
            "dependencies."
        )
    )
    parser.add_argument(
        "cuda",
        nargs="?",
        default=AUTO,
        choices=sorted(CUDA_WHEELS) + [AUTO],
        help="CUDA wheel channel, or 'default' (also the default when "
             "omitted) to pick one from the installed NVIDIA driver and "
             "skip whatever is already working. Without --index-url, torch "
             "comes from pip's own default index",
    )
    parser.add_argument(
        "--index-url",
        default=None,
        help="pip index to install torch from (default: pip's own, i.e. "
             "PIP_INDEX_URL or PyPI)",
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


def resolve_channel(args: argparse.Namespace) -> str:
    """Pick the wheel channel from the args, or from the NVIDIA driver.

    torch itself always installs from pip's own index (PIP_INDEX_URL or
    PyPI) unless --index-url says otherwise, on both paths — no download
    source is hardcoded. An explicit channel is honoured as given.
    """
    if args.cuda != AUTO:
        return args.cuda

    print("auto-detecting the environment...")
    driver = detect_driver_major()
    if driver is None:
        print(
            "error: no NVIDIA driver found (nvidia-smi missing or returned "
            "nothing).\n"
            "       Install the driver first, or name a channel explicitly:\n"
            "         python tools/install.py cu128\n"
            "       CPU-only inference is not supported by this build.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    channel = cuda_channel_for_driver(driver)
    if channel is None:
        floor = min(CUDA_WHEELS.values())
        print(
            f"error: NVIDIA driver {driver} is older than the minimum "
            f"supported (driver {floor}, channel cu124).\n"
            "       Upgrade the driver, then re-run this command.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"  driver: {driver}  ->  channel {channel}")
    if args.index_url:
        print(f"  torch index: {args.index_url}")
    return channel


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.version_info < (3, 11):
        print(
            f"error: Python {sys.version.split()[0]} is too old; use 3.11+",
            file=sys.stderr,
        )
        return 2

    channel = resolve_channel(args)
    min_driver = CUDA_WHEELS[channel]
    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "requirements.txt"
    if not requirements.is_file():
        print(f"error: requirements file not found: {requirements}", file=sys.stderr)
        return 2

    pip = [sys.executable, "-m", "pip"]
    print(
        f"Faceswapper CUDA installer: {channel} "
        f"(NVIDIA driver {min_driver}+)"
    )

    in_conda, conda_env = detect_conda()
    if in_conda:
        print(f"conda environment: {conda_env}")
        print(
            "note: conda owns the native libraries (CUDA runtime, ffmpeg, "
            "libGL);\n      Python packages still come from pip so there is "
            "exactly one onnxruntime build."
        )
    elif channel == "cu128":
        print(
            "note: no conda environment detected. On a system without a "
            "CUDA toolkit,\n      the cu128 pip wheels bundle their own "
            "CUDA runtime libraries."
        )

    # Auto mode skips what already works, so a half-installed environment
    # converges instead of being rebuilt from scratch.
    torch_ok = torch_cuda_ready()
    ort_ok = onnxruntime_cuda_ready()
    if args.cuda == AUTO:
        print(f"  torch:    {'cuda ready' if torch_ok else 'missing or no cuda'}")
        print(f"  onnxruntime: {'cuda provider ready' if ort_ok else 'missing or cpu-only'}")
        missing = missing_requirements(requirements)
        print(
            f"  requirements: {'all present' if not missing else 'missing ' + ', '.join(missing)}"
        )
        if torch_ok and ort_ok and not missing and not args.force_reinstall:
            print()
            print("environment already satisfied; nothing to install.")
            print("verify with: python tools/check_env.py --models ./models")
            return 0

    install_torch = not args.skip_torch and (args.force_reinstall or not torch_ok)
    if not install_torch and not args.skip_torch:
        print("skipping torch install: a working CUDA torch is already present")

    # Only strip existing ONNX Runtime variants when we are about to
    # install one — uninstalling a healthy install is pure downside.
    install_ort = args.force_reinstall or not ort_ok
    if install_ort:
        # Mixing CPU and GPU ONNX Runtime wheels causes confusing provider
        # errors, so remove every known variant before installing one.
        for package in ONNXRUNTIME_PACKAGES:
            run(
                pip + ["uninstall", "-y", package],
                dry_run=args.dry_run,
            )
    else:
        print("skipping onnxruntime uninstall: CUDA provider already present")

    if install_torch:
        if not args.skip_torch and in_conda and not args.force_reinstall:
            print(
                "warning: replacing torch inside a conda env can conflict with "
                "conda packages.\n"
                "         If torch was installed from conda-forge, re-run with "
                "--skip-torch."
            )
        torch_command = pip + ["install", "torch", "torchvision"]
        if args.index_url:
            torch_command += ["--index-url", args.index_url]
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
