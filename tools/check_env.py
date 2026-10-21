"""Pre-flight check for the cloud environment.

Verifies everything the CLI needs before you try a real swap:

    python tools/check_env.py --models ./models

Checks, in order:
  1. Python version
  2. NVIDIA driver / nvidia-smi
  3. ffmpeg, ffprobe, and curl
  4. torch + CUDA
  5. onnxruntime providers
  6. the Python packages the CLI imports
  7. model files and SHA-256 hashes

Exits 0 when everything is ready, 1 otherwise. Each failed check prints
what to do about it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Callable


OK = "  ok  "
FAIL = " FAIL "
WARN = " warn "

failures: list[str] = []
warnings: list[str] = []


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def check(
    name: str, fn: Callable[[], str | None], hint: str = ""
) -> None:
    """Run a check. The function returns None on success or an error message."""
    try:
        problem = fn()
    except Exception as exc:  # a crashing check is a failed check
        problem = f"{type(exc).__name__}: {exc}"
    if problem is None:
        print(f"{OK} {name}")
    else:
        print(f"{FAIL} {name}")
        print(f"         {problem}")
        if hint:
            print(f"         -> {hint}")
        failures.append(name)


def warn(name: str, message: str, hint: str = "") -> None:
    print(f"{WARN} {name}")
    print(f"         {message}")
    if hint:
        print(f"         -> {hint}")
    warnings.append(name)


def check_python() -> str | None:
    if sys.version_info < (3, 11):
        return (
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too "
            "old; 3.11 or newer is required."
        )
    return None


def check_driver() -> str | None:
    if not shutil.which("nvidia-smi"):
        return (
            "nvidia-smi not found. Install the NVIDIA driver "
            "(sudo ubuntu-drivers autoinstall) and reboot."
        )
    code, out = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    if code != 0:
        return f"nvidia-smi failed:\n{out}"
    if not out:
        return "nvidia-smi returned no GPU. Is the GPU attached to this instance?"
    for line in out.splitlines():
        print(f"         {line}")
    return None


def check_ffmpeg() -> str | None:
    if not shutil.which("ffmpeg"):
        return "ffmpeg not found. Install it with: sudo apt install -y ffmpeg"
    code, out = _run(["ffmpeg", "-hide_banner", "-encoders"])
    if code != 0:
        return f"ffmpeg failed to run:\n{out}"
    if "libx264" not in out:
        return (
            "this ffmpeg build has no libx264 encoder. The video pipeline "
            "needs it."
        )
    version_code, version_out = _run(["ffmpeg", "-version"])
    if version_code == 0 and version_out:
        print(f"         {version_out.splitlines()[0]}")
    return None


def check_media_tools() -> str | None:
    missing = [
        name for name in ("ffprobe", "curl")
        if shutil.which(name) is None
    ]
    if missing:
        return (
            f"missing system tool(s): {', '.join(missing)}. "
            "Install with: sudo apt install -y ffmpeg curl"
            + (
                " or: conda install -c conda-forge ffmpeg curl"
                if os.environ.get("CONDA_PREFIX") else ""
            )
        )
    return None


def detect_conda() -> str | None:
    """Return the active conda environment name, or None."""
    if not os.environ.get("CONDA_PREFIX"):
        return None
    return os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(
        os.environ["CONDA_PREFIX"]
    )


def check_torch() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return (
            "torch is not installed. Install the CUDA build with:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    import torch

    if not torch.cuda.is_available():
        return (
            f"torch.cuda.is_available() is False (torch {torch.__version__}, "
            f"built for CUDA {torch.version.cuda}).\n"
            "Either the NVIDIA driver is missing/too old, or a CPU-only torch "
            "wheel is installed."
        )
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"         gpu: {name} (sm_{cap[0]}{cap[1]})")
        print(f"         cuda: {torch.version.cuda}")
        # A tiny allocation proves the driver actually works, not just that
        # the driver is present.
        t = torch.zeros(16, device="cuda")
        torch.cuda.synchronize()
        del t
    except Exception as exc:
        return f"CUDA device present but not usable: {exc}"
    return None


def check_onnxruntime() -> str | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return (
            "onnxruntime is not installed. Install it with:\n"
            "  pip install onnxruntime-gpu"
        )
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"         providers: {', '.join(providers)}")
    if "CUDAExecutionProvider" not in providers:
        return (
            "onnxruntime has no CUDAExecutionProvider. You probably installed "
            "onnxruntime (CPU) instead of onnxruntime-gpu."
        )
    if "TensorrtExecutionProvider" in providers:
        print("         (TensorRT provider present; the CLI defaults to CUDA EP)")
    return None


def check_packages() -> str | None:
    required = [
        "cv2", "numpy", "onnx", "skimage", "PIL", "torchvision", "av",
    ]
    missing = [n for n in required if importlib.util.find_spec(n) is None]
    if missing:
        return (
            f"missing Python packages: {', '.join(missing)}. "
            "Install them with: pip install -r requirements.txt"
        )
    # PyAV wraps its own FFmpeg build; a broken wheel shows up as an
    # ImportError on `import av`, not on find_spec, so import it.
    try:
        import av
        _ = av.library_versions
    except Exception as exc:
        return f"PyAV is installed but cannot load its FFmpeg libraries: {exc}"
    return None


def check_models(models_dir: str) -> str | None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from rope.model_store import MODEL_SPECS, inspect_models

    statuses = inspect_models(models_dir, MODEL_SPECS, verify=True)
    invalid_required = [
        spec for spec in MODEL_SPECS
        if spec.required and statuses[spec.name] != "valid"
    ]
    if invalid_required:
        details = ", ".join(
            f"{spec.local_name} ({statuses[spec.name]})"
            for spec in invalid_required
        )
        return f"invalid required model(s) in {models_dir}: {details}"

    for spec in MODEL_SPECS:
        if not spec.required and statuses[spec.name] != "valid":
            warn(
                f"model {spec.local_name}",
                f"{statuses[spec.name]} (optional: {spec.role})",
            )
    for filename, role in (
        ("inswapper_512_level2.onnx", "swapper 512"),
        ("inswapper_256_phase1.onnx", "swapper 256-native"),
        ("GPEN-BFR-256.onnx", "restorer"),
        ("GPEN-BFR-512.onnx", "restorer"),
        ("codeformer_fp16.onnx", "restorer"),
        ("occluder.onnx", "occlusion mask"),
        ("faceparser_resnet34.onnx", "face parser"),
        ("dfl_xseg.onnx", "DFL XSeg mask"),
    ):
        if not os.path.isfile(os.path.join(models_dir, filename)):
            warn(f"model {filename}", f"not present (optional: {role})")
    return None


def check_cli_import() -> str | None:
    """Import the CLI modules the way the real command does."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import rope.cli  # noqa: F401
        import rope.headless  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight check for the Rope headless CLI environment."
    )
    parser.add_argument(
        "--models", default="./models",
        help="folder holding the ONNX model files (default: ./models)",
    )
    args = parser.parse_args()

    print("Rope environment check")
    print(f"  python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"  models: {os.path.abspath(args.models)}")
    conda_env = detect_conda()
    if conda_env:
        print(f"  conda:  {conda_env} (native deps from conda, python from pip)")
    print()

    check("python version", check_python, "use Python 3.11 or newer")
    check(
        "nvidia driver", check_driver,
        "sudo ubuntu-drivers autoinstall && sudo reboot",
    )
    check("ffmpeg", check_ffmpeg, "sudo apt install -y ffmpeg")
    check("media tools", check_media_tools, "sudo apt install -y ffmpeg curl")
    if conda_env:
        print(
            f"         (conda env '{conda_env}': use "
            "'conda install ffmpeg cuda-toolkit' for these)"
        )
    check(
        "torch + cuda", check_torch,
        "pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu128",
    )
    check("onnxruntime", check_onnxruntime, "pip install onnxruntime-gpu")
    check(
        "python packages", check_packages,
        "pip install -r requirements.txt",
    )
    check(
        "model files", lambda: check_models(args.models),
        (
            "run: python -m rope.cli --models "
            f"{args.models} download-models"
        ),
    )
    check("cli imports", check_cli_import, "pip install -r requirements.txt")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    if warnings:
        print(f"all checks passed ({len(warnings)} optional item(s) missing)")
    else:
        print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

