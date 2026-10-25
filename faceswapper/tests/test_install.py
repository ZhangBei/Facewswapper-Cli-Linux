"""Tests for tools/install.py auto-detection.

No GPU, no torch, no pip install — the detection logic is pure functions
over the outputs of nvidia-smi / importlib, so it can be exercised by
stubbing those and reading the source the way test_hyperswap does.

    python -m faceswapper.tests.test_install
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _load_install():
    """Import tools/install.py as a module without a package path."""
    path = REPO_ROOT / "tools" / "install.py"
    spec = importlib.util.spec_from_file_location("faceswapper_install", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = _load_install()


def test_driver_maps_to_channel() -> None:
    """The driver floor is the whole decision: newest channel the driver
    satisfies, and None below the oldest floor."""
    cases = [
        (570, "cu128"),
        (580, "cu128"),
        (560, "cu126"),
        (565, "cu126"),
        (550, "cu124"),
        (555, "cu124"),
        (549, None),
        (0, None),
    ]
    for driver, expected in cases:
        got = install.cuda_channel_for_driver(driver)
        if got != expected:
            fail(f"driver {driver} -> {got!r}, expected {expected!r}")
    print(f"  ok  driver version maps to the newest supported channel")


def test_no_download_source_is_hardcoded() -> None:
    """Same rule as the model downloader: no URL is written into the
    installer. torch comes from pip's own index (PIP_INDEX_URL or PyPI)
    unless --index-url is passed, so a mirror works by configuration
    alone."""
    source = (REPO_ROOT / "tools" / "install.py").read_text(encoding="utf-8")
    for needle in ("download.pytorch.org", "https://", "http://"):
        if needle in source:
            fail(f"install.py hardcodes a download source: {needle!r}")
    # The channel table must be channel -> driver floor, nothing more.
    for channel, floor in install.CUDA_WHEELS.items():
        if not isinstance(floor, int):
            fail(f"{channel} should map to a driver floor int, got {floor!r}")
    print("  ok  no download source is hardcoded; torch rides pip's index")


def test_auto_is_the_default_argument() -> None:
    """`python tools/install.py default` and a bare invocation must both
    take the auto path — the README documents `default` explicitly."""
    parser = install.build_parser()
    if parser.parse_args([]).cuda != install.AUTO:
        fail("omitting the channel should mean auto-detect")
    if parser.parse_args(["default"]).cuda != install.AUTO:
        fail("explicit 'default' should mean auto-detect")
    for channel in install.CUDA_WHEELS:
        if parser.parse_args([channel]).cuda != channel:
            fail(f"explicit {channel} was not honoured")
    # A channel that no longer exists must not be silently accepted.
    try:
        parser.parse_args(["cu999"])
    except SystemExit:
        pass
    else:
        fail("an unknown channel should be rejected")
    print("  ok  auto-detect is the default and explicit channels still work")


def test_detect_handles_missing_driver() -> None:
    """No nvidia-smi must yield None, not a crash or a bogus version."""
    saved = install.shutil.which
    install.shutil.which = lambda name: None
    try:
        if install.detect_driver_major() is not None:
            fail("missing nvidia-smi should report no driver")
    finally:
        install.shutil.which = saved
    print("  ok  missing nvidia-smi is reported, not raised")


def test_readiness_probes_are_import_based() -> None:
    """torch / onnxruntime readiness must come from importlib, never a
    hard import — install.py's whole job is to run before they exist."""
    body = install.torch_cuda_ready.__code__.co_names
    if "find_spec" not in body:
        fail(f"torch_cuda_ready should probe with find_spec: {body}")
    body = install.onnxruntime_cuda_ready.__code__.co_names
    if "find_spec" not in body:
        fail(f"onnxruntime_cuda_ready should probe with find_spec: {body}")

    # A probe must not raise when the package is absent — on this machine
    # torch is absent, which is exactly the failure mode to rule out.
    if install.torch_cuda_ready():
        fail("torch_cuda_ready reported true with no torch installed")
    print("  ok  readiness probes use find_spec and tolerate a missing package")


def test_missing_requirements_parses_the_file() -> None:
    """Names come from requirements.txt verbatim, so a comment or a
    version pin must not become a bogus distribution name."""
    reqs = REPO_ROOT / "requirements.txt"
    names = install.missing_requirements(reqs)
    # Everything in the file is checked as a distribution name; a spec like
    # "av>=13.0,<19.0" must reduce to "av", and the commented-out
    # "torchcodec" must not appear at all.
    for bogus in ("torchcodec", "tensorrt", "imageio-ffmpeg", "pynvvideocodec"):
        if bogus in names:
            fail(f"commented-out requirement {bogus!r} should not be checked")
    for pinned in ("av", "onnxruntime-gpu", "opencv-python-headless"):
        if pinned not in names:
            fail(f"{pinned!r} should be reported missing on a bare machine")
    for name in names:
        if re.search(r"[<>=!~\s#]", name):
            fail(f"distribution name is not clean: {name!r}")
    print(f"  ok  requirements.txt parsed into {len(names)} distribution names")


def test_auto_skips_a_satisfied_environment() -> None:
    """The point of the auto path: when everything already works it must
    not uninstall a healthy onnxruntime just to reinstall it."""
    source = (REPO_ROOT / "tools" / "install.py").read_text(encoding="utf-8")
    if "environment already satisfied" not in source:
        fail("auto mode has no satisfied-environment short circuit")
    if "skipping onnxruntime uninstall" not in source:
        fail("auto mode uninstalls onnxruntime unconditionally")
    if "skipping torch install" not in source:
        fail("auto mode reinstalls torch unconditionally")
    print("  ok  auto mode short-circuits an already-working environment")


def main() -> int:
    print("faceswapper install.py tests")
    test_driver_maps_to_channel()
    test_no_download_source_is_hardcoded()
    test_auto_is_the_default_argument()
    test_detect_handles_missing_driver()
    test_readiness_probes_are_import_based()
    test_missing_requirements_parses_the_file()
    test_auto_skips_a_satisfied_environment()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
