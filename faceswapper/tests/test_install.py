"""Tests for tools/install.py auto-detection.

不依赖本机是否装了 torch / ONNX Runtime / GPU：环境探测一律用 stub 注入，
因此 bare machine 与装好 CUDA 栈的机器上结论一致。

    python -m faceswapper.tests.test_install
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

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

    # probe 必须在包缺失时不抛异常。注入 find_spec 模拟 bare machine，
    # 不假设当前解释器没装 torch：GPU 机器上 torch.cuda.is_available()
    # 为 True 时，旧断言「本机一定没装 torch」会假失败。
    with mock.patch.object(
        install.importlib.util, "find_spec", return_value=None,
    ):
        if install.torch_cuda_ready():
            fail("torch_cuda_ready reported true with no torch installed")
        if install.onnxruntime_cuda_ready():
            fail("onnxruntime_cuda_ready reported true with no onnxruntime installed")
    print("  ok  readiness probes use find_spec and tolerate a missing package")


def test_missing_requirements_parses_the_file() -> None:
    """注释行、版本 pin 与空白行都不能变成 bogus distribution name。

    解析行为用一份受控的临时 requirements.txt 验证，不再假设本机是 bare
    machine —— 装了 av / onnxruntime-gpu 的 GPU 机器上旧断言会假失败。
    """
    real = install.missing_requirements(REPO_ROOT / "requirements.txt")
    for bogus in ("torchcodec", "tensorrt", "imageio-ffmpeg", "pynvvideocodec"):
        if bogus in real:
            fail(f"commented-out requirement {bogus!r} should not be checked")

    # 三个名字保证未安装，因此结果完全由解析逻辑决定，与本机环境无关。
    expected = {
        "faceswapper-test-absent-a",
        "faceswapper-test-absent-b",
        "faceswapper-test-absent-c",
    }
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "requirements.txt"
        fake.write_text(
            "# torchcodec>=0.4\n"
            "\n"
            "faceswapper-test-absent-a>=1.0\n"
            "faceswapper-test-absent-b<2.0,>=1.5\n"
            "faceswapper-test-absent-c ; python_version >= '3.0'\n",
            encoding="utf-8",
        )
        names = install.missing_requirements(fake)
    if set(names) != expected:
        fail(f"pin/comment parsing produced {names!r}, expected {sorted(expected)!r}")
    for name in names:
        if re.search(r"[<>=!~\s#]", name):
            fail(f"distribution name is not clean: {name!r}")
    print(f"  ok  requirements.txt parsing is environment-independent ({len(names)} names)")


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
