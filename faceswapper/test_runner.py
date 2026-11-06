"""离线测试套件的发现、分类与编排。

刻意 stdlib-only：本地没有 GPU 栈，也没有 numpy / torch / cv2 / ORT，
因此判定逻辑必须能脱离这些依赖做行为测试。执行子进程的动作由 ``run``
回调注入，``discover_modules()`` / ``classify()`` 都是纯函数。

    python -m faceswapper.tests              # 跑全部离线模块
    python -m faceswapper.tests --list       # 只列模块，不执行

每个 ``test_*.py`` 都是自带 ``main()`` 的独立脚本，失败时抛
``SystemExit(1)``；用子进程逐个执行，既避免一个模块中断整轮，也避免
模块之间共享 ``sys.modules`` 状态。
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

PASS = "PASS"
SKIP = "SKIP"
FAIL = "FAIL"

ALL_TESTS_PASSED = "all tests passed"

# 只有缺这些顶层包才算「环境性跳过」。其它 ModuleNotFoundError / ImportError
# 一律按真实失败处理，避免把代码里的 import 拼写错误藏成 SKIP。
ENVIRONMENT_MODULES = frozenset({
    "numpy", "torch", "torchvision", "onnxruntime", "onnx",
    "cv2", "av", "PIL", "skimage", "scipy", "torchcodec",
    "tensorrt", "pynvvideocodec", "imageio_ffmpeg",
})

_MISSING_MODULE = re.compile(
    r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)['\"]"
)


@dataclass(frozen=True)
class ModuleResult:
    name: str
    status: str
    returncode: int
    seconds: float
    detail: str = ""
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class SuiteReport:
    results: tuple[ModuleResult, ...]

    def count(self, status: str) -> int:
        return sum(1 for item in self.results if item.status == status)

    def exit_code(self, *, strict: bool = False) -> int:
        """有真实失败即非零；``strict`` 下环境性 SKIP 也算失败。"""
        if self.count(FAIL):
            return 1
        if strict and self.count(SKIP):
            return 1
        return 0


def discover_modules(tests_dir: Path) -> tuple[str, ...]:
    """列出 ``tests_dir`` 下的测试模块名，按文件名排序。

    只收 ``test_*.py``；``__init__.py`` / ``__main__.py`` 不匹配这个模式，
    所以 runner 入口不会把自己当成被测模块。
    """
    return tuple(sorted(
        path.stem for path in Path(tests_dir).glob("test_*.py")
        if path.is_file()
    ))


def module_command(name: str, python: str) -> list[str]:
    return [python, "-m", f"faceswapper.tests.{name}"]


def missing_module(stderr: str) -> str | None:
    """返回 stderr 里报告缺失的顶层包名；没有则返回 None。"""
    match = _MISSING_MODULE.search(stderr or "")
    if match is None:
        return None
    return match.group(1).split(".")[0]


def classify(
    name: str,
    returncode: int,
    stdout: str,
    stderr: str,
    seconds: float,
    *,
    timed_out: bool = False,
) -> ModuleResult:
    """把一次子进程结果归到 PASS / SKIP / FAIL。"""
    stdout = stdout or ""
    stderr = stderr or ""
    if timed_out:
        return ModuleResult(name, FAIL, 124, seconds, "timed out", stdout, stderr)
    if returncode == 0:
        if ALL_TESTS_PASSED in stdout:
            return ModuleResult(name, PASS, returncode, seconds, "", stdout, stderr)
        return ModuleResult(
            name, FAIL, returncode, seconds,
            f"rc=0 but {ALL_TESTS_PASSED!r} is missing", stdout, stderr,
        )

    missing = missing_module(stderr)
    if missing is not None and missing in ENVIRONMENT_MODULES:
        return ModuleResult(
            name, SKIP, returncode, seconds,
            f"missing runtime dependency: {missing}", stdout, stderr,
        )
    tail = (stderr or stdout).strip()[-500:]
    return ModuleResult(name, FAIL, returncode, seconds, tail, stdout, stderr)


def run_module(
    name: str,
    *,
    python: str | None = None,
    cwd: Path,
    timeout: float,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
) -> ModuleResult:
    """执行单个模块并分类；``run`` 可注入，便于离线做行为测试。"""
    runner = run or subprocess.run
    command = module_command(name, python or sys.executable)
    started = time.monotonic()
    timed_out = False
    try:
        proc = runner(
            command, cwd=str(cwd), capture_output=True,
            text=True, check=False, timeout=timeout,
        )
        returncode = int(proc.returncode)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
    except subprocess.TimeoutExpired:
        returncode, stdout, stderr, timed_out = 124, "", "", True
    except FileNotFoundError as exc:
        returncode, stdout, stderr = 127, "", str(exc)
    seconds = round(time.monotonic() - started, 2)
    return classify(
        name, returncode, stdout, stderr, seconds, timed_out=timed_out,
    )


def run_suite(
    names: Sequence[str],
    *,
    python: str | None = None,
    cwd: Path,
    timeout: float,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    on_result: Callable[[ModuleResult], None] | None = None,
) -> SuiteReport:
    """按给定顺序逐个执行，保持顺序与结果一一对应。"""
    results: list[ModuleResult] = []
    for name in names:
        result = run_module(
            name, python=python, cwd=cwd, timeout=timeout, run=run,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return SuiteReport(tuple(results))


def format_result(result: ModuleResult) -> str:
    line = (
        f"  {result.status:<4} {result.name:<30} "
        f"{result.seconds:>6.2f}s"
    )
    if result.detail:
        line += f"  {result.detail}"
    return line


def format_report(report: SuiteReport) -> str:
    lines = [
        "",
        "=== summary ===",
        f"  {len(report.results)} module(s): "
        f"{report.count(PASS)} PASS, {report.count(SKIP)} SKIP, "
        f"{report.count(FAIL)} FAIL",
    ]
    skipped = [item.name for item in report.results if item.status == SKIP]
    failed = [item.name for item in report.results if item.status == FAIL]
    if skipped:
        lines.append("  skipped (environment): " + ", ".join(skipped))
    if failed:
        lines.append("  failed: " + ", ".join(failed))
    return "\n".join(lines)
