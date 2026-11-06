"""Tests for the offline suite runner.

``faceswapper/test_runner.py`` is stdlib-only and takes an injected ``run``
callback, so discovery, classification, ordering and exit codes are all real
behaviour tests. No GPU, no models, no numpy / torch / ORT.

    python -m faceswapper.tests.test_suite_runner
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from faceswapper.test_runner import (
    ALL_TESTS_PASSED,
    FAIL,
    PASS,
    SKIP,
    SuiteReport,
    classify,
    discover_modules,
    format_report,
    missing_module,
    module_command,
    run_module,
    run_suite,
)

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_discover_modules_covers_the_real_suite() -> None:
    """发现集合必须恰好等于目录里的 test_*.py，不能漏掉任何模块。"""
    discovered = discover_modules(TESTS_DIR)
    expected = tuple(sorted(p.stem for p in TESTS_DIR.glob("test_*.py")))
    if discovered != expected:
        fail(f"discovery mismatch: {discovered!r} vs {expected!r}")
    if not discovered:
        fail("no test modules discovered")
    if "__main__" in discovered or "__init__" in discovered:
        fail("runner entry points must not be treated as test modules")
    for name in ("test_cli", "test_frame_upload", "test_suite_runner"):
        if name not in discovered:
            fail(f"{name} is missing from the suite")
    print(f"  ok  all {len(discovered)} offline modules are discovered")


def test_discover_modules_filters_and_sorts() -> None:
    """只收 test_*.py，按名字排序；helper / __main__ / 非 py 都不收。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("test_b.py", "test_a.py", "helper.py", "__init__.py",
                     "__main__.py", "test_c.txt", "notes.md"):
            (root / name).write_text("", encoding="utf-8")
        (root / "test_dir.py").mkdir()
        if discover_modules(root) != ("test_a", "test_b"):
            fail(f"unexpected discovery: {discover_modules(root)!r}")
    print("  ok  discovery filters non-modules and sorts by name")


def test_module_command_shape() -> None:
    cmd = module_command("test_cli", "python3")
    if cmd != ["python3", "-m", "faceswapper.tests.test_cli"]:
        fail(f"unexpected command: {cmd!r}")
    print("  ok  each module runs as its own -m subprocess")


def test_missing_module_extracts_top_level_package() -> None:
    cases = {
        "ModuleNotFoundError: No module named 'numpy'": "numpy",
        'ImportError: No module named "numpy.core._multiarray_umath"': "numpy",
        "ModuleNotFoundError: No module named 'torchcodec.decoders'": "torchcodec",
        "AssertionError: something else": None,
        "": None,
    }
    for stderr, expected in cases.items():
        if missing_module(stderr) != expected:
            fail(f"missing_module({stderr!r}) != {expected!r}")
    print("  ok  missing package names are extracted without guessing")


def test_classify_pass_requires_the_completion_marker() -> None:
    ok = classify("test_a", 0, f"some output\n{ALL_TESTS_PASSED}\n", "", 0.3)
    if ok.status != PASS or ok.returncode != 0:
        fail(f"a successful module was not PASS: {ok!r}")

    silent = classify("test_a", 0, "no marker here", "", 0.3)
    if silent.status != FAIL:
        fail("rc=0 without the completion marker must not be a silent pass")
    print("  ok  rc=0 only counts as PASS with the completion marker")


def test_classify_skips_only_known_runtime_dependencies() -> None:
    missing_numpy = (
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'numpy'\n"
    )
    skipped = classify("test_target_match", 1, "", missing_numpy, 0.1)
    if skipped.status != SKIP or "numpy" not in skipped.detail:
        fail(f"missing numpy should be SKIP: {skipped!r}")

    typo = (
        "ModuleNotFoundError: No module named 'numpyy'\n"
    )
    if classify("test_a", 1, "", typo, 0.1).status != FAIL:
        fail("a misspelled import must be FAIL, not hidden as SKIP")

    own_module = (
        "ModuleNotFoundError: No module named 'faceswapper.foo'\n"
    )
    if classify("test_a", 1, "", own_module, 0.1).status != FAIL:
        fail("a missing project module must be FAIL")

    assertion = "AssertionError: expected 3\n"
    if classify("test_a", 1, "", assertion, 0.1).status != FAIL:
        fail("a real assertion failure must be FAIL")
    print("  ok  only whitelisted runtime deps are SKIP; everything else FAIL")


def test_classify_timeout_and_failure_details() -> None:
    timed_out = classify("test_a", -9, "", "", 1.0, timed_out=True)
    if timed_out.status != FAIL or timed_out.returncode != 124:
        fail(f"timeout must be FAIL with rc=124: {timed_out!r}")
    broken = classify("test_a", 1, "", "boom\n" * 20, 0.2)
    if broken.status != FAIL or len(broken.detail) > 500:
        fail("failure detail must be a bounded tail of the output")
    print("  ok  timeouts are FAIL and failure details are bounded")


def test_run_module_uses_injected_runner() -> None:
    calls: list[dict] = []

    def fake(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return completed(0, f"header\n{ALL_TESTS_PASSED}\n", "")

    result = run_module(
        "test_cli", python="python3", cwd=REPO_ROOT, timeout=7.0, run=fake,
    )
    if result.status != PASS:
        fail(f"injected runner result was not PASS: {result!r}")
    if len(calls) != 1:
        fail("exactly one subprocess should be started")
    call = calls[0]
    if call["cmd"] != ["python3", "-m", "faceswapper.tests.test_cli"]:
        fail(f"wrong command: {call['cmd']!r}")
    if call["kwargs"]["cwd"] != str(REPO_ROOT):
        fail("subprocess cwd must be the repo root")
    if call["kwargs"]["capture_output"] is not True or call["kwargs"]["text"] is not True:
        fail("output must be captured as text")
    if call["kwargs"]["timeout"] != 7.0:
        fail("timeout was not forwarded")
    print("  ok  run_module builds a standalone -m command with a timeout")


def test_run_module_handles_timeout_and_missing_python() -> None:
    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    timed_out = run_module(
        "test_a", cwd=REPO_ROOT, timeout=1.0, run=slow,
    )
    if timed_out.status != FAIL or timed_out.returncode != 124:
        fail(f"TimeoutExpired was not classified as FAIL: {timed_out!r}")

    def absent(cmd, **kwargs):
        raise FileNotFoundError("no such interpreter")

    missing = run_module(
        "test_a", cwd=REPO_ROOT, timeout=1.0, run=absent,
    )
    if missing.status != FAIL or missing.returncode != 127:
        fail(f"FileNotFoundError was not classified as FAIL: {missing!r}")
    print("  ok  timeout and missing interpreter are explicit failures")


def test_run_suite_keeps_order_and_reports_exit_codes() -> None:
    outcomes = {
        "test_a": completed(0, f"{ALL_TESTS_PASSED}\n", ""),
        "test_b": completed(1, "", "ModuleNotFoundError: No module named 'numpy'\n"),
        "test_c": completed(1, "", "AssertionError: broken\n"),
    }
    seen: list[str] = []

    def fake(cmd, **kwargs):
        name = cmd[-1].rsplit(".", 1)[-1]
        seen.append(name)
        return outcomes[name]

    report = run_suite(
        ("test_a", "test_b", "test_c"), cwd=REPO_ROOT, timeout=5.0, run=fake,
        on_result=lambda result: seen.append(f"result:{result.name}"),
    )
    if [r.name for r in report.results] != ["test_a", "test_b", "test_c"]:
        fail("results must follow the requested module order")
    if [r.status for r in report.results] != [PASS, SKIP, FAIL]:
        fail("statuses do not match the injected outcomes")
    if (report.count(PASS), report.count(SKIP), report.count(FAIL)) != (1, 1, 1):
        fail("summary counts are wrong")
    if report.exit_code() != 1 or report.exit_code(strict=True) != 1:
        fail("a real failure must always be a non-zero exit")
    if seen[0] != "test_a" or "result:test_b" not in seen:
        fail("on_result callback was not invoked in order")

    skips_only = SuiteReport((
        classify("test_a", 0, f"{ALL_TESTS_PASSED}\n", "", 0.1),
        classify("test_b", 1, "", "ModuleNotFoundError: No module named 'numpy'\n", 0.1),
    ))
    if skips_only.exit_code() != 0:
        fail("an environment-only SKIP must not break the offline suite")
    if skips_only.exit_code(strict=True) != 1:
        fail("--strict must upgrade SKIP to a non-zero exit")
    summary = format_report(skips_only)
    for expected in ("1 PASS", "1 SKIP", "0 FAIL", "test_b"):
        if expected not in summary:
            fail(f"summary missing {expected!r}: {summary!r}")
    print("  ok  ordering, counts, strict mode and summary are all correct")


def test_cli_list_is_dry_and_complete() -> None:
    """--list 只发现、不执行：无 GPU 栈也能跑，且不会递归启动测试。"""
    proc = subprocess.run(
        [sys.executable, "-m", "faceswapper.tests", "--list"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        check=False, timeout=60,
    )
    if proc.returncode != 0:
        fail(f"--list failed: rc={proc.returncode} stderr={proc.stderr!r}")
    for name in discover_modules(TESTS_DIR):
        if name not in proc.stdout:
            fail(f"--list output is missing {name!r}")
    if ALL_TESTS_PASSED in proc.stdout:
        fail("--list must not execute any test module")
    if f"{len(discover_modules(TESTS_DIR))} module(s)" not in proc.stdout:
        fail("--list must report the module count")
    print("  ok  python -m faceswapper.tests --list is a safe dry run")


def main() -> int:
    print("faceswapper offline suite runner tests")
    test_discover_modules_covers_the_real_suite()
    test_discover_modules_filters_and_sorts()
    test_module_command_shape()
    test_missing_module_extracts_top_level_package()
    test_classify_pass_requires_the_completion_marker()
    test_classify_skips_only_known_runtime_dependencies()
    test_classify_timeout_and_failure_details()
    test_run_module_uses_injected_runner()
    test_run_module_handles_timeout_and_missing_python()
    test_run_suite_keeps_order_and_reports_exit_codes()
    test_cli_list_is_dry_and_complete()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
