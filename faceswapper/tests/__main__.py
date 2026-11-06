"""``python -m faceswapper.tests`` 的入口：一键跑完所有离线测试模块。

子进程的工作目录固定为 repo root，因此从任意目录调用都能 import
``faceswapper``；单模块命令 ``python -m faceswapper.tests.test_cli`` 仍然可用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faceswapper.test_runner import (
    FAIL,
    discover_modules,
    format_report,
    format_result,
    run_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent


def _normalize_module(raw: str) -> str:
    name = raw[:-3] if raw.endswith(".py") else raw
    if name.startswith("faceswapper.tests."):
        name = name.rsplit(".", 1)[-1]
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键运行 faceswapper 离线测试套件（不需要 GPU / 模型）",
    )
    parser.add_argument("--list", action="store_true",
                        help="只列出模块，不执行任何测试")
    parser.add_argument("--module", action="append", default=[], metavar="NAME",
                        help="只跑指定模块，可重复（写模块名或 .py 文件名）")
    parser.add_argument("--strict", action="store_true",
                        help="环境性 SKIP 也算失败，用于 GPU 机器上的正式验收")
    parser.add_argument("--timeout", type=float, default=120.0, metavar="SECONDS",
                        help="单个模块的超时秒数，默认 120")
    parser.add_argument("--python", default=sys.executable,
                        help="解释器路径，默认当前解释器")
    args = parser.parse_args(argv)

    available = discover_modules(TESTS_DIR)
    if args.list:
        for name in available:
            print(name)
        print(f"{len(available)} module(s)")
        return 0

    if args.module:
        names = tuple(dict.fromkeys(_normalize_module(raw) for raw in args.module))
        unknown = [name for name in names if name not in available]
        if unknown:
            parser.error("unknown module(s): " + ", ".join(unknown))
    else:
        names = available

    def on_result(result) -> None:
        print(format_result(result), flush=True)
        if result.status == FAIL:
            tail = (result.stderr or result.stdout).strip()[-1500:]
            if tail:
                print("        " + tail.replace("\n", "\n        "), flush=True)

    print(
        f"faceswapper offline tests: {len(names)} module(s), "
        f"cwd={REPO_ROOT}", flush=True,
    )
    report = run_suite(
        names, python=args.python, cwd=REPO_ROOT,
        timeout=args.timeout, on_result=on_result,
    )
    print(format_report(report))
    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
