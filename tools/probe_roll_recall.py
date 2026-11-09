"""S3 扩角度的收益量化：大角度 roll 导致 detector 漏检的场景。

stdlib-only，本机可直接跑，不需要 GPU / torch / 模型。

背景
----
现有整帧朝向兜底只在 0 检出帧试 90/180/270，加上第一遍的 0 度，等价于
4 个「正立轴」。S3 想把轴补到 8 向（步长 45）或 16 向（步长 22.5），救回
落在 4 向盲区里的歪脸。

这里只有一个真正的未知数：detector 对 roll 的容忍半宽 ``theta_tol``。
若脸偏离最近正立轴 <= theta_tol 就能检出，那么：

  - theta_tol >= 45 度：4 向的覆盖区间首尾相接，8/16 向增益**恒为 0**，S3 白做。
  - theta_tol < 45 度：出现 4 向救不回的盲区，8 向才有收益。

``delta`` 是非 90 倍数副本的等效容忍角折损：45/22.5 度副本用 BILINEAR 真
插值且 expand=True 画布膨胀使脸变小（16:9 帧约缩到 90.5%，方图 70.7%），
而 90 倍数是像素重排、无插值。文档 P4 估 delta≈5 度。16 向的画布膨胀不
比 45 度小，统一 delta 对 16 向偏乐观。

模型只刻画 roll，忽略 yaw / pitch / 脸大小 / 遮挡，且假设 roll 在 0-360
均匀分布——所以算出来的是**上界**，真机素材的角度分布通常更集中。

用法
----
    python tools/probe_roll_recall.py                       # theta_tol 扫表
    python tools/probe_roll_recall.py --theta-tol 30       # 单点细算
    python tools/probe_roll_recall.py --delta 3 --p0 0.08  # 换折损与漏检帧占比
    python tools/probe_roll_recall.py --self-check         # 几何自校验

真机量 theta_tol（需要 GPU 机器 + 模型，本工具会逐角度调 inspect）：

    python tools/probe_roll_recall.py --probe --image frame.png --models ./models

    # 只打印要跑的命令，不执行（可复制到别处 / 改参数）
    python tools/probe_roll_recall.py --emit-probe-plan --image frame.png

量出来的 theta_tol 填回 ``--theta-tol`` 看增益。``--probe`` 会自动加
``--no-auto-orient``：不关掉现有 90 倍数兜底，0 检出角度会被兜底救回来，
测出的 theta_tol 会虚高。另外不要传 ``--detect-score``（cli.py:176 有
unit bug，会把阈值写成 50.0 导致恒 0 检出）。

``--probe`` 用 PIL 的 rotate(expand=True) 生成副本，与生产的
``v2.functional.rotate`` 旋转方向定义可能相反；theta_tol 是对称的半宽，
方向不影响结论，但它是决策量不是像素级精确值。
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 现有实现：0 度由第一遍检测覆盖，probe 只跑 90/180/270。
ANGLE_SETS: dict[str, int] = {"4": 4, "8": 8, "16": 16}
# 90 倍数是像素重排，无插值缩窗；其余轴按 delta 折损。
QUARTER_TURN_EPS = 1e-9
DETECTED_RE = re.compile(r"(\d+)\s+face\(s\)\s+detected")


def circular_dist(a: float, b: float) -> float:
    """环形角度距离，0-180。"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def axes_for(n_angles: int) -> list[float]:
    return [k * 360.0 / n_angles for k in range(n_angles)]


def is_quarter_turn(axis: float) -> bool:
    return abs(axis % 90.0) < QUARTER_TURN_EPS or abs(axis % 90.0 - 90.0) < QUARTER_TURN_EPS


def coverage(theta_tol: float, n_angles: int, delta: float = 0.0,
             step: float = 0.25) -> float:
    """roll 均匀分布时，n_angles 向探测至少有一个轴能检出的比例。"""
    axes = axes_for(n_angles)
    tolerances = [
        theta_tol if is_quarter_turn(a) else max(0.0, theta_tol - delta)
        for a in axes
    ]
    total = int(round(360.0 / step))
    hit = 0
    for i in range(total):
        # 用采样区间中点，避免采样点恰好落在容忍区间边界上被两侧
        # 同时计入（那样每段会多算一个 step，覆盖率虚高）。
        theta = (i + 0.5) * step
        if any(circular_dist(theta, a) < t
               for a, t in zip(axes, tolerances)):
            hit += 1
    return hit / total


def gain_row(theta_tol: float, delta: float, p0: float,
             pr: float) -> dict[str, float]:
    """4/8/16 向覆盖率、相对漏检帧的救回率、整体帧收益 pp。"""
    c4 = coverage(theta_tol, 4, delta)
    c8 = coverage(theta_tol, 8, delta)
    c16 = coverage(theta_tol, 16, delta)
    denom = 1.0 - c4
    return {
        "c4": c4,
        "c8": c8,
        "c16": c16,
        "rescue8": (c8 - c4) / denom if denom > 1e-12 else 0.0,
        "rescue16": (c16 - c4) / denom if denom > 1e-12 else 0.0,
        "pp8": p0 * pr * (c8 - c4) * 100.0,
        "pp16": p0 * pr * (c16 - c4) * 100.0,
    }


def print_sweep(delta: float, p0: float, pr: float) -> None:
    print(f"theta_tol 扫表  delta={delta:g}  p0={p0:g}  pr={pr:g}")
    print("  p0 = 0 检出帧占比；pr = 其中因 roll 漏检的比例")
    print("  整体收益 pp = p0 * pr * (覆盖率增量) * 100\n")
    header = (f"{'theta_tol':>9} {'4向':>8} {'8向':>8} {'16向':>8}"
              f" {'8向救回':>9} {'8向pp':>8} {'16向pp':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tt in range(20, 51, 2):
        r = gain_row(float(tt), delta, p0, pr)
        print(f"{tt:>9} {r['c4'] * 100:>7.1f}% {r['c8'] * 100:>7.1f}%"
              f" {r['c16'] * 100:>7.1f}% {r['rescue8'] * 100:>8.1f}%"
              f" {r['pp8']:>8.3f} {r['pp16']:>8.3f}")
    print()
    r45 = gain_row(45.0, delta, p0, pr)
    print(f"  判据：theta_tol >= 45 度时 4 向覆盖率 {r45['c4'] * 100:.1f}%，"
          f"8 向增益 {r45['pp8']:.3f} pp")
    print("  theta_tol 越小于 45 度，S3 收益越大；先去真机量出这个数。")


def print_point(theta_tol: float, delta: float, p0: float,
                pr: float) -> None:
    r = gain_row(theta_tol, delta, p0, pr)
    print(f"theta_tol={theta_tol:g}  delta={delta:g}  p0={p0:g}  pr={pr:g}")
    print(f"  4 向覆盖率   {r['c4'] * 100:.2f}%")
    print(f"  8 向覆盖率   {r['c8'] * 100:.2f}%  (+{(r['c8'] - r['c4']) * 100:.2f} pp)")
    print(f"  16 向覆盖率  {r['c16'] * 100:.2f}%  (+{(r['c16'] - r['c4']) * 100:.2f} pp)")
    print(f"  在 4 向漏检帧里的救回率：8 向 {r['rescue8'] * 100:.1f}%，"
          f"16 向 {r['rescue16'] * 100:.1f}%")
    print(f"  整体帧收益：8 向 {r['pp8']:.3f} pp，16 向 {r['pp16']:.3f} pp")
    if theta_tol >= 45.0:
        print("  结论：4 向已无缝覆盖，S3 扩角度零收益，不必做。")
    elif r["pp8"] < 0.1:
        print("  结论：整体收益 < 0.1 pp，S3 扩角度性价比低。")
    else:
        print("  结论：8 向有可测收益，值得做 S3；16 向增量再单独评估。")


def self_check() -> int:
    """几何自校验：解析解、饱和、单调性、delta 折损。"""
    def close(a: float, b: float, tol: float = 0.004) -> None:
        if not math.isclose(a, b, abs_tol=tol):
            raise AssertionError(f"expected {b}, got {a}")

    # delta=0 时覆盖率有解析解 min(1, n * 2 * theta_tol / 360)。
    close(coverage(20.0, 4, 0.0), min(1.0, 4 * 2 * 20 / 360))
    close(coverage(20.0, 8, 0.0), min(1.0, 8 * 2 * 20 / 360))
    close(coverage(20.0, 16, 0.0), min(1.0, 16 * 2 * 20 / 360))
    close(coverage(45.0, 4, 0.0), 1.0)
    close(coverage(45.0, 8, 0.0), 1.0)
    # 45 度平手：4 向刚好无缝，8 向增益恒 0（含 delta）。
    for d in (0.0, 5.0, 10.0):
        close(gain_row(45.0, d, 1.0, 1.0)["pp8"], 0.0, 1e-9)
    # 单调性：容忍角越大覆盖率不减。
    for n in ANGLE_SETS.values():
        prev = -1.0
        for tt in (10, 20, 30, 40, 50, 60):
            cur = coverage(float(tt), n, 5.0)
            assert cur >= prev - 1e-9, f"coverage decreased at {n=} {tt=}"
            prev = cur
    # delta 只对非 90 倍数轴生效：4 向不受影响，8 向受影响（未饱和时）。
    close(coverage(25.0, 4, 0.0), coverage(25.0, 4, 8.0), 1e-9)
    assert coverage(25.0, 8, 8.0) < coverage(25.0, 8, 0.0)
    print("  ok  解析解 / 45 度饱和 / 单调性 / delta 只作用于插值轴")
    print("all self-checks passed")
    return 0


def probe_commands(image: Path, models: Path, work_dir: Path,
                   step: int, max_angle: int,
                   detect_input_size: int) -> list[tuple[int, list[str]]]:
    """给出每个角度的 (角度, inspect 命令)。"""
    out = []
    for angle in range(0, max_angle + 1, step):
        rotated = work_dir / f"roll_{angle:03d}.png"
        cmd = [
            sys.executable, "-m", "faceswapper.cli",
            "--models", str(models),
            "inspect", "--target", str(rotated),
            "--no-auto-orient",
            "--detect-input-size", str(detect_input_size),
        ]
        out.append((angle, cmd))
    return out


def emit_probe_plan(image: Path, models: Path, work_dir: Path,
                    step: int, max_angle: int,
                    detect_input_size: int) -> int:
    print("# 真机量 theta_tol 的命令（在仓库根目录、GPU 机器上执行）")
    print("# 必须 --no-auto-orient，否则现有 90 倍数兜底会污染测量。")
    print("# 不要传 --detect-score（cli.py:176 有 unit bug）。")
    print(f"# 用一张只有单个正脸、无其他脸的真实帧：{image}")
    print(f"mkdir -p {work_dir}")
    print(f"# 逐角度生成 expand=True 的旋转副本（PIL BILINEAR）：")
    print(f"python -c \"from PIL import Image; import pathlib; "
          f"im=Image.open(r'{image}'); d=pathlib.Path(r'{work_dir}'); "
          f"[im.rotate(a, resample=Image.BILINEAR, expand=True)"
          f".save(d / f'roll_{{a:03d}}.png') "
          f"for a in range(0, {max_angle + 1}, {step})]\"")
    for angle, cmd in probe_commands(image, models, work_dir, step,
                                     max_angle, detect_input_size):
        print(f"# roll = {angle} 度")
        # 这份计划是给另一台 GPU 机器执行的，不能写死本机 python 路径。
        print("python " + " ".join(cmd[1:]))
    print("# 从 0 度起连续检出 >0 脸的最大角度即观测到的 theta_tol，")
    print("# 填回：python tools/probe_roll_recall.py --theta-tol <观测值>")
    return 0


def run_probe(image: Path, models: Path, work_dir: Path,
              step: int, max_angle: int,
              detect_input_size: int) -> int:
    """逐角度旋转真实帧并调 inspect，估出 theta_tol。"""
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow 生成旋转副本（GPU 机器上随 torchvision 安装）",
              file=sys.stderr)
        return 2
    if not image.is_file():
        print(f"找不到输入帧：{image}", file=sys.stderr)
        return 2
    work_dir.mkdir(parents=True, exist_ok=True)
    src = Image.open(image)
    counts: list[tuple[int, int]] = []
    print(f"逐角度探测：{image}  step={step}  max={max_angle}")
    print("  角度   检出脸数")
    for angle in range(0, max_angle + 1, step):
        rotated = work_dir / f"roll_{angle:03d}.png"
        src.rotate(angle, resample=Image.BILINEAR,
                   expand=True).save(rotated)
        cmd = [
            sys.executable, "-m", "faceswapper.cli",
            "--models", str(models),
            "inspect", "--target", str(rotated),
            "--no-auto-orient",
            "--detect-input-size", str(detect_input_size),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True)
        match = DETECTED_RE.search(proc.stdout)
        if proc.returncode != 0 or match is None:
            print(f"  {angle:>3}   失败（exit={proc.returncode}）")
            print(proc.stdout.strip()[-400:], file=sys.stderr)
            print(proc.stderr.strip()[-400:], file=sys.stderr)
            print("inspect 未能给出脸数，theta_tol 测量中止。",
                  file=sys.stderr)
            return 2
        n = int(match.group(1))
        counts.append((angle, n))
        print(f"  {angle:>3}   {n}")

    theta_tol = 0
    for angle, n in counts:
        if n <= 0:
            break
        theta_tol = angle
    print(f"\n观测 theta_tol ≈ {theta_tol} 度"
          f"（从 0 起连续检出到 {theta_tol} 度）")
    print(f"填回看增益：python tools/probe_roll_recall.py "
          f"--theta-tol {theta_tol}")
    print("提示：若各角度检出数不为 0/1 的干净台阶，说明帧里有多张脸或"
          "背景误检，换一张单人正脸帧重测。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="量化 S3 扩角度对 roll 漏检的收益（stdlib-only）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--theta-tol", type=float, default=None,
                   help="detector 对 roll 的容忍半宽（度），给了就单点细算")
    p.add_argument("--delta", type=float, default=5.0,
                   help="非 90 倍数副本的等效容忍角折损（度，默认 5）")
    p.add_argument("--p0", type=float, default=0.05,
                   help="0 检出帧占比（默认 0.05）")
    p.add_argument("--pr", type=float, default=1.0,
                   help="0 检出帧里因大角度 roll 漏检的比例（默认 1.0）")
    p.add_argument("--self-check", action="store_true",
                   help="跑几何自校验后退出")
    p.add_argument("--emit-probe-plan", action="store_true",
                   help="打印真机量 theta_tol 的命令，不执行")
    p.add_argument("--probe", action="store_true",
                   help="真机逐角度调 inspect 量 theta_tol（需 GPU + 模型 + Pillow）")
    p.add_argument("--image", type=Path, default=Path("frame.png"),
                   help="真机测量用的单人脸帧")
    p.add_argument("--models", type=Path, default=Path("./models"),
                   help="模型目录（--probe / --emit-probe-plan）")
    p.add_argument("--work-dir", type=Path, default=Path(".roll_probe"),
                   help="旋转副本输出目录")
    p.add_argument("--step", type=int, default=5,
                   help="真机测量角度步长（默认 5 度）")
    p.add_argument("--max-angle", type=int, default=90,
                   help="真机测量最大角度（默认 90）")
    p.add_argument("--detect-input-size", type=int, default=640,
                   choices=[320, 416, 480, 640, 800, 960],
                   help="真机测量用的 detector 输入尺寸，须与生产一致")
    args = p.parse_args(argv)

    if args.self_check:
        return self_check()
    if args.emit_probe_plan:
        return emit_probe_plan(args.image, args.models, args.work_dir,
                               args.step, args.max_angle,
                               args.detect_input_size)
    if args.probe:
        return run_probe(args.image, args.models, args.work_dir,
                         args.step, args.max_angle,
                         args.detect_input_size)
    if args.theta_tol is not None:
        print_point(args.theta_tol, args.delta, args.p0, args.pr)
    else:
        print_sweep(args.delta, args.p0, args.pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
