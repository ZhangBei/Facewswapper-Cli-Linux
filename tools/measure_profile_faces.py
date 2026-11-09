#!/usr/bin/env python3
"""侧脸（ROP / yaw / pitch）Step 0：yaw 分档检出率量化与判据。

方案《侧脸（大 yaw  pitch）问题：具体方案与实现步骤》§4 第 1 条把
``yaw -> 检出率`` 曲线列为**必做前提**，并写明「没有这条曲线，下面所有
选型都是猜」。配套的 ``profile_face_plan.py`` 只给了 CSV 表头，本工具补上
「采集 -> 分析 -> 判据」闭环。

纯 stdlib 部分（``--emit-template`` / ``--analyze`` / ``--self-check``）
本地可直接跑，不需要 GPU / torch / numpy。``--collect`` 需要项目的 GPU
机器与模型目录，第三方依赖在函数内延迟 import，本地不会因缺栈而失败。

用法::

    python tools/measure_profile_faces.py --emit-template
    python tools/measure_profile_faces.py --collect frames/ --models ./models \
        --csv yaw.csv --yaw-bin 30-45
    python tools/measure_profile_faces.py --analyze yaw.csv
    python tools/measure_profile_faces.py --self-check

分档建议：``0-15 / 15-30 / 30-45 / 45-60 / >60``，每档 50-100 张，控制
脸尺寸与分辨率不成为混杂变量。文件名里写 ``yawNN``（如
``yaw35_001.jpg``）可让脚本自动分档；否则用 ``--yaw-bin`` 给整批指定，
或事后补 ``yaw_bin`` 列。

``--collect`` 用项目自己的 ``Models.run_detect(..., with_scores=True)``，
因此与生产路径同源；重试轮沿用 ``detect_retry`` 的 960 / +0.2 参数，
``kps_quality`` 也来自同一个 stdlib 模块。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faceswapper.detect_retry import (  # noqa: E402  stdlib-only
    RETRY_INPUT_SIZE,
    RETRY_SCORE_BUMP,
    kps_quality,
)

CSV_COLUMNS = (
    "yaw_bin", "img_id", "face_px", "detected_count", "max_conf",
    "is_zero_detect", "detect_input_size", "detect_score",
    "retry_detected_count", "retry_max_conf", "kps_quality",
    "target_match_score", "matched",
)
YAW_BINS = ("0-15", "15-30", "30-45", "45-60", ">60", "unknown")
LARGE_YAW_BINS = ("45-60", ">60")
YAW_NAME_RE = re.compile(r"yaw(-?\d+(?:\.\d+)?)", re.IGNORECASE)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def bin_yaw(degrees) -> str:
    """把偏航角（度）落到方案的分档里；左右对称，取绝对值。"""
    if degrees is None:
        return "unknown"
    try:
        a = abs(float(degrees))
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(a):
        return "unknown"
    if a < 15:
        return "0-15"
    if a < 30:
        return "15-30"
    if a < 45:
        return "30-45"
    if a < 60:
        return "45-60"
    return ">60"


def yaw_from_name(name: str):
    """从 ``yaw35_001.jpg`` 这类文件名里取角度，取不到返回 None。"""
    m = YAW_NAME_RE.search(name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def face_px(kps) -> float:
    """5 点包围盒对角线，作为脸尺寸的代理（避免混入分辨率差异）。"""
    try:
        xs = [float(p[0]) for p in kps]
        ys = [float(p[1]) for p in kps]
    except (TypeError, ValueError, IndexError):
        return 0.0
    if not xs or not ys:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mean(rows, key: str) -> float:
    values = [_num(r, key) for r in rows]
    return sum(values) / len(values) if values else 0.0


def analyze_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """按 yaw_bin 聚合，返回 ``(分档报告, 总体报告)``。纯 stdlib。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        b = str(row.get("yaw_bin") or "unknown").strip()
        if b not in YAW_BINS:
            b = "unknown"
        grouped[b].append(row)

    reports = []
    for b in YAW_BINS:
        items = grouped.get(b, [])
        if not items:
            continue
        zeros = [r for r in items if _num(r, "is_zero_detect") >= 1
                 or _num(r, "detected_count") == 0]
        recovered = [r for r in zeros
                     if _num(r, "retry_detected_count") > 0]
        reports.append({
            "bin": b,
            "n": len(items),
            "zero_rate": len(zeros) / len(items),
            "retry_recovered_rate": (len(recovered) / len(zeros)
                                     if zeros else 0.0),
            "mean_face_px": _mean(items, "face_px"),
            "mean_max_conf": _mean(items, "max_conf"),
            "mean_kps_quality": _mean(items, "kps_quality"),
        })

    total = len(rows)
    large_n = sum(r["n"] for r in reports if r["bin"] in LARGE_YAW_BINS)
    overall = {
        "total": total,
        "large_yaw_pct": (large_n / total * 100.0) if total else 0.0,
        "zero_rate": (sum(r["zero_rate"] * r["n"] for r in reports) / total
                      if total else 0.0),
    }
    return reports, overall


def recommendations(reports: list[dict], overall: dict) -> list[str]:
    """把方案 §4 的判据写成可执行的下一步建议。"""
    out: list[str] = []
    if overall["total"] == 0:
        return ["没有有效样本：先按 --emit-template 采集并填好 CSV。"]

    pct = overall["large_yaw_pct"]
    if pct < 5.0:
        out.append(
            f"yaw>45° 占比 {pct:.1f}% < 5%：方案 §4 第 1 条判据命中，"
            "所有训练 / 换模型方案（P4 / P5）直接排除，只做零成本 A/B。"
        )
    elif pct > 10.0:
        out.append(
            f"yaw>45° 占比 {pct:.1f}% > 10%：若 T1-T5 与 P1/P2 全部到顶仍"
            "不够，才评估 P4（SCRFD finetune）；P5 仍不建议（精度退步）。"
        )
    else:
        out.append(
            f"yaw>45° 占比 {pct:.1f}%：按 T1 -> T2 -> T3/T4 -> T5 逐档 A/B，"
            "不要一次全开（mirror+dim2+glintr100 = 识别 ×32）。"
        )

    small = [r for r in reports if r["bin"] in ("0-15", "15-30", "30-45")]
    large = [r for r in reports if r["bin"] in LARGE_YAW_BINS]
    if large:
        worst = max(large, key=lambda r: r["zero_rate"])
        # 用低 yaw 档的脸尺寸作参照（方案要求「控制脸尺寸一致」），
        # 不能用全档均值——那样单档数据永远判不出「脸偏小」。
        ref_px = (sum(r["mean_face_px"] for r in small) / len(small)
                  if small else worst["mean_face_px"])
        if worst["zero_rate"] >= 0.3 and worst["mean_face_px"] >= 0.8 * ref_px:
            out.append(
                f"{worst['bin']}° 档零检出率 {worst['zero_rate'] * 100:.1f}% "
                f"且脸尺寸不小（{worst['mean_face_px']:.0f}px vs 低 yaw 参照 "
                f"{ref_px:.0f}px）：姿态本身导致，走 L1/L2/L3。"
            )
        elif worst["zero_rate"] >= 0.3:
            out.append(
                f"{worst['bin']}° 档零检出率高但脸偏小"
                f"（{worst['mean_face_px']:.0f}px vs {ref_px:.0f}px）："
                "优先只试 --detect-input-size 960，别动权重"
                "（方案 §4 Step 0 判据）。"
            )
    if small and any(r["zero_rate"] >= 0.2 for r in small):
        out.append(
            "低 yaw 档也有明显漏检：先查小脸 / 模糊 / 遮挡，别归因于姿态。"
        )

    zeros = [r for r in reports if r["zero_rate"] > 0]
    if zeros:
        avg_recovery = sum(r["retry_recovered_rate"] for r in zeros) / len(zeros)
        if avg_recovery < 0.2:
            out.append(
                f"重试轮平均只救回 {avg_recovery * 100:.1f}% 的零检出样本："
                "方案 P4 指出的「升 960 的置信度增益 > +0.2」假设可能不成立，"
                "--detect-retry 收益存疑，需用 retry_max_conf 列核对。"
            )
        else:
            out.append(
                f"重试轮平均救回 {avg_recovery * 100:.1f}%："
                "--detect-retry 的 +0.2 阈值假设基本成立，可进 T1。"
            )
    out.append(
        "每档 A/B 都要同时记「该换没换帧数 / 误换帧数 / 命中帧率 / 逐帧"
        "耗时 / 不传开关时 PSNR=inf」；P1/P2 的阈值须用本 CSV 标定。"
    )
    return out


def emit_template() -> int:
    print("yaw 分档采集 CSV 表头（每档 50-100 张，控制脸尺寸一致）：")
    print(",".join(CSV_COLUMNS))
    print()
    print("分档：0-15 / 15-30 / 30-45 / 45-60 / >60（yaw_bin 列）")
    print("文件名写 yawNN 可自动分档；否则用 --yaw-bin 指定或事后补列。")
    return 0


def print_analysis(rows: list[dict]) -> int:
    reports, overall = analyze_rows(rows)
    print(f"样本总数 {overall['total']}  yaw>45° 占比 "
          f"{overall['large_yaw_pct']:.1f}%  整体零检出率 "
          f"{overall['zero_rate'] * 100:.1f}%")
    print()
    header = (f"{'yaw_bin':<10} {'n':>5} {'zero%':>8} {'retry救回%':>11}"
              f" {'face_px':>9} {'max_conf':>10} {'kps_q':>7}")
    print(header)
    print("-" * len(header))
    for r in reports:
        print(f"{r['bin']:<10} {r['n']:>5} {r['zero_rate'] * 100:>7.1f}%"
              f" {r['retry_recovered_rate'] * 100:>10.1f}%"
              f" {r['mean_face_px']:>9.1f} {r['mean_max_conf']:>10.3f}"
              f" {r['mean_kps_quality']:>7.3f}")
    print()
    print("--- 判据与下一步 ---")
    for line in recommendations(reports, overall):
        print(f"  - {line}")
    return 0


def analyze(csv_path: Path) -> int:
    if not csv_path.is_file():
        print(f"找不到 CSV：{csv_path}", file=sys.stderr)
        return 2
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return print_analysis(rows)


def collect(img_dir: Path, models_dir: Path, csv_path: Path,
            yaw_bin: str | None, detect_type: str, input_size: int,
            detect_score: float) -> int:
    """真机采集：逐张跑 detector（含重试轮），写 yaw 分档 CSV。

    GPU 依赖在函数内延迟 import，本地无 GPU 时只在调用本模式才报错。
    """
    try:
        import cv2
        import numpy as np
        import torch
        from faceswapper.Models import Models
    except ImportError as exc:
        print(f"--collect 需要项目 GPU 栈：{exc}", file=sys.stderr)
        print("请在 Ubuntu + NVIDIA GPU 环境、装好依赖后运行。",
              file=sys.stderr)
        return 2
    if not img_dir.is_dir():
        print(f"找不到图片目录：{img_dir}", file=sys.stderr)
        return 2

    images = sorted(p for p in img_dir.rglob("*")
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"{img_dir} 里没有图片", file=sys.stderr)
        return 2

    models = Models()
    models.set_models_folder(str(models_dir))
    retry_score = min(1.0, float(detect_score) + RETRY_SCORE_BUMP / 100.0)
    print(f"采集 {len(images)} 张  detect_type={detect_type} "
          f"input_size={input_size} score={detect_score} "
          f"retry=({RETRY_INPUT_SIZE}, {retry_score})")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, path in enumerate(images, 1):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  跳过无法解码的 {path.name}", file=sys.stderr)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda")
            img = img.permute(2, 0, 1)  # (3,H,W) uint8 CUDA

            kpss, scores = models.run_detect(
                img, detect_type, max_num=10, score=detect_score,
                input_size=input_size, with_scores=True,
            )
            retry_kpss, retry_scores = models.run_detect(
                img, detect_type, max_num=10, score=retry_score,
                input_size=RETRY_INPUT_SIZE, with_scores=True,
            )
            n = int(len(kpss))
            max_conf = float(max(scores)) if len(scores) else 0.0
            retry_max_conf = (float(max(retry_scores))
                              if len(retry_scores) else 0.0)
            px = max((face_px(k) for k in kpss), default=0.0)
            quality = min((kps_quality(k) for k in kpss), default=0.0)
            resolved_bin = str(yaw_bin) if yaw_bin else bin_yaw(
                yaw_from_name(path.name)
            )
            writer.writerow({
                "yaw_bin": resolved_bin,
                "img_id": path.name,
                "face_px": f"{px:.1f}",
                "detected_count": n,
                "max_conf": f"{max_conf:.4f}",
                "is_zero_detect": int(n == 0),
                "detect_input_size": input_size,
                "detect_score": detect_score,
                "retry_detected_count": int(len(retry_kpss)),
                "retry_max_conf": f"{retry_max_conf:.4f}",
                "kps_quality": f"{quality:.4f}",
                "target_match_score": "",
                "matched": "",
            })
            if index % 25 == 0:
                print(f"  {index}/{len(images)}")
    print(f"已写出 {csv_path}；用 --analyze 看分档曲线与判据。")
    return 0


def self_check() -> int:
    """纯 stdlib 自校验：分档、聚合、判据、退化输入。"""
    def close(a, b, tol=1e-9):
        if not math.isclose(a, b, abs_tol=tol):
            raise AssertionError(f"expected {b}, got {a}")

    for deg, expected in ((0, "0-15"), (14.9, "0-15"), (15, "15-30"),
                          (44.9, "30-45"), (45, "45-60"), (90, ">60"),
                          (-60, ">60"), (None, "unknown"),
                          (float("nan"), "unknown"), ("abc", "unknown")):
        if bin_yaw(deg) != expected:
            raise AssertionError(f"bin_yaw({deg!r}) = {bin_yaw(deg)!r}")
    if yaw_from_name("yaw35_001.jpg") != 35.0:
        raise AssertionError("yaw_from_name must parse yawNN")
    if yaw_from_name("plain.jpg") is not None:
        raise AssertionError("yaw_from_name must ignore unrelated names")

    def row(b, zero, retry, px=100.0, conf=0.8, q=0.9):
        return {"yaw_bin": b, "detected_count": 0 if zero else 1,
                "is_zero_detect": int(zero),
                "retry_detected_count": int(retry), "face_px": px,
                "max_conf": conf, "kps_quality": q}

    reports, overall = analyze_rows([])
    if reports or overall["total"] != 0:
        raise AssertionError("empty CSV must produce empty reports")
    if "没有有效样本" not in recommendations(reports, overall)[0]:
        raise AssertionError("empty CSV must say so explicitly")

    rows = ([row("0-15", False, False) for _ in range(90)]
            + [row("45-60", True, True, px=110) for _ in range(8)]
            + [row(">60", True, False, px=105) for _ in range(2)])
    reports, overall = analyze_rows(rows)
    close(overall["total"], 100)
    close(overall["large_yaw_pct"], 10.0)
    large = [r for r in reports if r["bin"] == "45-60"][0]
    close(large["zero_rate"], 1.0)
    close(large["retry_recovered_rate"], 1.0)
    notes = recommendations(reports, overall)
    if not any("姿态本身" in n for n in notes):
        raise AssertionError("large-yaw zero detect with big faces must be flagged")
    # 大 yaw 档脸明显小于低 yaw 参照时，给出「只试 960」的建议。
    small_rows = ([row("0-15", False, False, px=100) for _ in range(20)]
                  + [row("45-60", True, False, px=20) for _ in range(20)])
    notes = recommendations(*analyze_rows(small_rows))
    if not any("960" in n for n in notes):
        raise AssertionError("small faces must route to --detect-input-size 960")
    # 占比 <5% 时排除训练方案。
    low = [row("0-15", False, False) for _ in range(98)] + [
        row(">60", True, False) for _ in range(2)]
    if not any("P4" in n and "排除" in n
               for n in recommendations(*analyze_rows(low))):
        raise AssertionError("yaw>45% <5% must exclude P4/P5")
    # 缺列 / 空值不能崩。
    analyze_rows([{"yaw_bin": ""}, {"yaw_bin": "bogus"}])
    print("  ok  bin_yaw / yaw_from_name / aggregate / verdicts / degenerate")
    print("all self-checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="侧脸 Step 0：yaw 分档检出率量化与判据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-template", action="store_true",
                      help="打印 CSV 表头与采集说明")
    mode.add_argument("--analyze", type=Path, metavar="CSV",
                      help="分析已采集的 CSV，输出分档曲线与判据（纯 stdlib）")
    mode.add_argument("--collect", type=Path, metavar="IMG_DIR",
                      help="真机逐张采集（需 GPU + 模型，依赖延迟 import）")
    mode.add_argument("--self-check", action="store_true",
                      help="跑纯 stdlib 自校验")
    p.add_argument("--models", type=Path, default=Path("./models"),
                   help="--collect 的模型目录")
    p.add_argument("--csv", type=Path, default=Path("yaw_profile.csv"),
                   help="--collect 的输出 CSV")
    p.add_argument("--yaw-bin", default=None, choices=YAW_BINS,
                   help="给整批图片指定 yaw 分档；省略则从文件名 yawNN 推断")
    p.add_argument("--detect-type", default="Retinaface",
                   choices=["Retinaface", "SCRDF"])
    p.add_argument("--detect-input-size", type=int, default=640,
                   choices=[320, 416, 480, 640, 800, 960])
    p.add_argument("--detect-score", type=float, default=0.5,
                   help="首轮检测阈值 0-1（注意 CLI 的 --detect-score 有 "
                        "unit bug，这里是独立的 0-1 值）")
    args = p.parse_args(argv)

    if args.emit_template:
        return emit_template()
    if args.analyze is not None:
        return analyze(args.analyze)
    if args.collect is not None:
        return collect(args.collect, args.models, args.csv, args.yaw_bin,
                       args.detect_type, args.detect_input_size,
                       args.detect_score)
    return self_check()


if __name__ == "__main__":
    sys.exit(main())
