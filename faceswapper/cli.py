"""headless Faceswapper 的命令行入口。

图片：

    python -m faceswapper.cli swap \
        --source face.jpg --target target.jpg --output out.png

    python -m faceswapper.cli swap \
        --source face.jpg --target-dir ./targets --output-dir ./out

视频（一个 source 身份，每一张脸）：

    python -m faceswapper.cli video \
        --source face.jpg --target clip.mp4 --output out.mp4

查看检测结果：

    python -m faceswapper.cli inspect --target target.jpg

模型：

    python -m faceswapper.cli download-models
    python -m faceswapper.cli check-models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any

from faceswapper.errors import HeadlessError
from faceswapper.model_store import (
    DEFAULT_PROVIDER,
    ENDPOINT_ENV,
    MODEL_SPECS,
    PROVIDER_LABELS,
    ensure_models,
    format_size,
    inspect_models,
    specs_for_names,
    specs_for_parameters,
)
from faceswapper.parameters import PARAMETER_BY_NAME
from faceswapper.parameters_io import load as load_params


def _format_elapsed(seconds: float) -> str:
    """把秒数格式化成 ``HH:MM:SS.s``。

    先换成整数十分之一秒再做进位，避免 59.99 秒被浮点格式化成 ``60.0``。
    """
    tenths = max(0, round(float(seconds) * 10))
    hours, tenths = divmod(tenths, 36000)
    minutes, tenths = divmod(tenths, 600)
    secs, tenth = divmod(tenths, 10)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{tenth}"


def _format_avg_fps(frames: int | float | None, seconds: float) -> str:
    """把平均处理速度格式化成可直接拼在 elapsed 后面的片段。

    帧数来自视频编码循环的实际计数，不是容器声明的 ``stream.frames``——
    后者可能为 0 或只是估算值。缺帧数、零帧、非正耗时都返回空串，避免
    除零或把一次不完整运行误报成 0.0 fps。
    """
    if frames is None or frames <= 0 or seconds <= 0:
        return ""
    return f" (avg {frames / seconds:.1f} fps)"


def _format_profile(times: dict[str, float]) -> str:
    """把分段耗时格式化成 ``profile: decode=Xs swap=Ys encode=Zs``。

    三段顺序固定，缺的段补 0.00s。并行路径下三段会互相重叠，之和不等于
    总用时，所以这里不做任何归一化，只呈现各段自身的累计耗时。
    """
    parts = [
        f"{key}={float(times.get(key, 0.0)):.2f}s"
        for key in ("decode", "swap", "encode")
    ]
    return "profile: " + " ".join(parts)


# CLI 接受的图片扩展名，source 脸和 target 图片都算。放在这里（而不是
# faceswapper.headless）是为了让 argument parser 和 --help 在没有装
# OpenCV / torch 的机器上也能用。
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
}

# `download-models --model` 和 `check-models --model` 接受的模型组/模型名。
# 两边共用同一份列表，以保持步调一致。
DOWNLOAD_MODEL_CHOICES = [
    "base", "retinaface", "arcface", "inswapper", "gfpgan",
    "scrfd", "hyperswap", "gpen256", "gpen512", "codeformer",
    "xseg", "faceparser", "dflxseg", "resnet50", "all",
]

# 由 --model-provider（顶层）和 download-models --provider 共用。
_PROVIDER_HELP = (
    "download source: a URL, or one of "
    + "/".join(sorted(PROVIDER_LABELS))
    + f"; repeatable (default: {DEFAULT_PROVIDER}, "
    f"or ${ENDPOINT_ENV} when set)"
)


def _headless():
    """延迟导入 faceswapper.headless。

    重依赖（OpenCV、torch、ONNX Runtime）只在真正执行换脸的指令时才需要，
    这样在还没装完运行时的机器上 `--help` 和 `params` 依然可用。
    """
    try:
        from faceswapper import headless
    except ImportError as exc:
        raise HeadlessError(
            f"missing runtime dependency: {exc}\n"
            "Install the CUDA runtime with:\n"
            "  python tools/install.py cu128"
        ) from exc
    return headless


def _collect_sources(values: list[str] | None) -> list[str]:
    """展开 --source：既可以是文件，也可以是装脸图的目录。"""
    out: list[str] = []
    for value in values or []:
        if os.path.isdir(value):
            for name in sorted(os.listdir(value)):
                path = os.path.join(value, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    out.append(path)
        else:
            out.append(value)
    return out


def _collect_targets(values: list[str] | None, target_dir: str | None) -> list[str]:
    """展开 --target，再加上 --target-dir 目录里的所有内容。"""
    out: list[str] = []
    for value in values or []:
        if os.path.isdir(value):
            for name in sorted(os.listdir(value)):
                path = os.path.join(value, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    out.append(path)
        else:
            out.append(value)
    if target_dir:
        for name in sorted(os.listdir(target_dir)):
            path = os.path.join(target_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                out.append(path)
    return out


def _parameter_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """把保存的 parameters 和显式给出的 CLI 旗标合并成一个 dict。"""
    overrides: dict[str, Any] = {}
    if getattr(args, "params", None):
        overrides.update(load_params(args.params))

    # 显式给出的 CLI 旗标优先于保存的 parameter 文件。
    if getattr(args, "detect_type", None):
        overrides["DetectTypeTextSel"] = args.detect_type
    if getattr(args, "detect_input_size", None):
        overrides["DetectInputSizeTextSel"] = str(args.detect_input_size)
    if getattr(args, "detect_score", None) is not None:
        overrides["DetectScoreSlider"] = args.detect_score * 100
    if getattr(args, "swapper_model", None):
        overrides["SwapperTypeTextSel"] = args.swapper_model
    if getattr(args, "color_match", False):
        overrides["ColorMatchSwitch"] = True
    if getattr(args, "restorer", None):
        overrides["RestorerSwitch"] = True
        overrides["RestorerTypeTextSel"] = args.restorer
    if getattr(args, "occluder", False):
        overrides["OccluderSwitch"] = True
    if getattr(args, "xseg", False):
        overrides["DFLXSegSwitch"] = True
    if getattr(args, "face_parser", False):
        overrides["FaceParserSwitch"] = True
    return overrides


def _build_options(args: argparse.Namespace, headless=None) -> "SwapOptions":
    """把解析后的 args 变成 SwapOptions，并加载保存的 parameters。"""
    if headless is None:
        headless = _headless()
    SwapOptions = headless.SwapOptions
    overrides = _parameter_overrides(args)

    options = SwapOptions(
        models_folder=args.models,
        face_policy=args.face_policy,
        detect_type=args.detect_type or "Retinaface",
        detect_input_size=args.detect_input_size or 640,
        detect_score=(
            args.detect_score / 100.0
            if args.detect_score is not None
            else 0.5
        ),
        merge_mode=args.merge or "Mean",
        overrides=overrides,
        backend=args.backend,
        arena_strategy=args.arena_strategy,
    )
    return options


def _prepare_models(args: argparse.Namespace) -> None:
    """只下载本次运行真正会加载的模型。

    选择依据是 parameter snapshot（保存的 parameters 加上覆盖它们的 CLI
    旗标），所以打开 --restorer 或 --occluder 只会拉进那一个文件，而不是
    所有可选模型。
    """
    parameters = dict(_parameter_overrides(args))
    # swap pipeline 读的就是这份 snapshot，所以也把默认值填进去：一个从没提过
    # RestorerSwitch 的 --params 文件，对默认关闭的 restorer 也能得到正确
    # 答案。
    try:
        from faceswapper.parameters import default_values
        merged = {**default_values(), **parameters}
    except Exception:
        merged = parameters
    # 旗标已经折叠进上面的 snapshot，所以 spec 解析只读 snapshot——每个值
    # 只有一个真相来源。
    specs = specs_for_parameters(merged)
    ensure_models(
        args.models,
        specs,
        providers=args.model_provider or None,
        allow_download=not args.no_auto_download,
    )


def _print_model_list(
    models_dir: str,
    specs,
    statuses: dict[str, str] | None = None,
) -> None:
    statuses = statuses or inspect_models(models_dir, specs)
    print(f"models: {os.path.abspath(models_dir)}")
    for spec in specs:
        status = statuses[spec.name]
        # 用户自备的文件没有上游 size 可报。
        size = format_size(spec.size) if spec.size else "  ?"
        print(
            f"{status:<8} {spec.local_name:<28} "
            f"{size:>10}  {spec.role}"
        )


def _cmd_swap(args: argparse.Namespace) -> int:
    sources = _collect_sources(args.source)
    if not sources:
        raise HeadlessError("no source face supplied (use --source)")
    for path in sources:
        if not os.path.isfile(path):
            raise HeadlessError(f"source face not found: {path}")

    targets = _collect_targets(args.target, args.target_dir)
    if not targets:
        raise HeadlessError("no target image supplied (use --target or --target-dir)")
    for path in targets:
        if not os.path.isfile(path):
            raise HeadlessError(f"target image not found: {path}")

    # 只在输入确认无误之后才导入重运行时，这样路径打错能以清晰的消息快速失败。
    _prepare_models(args)
    headless = _headless()
    options = _build_options(args, headless)

    if args.output and len(targets) == 1:
        out_path = args.output
    else:
        out_dir = args.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = None

    with headless.ImageSwapper(options) as swapper:
        if out_path:
            report = swapper.swap(sources, targets[0], out_path)
            print(
                f"swapped {report['swapped_faces']}/{report['detected_faces']} "
                f"face(s) -> {report['output']}"
            )
        else:
            def progress(index: int, total: int, target: str) -> None:
                print(f"[{index}/{total}] {target}", flush=True)

            reports = swapper.swap_batch(
                sources, targets, out_dir, on_progress=progress
            )
            total_faces = sum(r["swapped_faces"] for r in reports)
            print(
                f"done: {len(reports)} image(s), {total_faces} face(s) swapped "
                f"-> {out_dir}"
            )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """列出 target 图片里检测到的脸（不换脸）。"""
    if not os.path.isfile(args.target):
        raise HeadlessError(f"target image not found: {args.target}")
    _prepare_models(args)
    headless = _headless()
    options = _build_options(args, headless)
    with headless.ImageSwapper(options) as swapper:
        rgb = headless._read_image(args.target)
        slots = swapper._detect_face_slots(rgb)
        print(f"{args.target}: {len(slots)} face(s) detected")
        for index, slot in enumerate(slots):
            print(f"  [{index}] scale {slot['Scale']:.1f}")
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    """把一个 source 身份换到视频里的每一张脸上。"""
    sources = _collect_sources(args.source)
    if not sources:
        raise HeadlessError("no source face supplied (use --source)")
    for path in sources:
        if not os.path.isfile(path):
            raise HeadlessError(f"source face not found: {path}")

    if not os.path.isfile(args.target):
        raise HeadlessError(f"video not found: {args.target}")
    target_faces = _collect_sources(args.target_face)
    for path in target_faces:
        if not os.path.isfile(path):
            raise HeadlessError(f"target person reference not found: {path}")
    if args.output is None:
        raise HeadlessError("--output is required for video (e.g. out.mp4)")
    if os.path.abspath(args.output) == os.path.abspath(args.target):
        raise HeadlessError("--output must differ from --target")

    _prepare_models(args)
    headless = _headless()

    options = headless.VideoOptions(
        models_folder=args.models,
        # 多张源图会先合并成一个身份，整段视频只用这一个身份；且总是
        # 替换所有通过 --target-face 筛选的脸。
        face_policy="all",
        target_person_paths=target_faces,
        target_threshold=args.target_threshold,
        detect_type=args.detect_type or "Retinaface",
        detect_input_size=args.detect_input_size or 640,
        detect_score=(
            args.detect_score / 100.0
            if args.detect_score is not None
            else 0.5
        ),
        merge_mode=args.merge or "Mean",
        overrides=_parameter_overrides(args),
        backend=args.backend,
        crf=args.crf,
        preset=args.preset,
        keep_audio=not args.no_audio,
        every_nth=args.every_nth,
        workers=args.workers,
        profile=args.profile,
        encode_thread=args.encode_thread,
        defer_d2h=args.defer_d2h,
        force_recognize=args.force_recognize,
        encoder=args.encoder,
        arena_strategy=args.arena_strategy,
    )

    def progress(index: int, total: int) -> None:
        if total > 0:
            pct = 100.0 * index / total
            print(f"\rframe {index}/{total} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\rframe {index}", end="", flush=True)

    with headless.VideoSwapper(options) as swapper:
        report = swapper.swap(sources, args.target, args.output, on_progress=progress)
    # 把实际处理帧数交回 main()，让它和 elapsed 用同一个时间基准算 avg fps。
    # 不改动 report["frames"]（容器元数据），done 行的输出因此保持不变。
    args._processed_frames = int(report.get("processed_frames", 0))
    print()
    print(
        f"done: {report['frames']} frame(s) at {report['fps']:.3f} fps, "
        f"{report['width']}x{report['height']}, "
        f"audio={'yes' if report['audio'] else 'no'} -> {report['output']}"
    )
    if args.profile and report.get("profile"):
        print(_format_profile(report["profile"]))
    return 0


def _cmd_params(args: argparse.Namespace) -> int:
    """打印已知的 parameter 名及其默认值。"""
    for name, param in PARAMETER_BY_NAME.items():
        try:
            default = param.default
        except KeyError:
            continue
        print(f"{name}\t{param.kind}\t{default}")
    return 0


def _cmd_download_models(args: argparse.Namespace) -> int:
    specs = specs_for_names(args.model or ["base"])
    if args.list:
        _print_model_list(args.models, specs)
        return 0
    ensure_models(
        args.models,
        specs,
        providers=args.provider or None,
        force=args.force,
    )
    _print_model_list(args.models, specs)
    return 0


def _cmd_check_models(args: argparse.Namespace) -> int:
    explicit_selection = bool(args.model)
    specs = specs_for_names(args.model) if args.model else MODEL_SPECS
    statuses = inspect_models(
        args.models,
        specs,
        verify=not args.quick,
    )
    _print_model_list(args.models, specs, statuses)
    # 用户自备的 spec（上游没有 release、没有 checksum）只能报告为 present 或
    # missing；缺一个只是提示，不算失败，因为它从不被下载，而且很少用得上。
    invalid = [
        spec for spec in specs
        if statuses[spec.name] not in ("valid", "present")
        and (explicit_selection or spec.required) and spec.size
    ]
    if invalid:
        repair_args = " ".join(
            f"--model {spec.name}" for spec in invalid
        )
        raise HeadlessError(
            f"invalid model(s): {', '.join(spec.local_name for spec in invalid)}\n"
            f"repair with: python -m faceswapper.cli --models {args.models} "
            f"download-models {repair_args}"
        )
    if not args.quick:
        print("all model checksum checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceswapper",
        description="Faceswapper - command-line face swapping.",
        # 顶层有 --models / --model-provider。argparse 默认允许长选项缩写，
        # 且在把参数交给子解析器之前会全局扫描一遍所有参数：子命令里的
        # --model 会被主 parser 当成这两个顶层选项的歧义前缀并直接报错。
        # 关掉缩写后 --model 才能原样传给子解析器。
        allow_abbrev=False,
    )
    parser.add_argument(
        "--models",
        default="./models",
        help="存放 det_10g.onnx / w600k_r50.onnx / inswapper_*.onnx 的目录",
    )
    parser.add_argument(
        "--model-provider",
        action="append",
        default=[],
        help=_PROVIDER_HELP,
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="base 模型缺失或损坏时直接失败，不去下载",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--params", default=None,
                       help="从中加载默认值的 saved_parameters.json")
        p.add_argument("--detect-type", choices=["Retinaface", "SCRDF"],
                       default=None, help="人脸检测模型")
        p.add_argument("--detect-input-size", type=int, choices=[320, 416, 480, 640],
                       default=None, help="detector 输入分辨率")
        p.add_argument("--detect-score", type=int, default=None,
                       help="检测分数阈值，1-100")
        p.add_argument("--merge",
                       choices=["Mean", "Median", "Sph", "Geo", "Qual"],
                       default=None, help="如何融合多个 source 脸")
        p.add_argument("--swapper-model",
                       choices=["inswapper-128", "inswapper-256", "inswapper-512", "hyperswap-256"],
                       default=None, help="swapper 模型与输出分辨率")
        p.add_argument("--color-match", action="store_true",
                       help="启用 LAB 色彩匹配")
        p.add_argument("--restorer", nargs="?", const="GFPGAN",
                       choices=["GFPGAN", "CF", "GPEN256", "GPEN512"],
                       default=None,
                       help="用指定模型修复换好的脸（只下载那一个"
                            "文件）；省略值则用 GFPGAN")
        p.add_argument("--occluder", action="store_true",
                       help="遮挡掩码（下载 occluder.onnx）")
        p.add_argument("--xseg", action="store_true",
                       help="DeepFaceLab XSeg 遮挡掩码（下载"
                            "dfl_xseg.onnx）")
        p.add_argument("--face-parser", action="store_true",
                       help="用 face parser 生成背景掩码（下载"
                            "faceparser_resnet34.onnx）")
        p.add_argument("--backend", choices=["cuda", "trt"], default="cuda",
                       help="推理后端（默认 cuda，不用 TensorRT）")
        p.add_argument("--arena-strategy", choices=["same", "pow2"],
                       default="same",
                       help="CUDA EP 的 VRAM arena 扩展策略：same（默认，按需"
                            "分配）或 pow2（减少重复 cudaMalloc，只是让显存"
                            "数字变大，不是提速手段）")

    swap = sub.add_parser("swap", help="换一张或多张图片里的脸")
    add_common(swap)
    swap.add_argument("--source", action="append", default=[],
                       help="source 脸图片（可重复，或给目录）")
    swap.add_argument("--target", action="append", default=[],
                       help="target 图片（可重复，或给目录）")
    swap.add_argument("--target-dir", default=None,
                       help="待处理的 target 图片目录")
    swap.add_argument("--output", default=None,
                       help="输出图片路径（仅单个 target）")
    swap.add_argument("--output-dir", default=None,
                       help="输出目录（批量模式）")
    swap.add_argument("--face-policy", choices=["all", "largest"],
                       default="all",
                       help="换哪些 target 脸（默认 all）")
    swap.set_defaults(func=_cmd_swap)

    video = sub.add_parser(
        "video", help="把视频里每一张脸换成一个 source 身份"
    )
    add_common(video)
    video.add_argument("--source", action="append", default=[],
                       help="source face image (repeatable, or a folder)")
    video.add_argument("--target-face", action="append", default=[],
                       help="视频里要被替换掉的人的参考图（可重复，"
                            "或给目录）。省略则换掉每一张检测到的脸")
    video.add_argument("--target-threshold", type=float, default=55.0,
                       help="匹配 target 人物的相似度阈值（0-100，"
                            "默认 55）")
    video.add_argument("--target", required=True,
                       help="输入视频文件")
    video.add_argument("--output", required=True,
                       help="输出视频文件（如 out.mp4）")
    video.add_argument("--crf", type=int, default=18,
                       help="H.264 质量，越小越好（默认 18）")
    video.add_argument("--preset", default="medium",
                       choices=["ultrafast", "veryfast", "fast", "medium",
                                "slow", "veryslow"],
                       help="x264 速度/质量预设（默认 medium）")
    video.add_argument("--encoder", choices=["libx264", "nvenc"],
                       default="libx264",
                       help="视频编码器：libx264（默认）或 h264_nvenc。"
                            "nvenc 会先探测可用性，不可用直接报错；CRF/preset "
                            "会映射成 CQ/p1-p7，语义不等价")
    video.add_argument("--no-audio", action="store_true",
                       help="不拷贝源音频轨")
    video.add_argument("--every-nth", type=int, default=1,
                       help="只换第 N 帧，其余原样透传（1 = 全部帧；"
                            "用于计时测试）")
    video.add_argument("--workers", type=int, default=1,
                       help="并行处理帧的 worker 数（默认 1 = 串行；>1 时每个 "
                            "worker 各持一份模型 session，GPU 与显存占用随之"
                            "上升，建议 2-8）")
    video.add_argument("--profile", action="store_true",
                       help="打印 decode / swap / encode 三段的累计耗时；"
                            "并行时三段会重叠，之和不等于总用时")
    video.add_argument("--encode-thread", action="store_true",
                       help="把编码移到独立线程，与换脸推理解耦（默认关闭；"
                            "编码失败会中断命令，输出顺序不变）")
    video.add_argument("--defer-d2h", action="store_true",
                       help="把逐帧 GPU→CPU 拷贝从 worker 线程推迟到帧的"
                            "消费侧（仅 --workers > 1 时生效；不删除任何"
                            "CUDA 同步，默认关闭）")
    video.add_argument("--force-recognize", action="store_true",
                       help="保留每帧逐脸 ArcFace 推理的旧路径。默认会跳过"
                            "没有读取方的 embedding 计算（仅未指定 "
                            "--target-face 时生效，像素输出不变）")
    video.set_defaults(func=_cmd_video)

    inspect = sub.add_parser("inspect", help="列出图片里检测到的脸")
    add_common(inspect)
    inspect.add_argument("--target", required=True, help="要查看的图片")
    inspect.set_defaults(func=_cmd_inspect)

    params = sub.add_parser("params", help="打印已知 parameter 及其默认值")
    params.set_defaults(func=_cmd_params)

    download = sub.add_parser(
        "download-models",
        help="下载缺失或损坏的 ONNX 模型",
    )
    download.add_argument(
        "--model",
        action="append",
        choices=DOWNLOAD_MODEL_CHOICES,
        default=[],
        help="模型组或模型名；可重复（默认 base）",
    )
    download.add_argument(
        "--provider",
        action="append",
        default=[],
        help=_PROVIDER_HELP,
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="即使当前文件有效也重新下载",
    )
    download.add_argument(
        "--list",
        action="store_true",
        help="只列出所选模型及本地状态，不下载",
    )
    download.set_defaults(func=_cmd_download_models)

    check_models = sub.add_parser(
        "check-models",
        help="校验本地模型文件及其 SHA-256 哈希",
    )
    check_models.add_argument(
        "--model",
        action="append",
        choices=DOWNLOAD_MODEL_CHOICES,
        default=[],
        help="模型组或模型名；可重复（默认 all）",
    )
    check_models.add_argument(
        "--quick",
        action="store_true",
        help="只检查存在性、size 与已缓存的哈希",
    )
    check_models.set_defaults(func=_cmd_check_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        result = int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except HeadlessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    # 只有真正跑完换脸的命令才报总用时；失败和中途退出不打印，避免把
    # 半个任务的时间误报成一次完整运行。avg fps 只有 video 有帧数可算，
    # swap 没有，因此保持原来的单行 elapsed。
    if args.command in ("swap", "video"):
        seconds = time.monotonic() - started
        frames = getattr(args, "_processed_frames", None)
        print(f"elapsed: {_format_elapsed(seconds)}{_format_avg_fps(frames, seconds)}")
    return result


if __name__ == "__main__":
    sys.exit(main())
