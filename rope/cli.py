"""Command-line entry point for headless Rope.

Image:

    python -m rope.cli swap \
        --source face.jpg --target target.jpg --output out.png

    python -m rope.cli swap \
        --source face.jpg --target-dir ./targets --output-dir ./out

Video (one source identity, every face):

    python -m rope.cli video \
        --source face.jpg --target clip.mp4 --output out.mp4

Inspection:

    python -m rope.cli inspect --target target.jpg

Models:

    python -m rope.cli download-models
    python -m rope.cli check-models
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any

from rope.errors import HeadlessError
from rope.model_store import (
    BASE_MODELS,
    DEFAULT_PROVIDERS,
    MODEL_SPECS,
    PROVIDER_LABELS,
    ensure_models,
    format_size,
    inspect_models,
    specs_for_names,
)
from rope.parameters import PARAMETER_BY_NAME
from rope.parameters_io import load as load_params


# Image extensions the CLI accepts for source faces and target images.
# Kept here (instead of in rope.headless) so the argument parser and the
# --help text work without OpenCV / torch installed.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
}


def _headless():
    """Import rope.headless lazily.

    The heavy deps (OpenCV, torch, ONNX Runtime) are only needed once a
    command actually swaps something, so `--help` and `params` stay
    usable on a machine that has not finished installing the runtime.
    """
    try:
        from rope import headless
    except ImportError as exc:
        raise HeadlessError(
            f"missing runtime dependency: {exc}\n"
            "Install the CUDA runtime with:\n"
            "  python tools/install.py cu128"
        ) from exc
    return headless


def _collect_sources(values: list[str] | None) -> list[str]:
    """Expand --source values: files, or directories of face images."""
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
    """Expand --target values plus everything inside --target-dir."""
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
    """Collect saved parameters plus explicit CLI flags into one dict."""
    overrides: dict[str, Any] = {}
    if getattr(args, "params", None):
        overrides.update(load_params(args.params))

    # Explicit CLI flags win over the saved parameter file.
    if getattr(args, "detect_type", None):
        overrides["DetectTypeTextSel"] = args.detect_type
    if getattr(args, "detect_input_size", None):
        overrides["DetectInputSizeTextSel"] = str(args.detect_input_size)
    if getattr(args, "detect_score", None) is not None:
        overrides["DetectScoreSlider"] = args.detect_score * 100
    if getattr(args, "merge", None):
        overrides["MergeTextSel"] = args.merge
    if getattr(args, "swapper_size", None):
        overrides["SwapperTypeTextSel"] = args.swapper_size
    if getattr(args, "color_match", False):
        overrides["ColorMatchSwitch"] = True
    return overrides


def _build_options(args: argparse.Namespace, headless=None) -> "SwapOptions":
    """Turn parsed args into SwapOptions, loading saved parameters."""
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
    )
    return options


def _prepare_models(args: argparse.Namespace) -> None:
    """Validate base models and download only missing/corrupt files."""
    specs = list(BASE_MODELS)
    if getattr(args, "detect_type", None) == "SCRDF":
        scrfd = next(
            (spec for spec in specs_for_names(["scrfd"]) if spec.name == "scrfd"),
            None,
        )
        if scrfd is not None:
            specs.append(scrfd)
    ensure_models(
        args.models,
        specs,
        providers=args.model_provider or DEFAULT_PROVIDERS,
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
        print(
            f"{status:<8} {spec.local_name:<28} "
            f"{format_size(spec.size):>10}  {spec.role}"
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

    # Import the heavy runtime only once the inputs are known to be sane,
    # so a typo in a path fails fast with a clear message.
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
    """List the faces found in a target image (no swapping)."""
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
            thumb = slot["Thumbnail"]
            print(f"  [{index}] thumbnail {thumb.shape[1]}x{thumb.shape[0]}")
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    """Swap one source identity into every face of a video."""
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
        # Video CLI is deliberately single-source / all-faces only.
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
    )

    def progress(index: int, total: int) -> None:
        if total > 0:
            pct = 100.0 * index / total
            print(f"\rframe {index}/{total} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\rframe {index}", end="", flush=True)

    with headless.VideoSwapper(options) as swapper:
        report = swapper.swap(sources, args.target, args.output, on_progress=progress)
    print()
    print(
        f"done: {report['frames']} frame(s) at {report['fps']:.3f} fps, "
        f"{report['width']}x{report['height']}, "
        f"audio={'yes' if report['audio'] else 'no'} -> {report['output']}"
    )
    return 0


def _cmd_params(args: argparse.Namespace) -> int:
    """Print the known parameter names and their defaults."""
    for name, param in PARAMETER_BY_NAME.items():
        if param.kind == "button":
            continue
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
        providers=args.provider or DEFAULT_PROVIDERS,
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
    invalid = [
        spec for spec in specs
        if statuses[spec.name] != "valid"
        and (explicit_selection or spec.required)
    ]
    if invalid:
        repair_args = " ".join(
            f"--model {spec.name}" for spec in invalid
        )
        raise HeadlessError(
            f"invalid model(s): {', '.join(spec.local_name for spec in invalid)}\n"
            f"repair with: python -m rope.cli --models {args.models} "
            f"download-models {repair_args}"
        )
    if not args.quick:
        print("all model SHA-256 checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rope",
        description="Headless Rope - command-line face swapping.",
    )
    parser.add_argument(
        "--models",
        default="./models",
        help="folder holding det_10g.onnx / w600k_r50.onnx / inswapper_*.onnx",
    )
    parser.add_argument(
        "--model-provider",
        action="append",
        choices=sorted(PROVIDER_LABELS),
        default=[],
        help="model source used for automatic downloads; repeatable",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="fail instead of downloading missing/corrupt base models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--params", default=None,
                       help="saved_parameters.json to load defaults from")
        p.add_argument("--detect-type", choices=["Retinaface", "SCRDF"],
                       default=None, help="face detection model")
        p.add_argument("--detect-input-size", type=int, choices=[320, 416, 480, 640],
                       default=None, help="detector input resolution")
        p.add_argument("--detect-score", type=int, default=None,
                       help="detection score threshold, 1-100")
        p.add_argument("--merge",
                       choices=["Mean", "Median", "Sph", "Geo", "Qual"],
                       default=None, help="how to combine several source faces")
        p.add_argument("--swapper-size", choices=["128", "256", "512"],
                       default=None, help="swapper output resolution")
        p.add_argument("--color-match", action="store_true",
                       help="enable LAB color matching")
        p.add_argument("--backend", choices=["cuda", "trt"], default="cuda",
                       help="inference backend (default: cuda, no TensorRT)")

    swap = sub.add_parser("swap", help="swap faces in one or more images")
    add_common(swap)
    swap.add_argument("--source", action="append", default=[],
                       help="source face image (repeatable, or a folder)")
    swap.add_argument("--target", action="append", default=[],
                       help="target image (repeatable, or a folder)")
    swap.add_argument("--target-dir", default=None,
                       help="folder of target images to process")
    swap.add_argument("--output", default=None,
                       help="output image path (single target only)")
    swap.add_argument("--output-dir", default=None,
                       help="output folder (batch mode)")
    swap.add_argument("--face-policy", choices=["all", "largest"],
                       default="all",
                       help="which target faces to swap (default: all)")
    swap.set_defaults(func=_cmd_swap)

    video = sub.add_parser(
        "video", help="swap every face in a video to one source identity"
    )
    add_common(video)
    video.add_argument("--source", action="append", default=[],
                       help="source face image (repeatable, or a folder)")
    video.add_argument("--target-face", action="append", default=[],
                       help="reference image of the person to replace in the "
                            "video (repeatable, or a folder). Omit to swap "
                            "every detected face")
    video.add_argument("--target-threshold", type=float, default=55.0,
                       help="similarity threshold (0-100) for matching the "
                            "target person (default: 55)")
    video.add_argument("--target", required=True,
                       help="input video file")
    video.add_argument("--output", required=True,
                       help="output video file (e.g. out.mp4)")
    video.add_argument("--crf", type=int, default=18,
                       help="H.264 quality, lower is better (default: 18)")
    video.add_argument("--preset", default="medium",
                       choices=["ultrafast", "veryfast", "fast", "medium",
                                "slow", "veryslow"],
                       help="x264 speed/quality preset (default: medium)")
    video.add_argument("--no-audio", action="store_true",
                       help="do not copy the source audio track")
    video.add_argument("--every-nth", type=int, default=1,
                       help="only swap every Nth frame, pass the rest "
                            "through (1 = all frames; for timing tests)")
    video.set_defaults(func=_cmd_video)

    inspect = sub.add_parser("inspect", help="list faces detected in an image")
    add_common(inspect)
    inspect.add_argument("--target", required=True, help="image to inspect")
    inspect.set_defaults(func=_cmd_inspect)

    params = sub.add_parser("params", help="print known parameters and defaults")
    params.set_defaults(func=_cmd_params)

    download = sub.add_parser(
        "download-models",
        help="download missing or corrupt ONNX models",
    )
    download.add_argument(
        "--model",
        action="append",
        choices=["base", "retinaface", "arcface", "inswapper", "gfpgan", "scrfd"],
        default=[],
        help="model group or model name; repeatable (default: base)",
    )
    download.add_argument(
        "--provider",
        action="append",
        choices=sorted(PROVIDER_LABELS),
        default=[],
        help="download source; repeatable (default: all in order)",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="download again even when the current file is valid",
    )
    download.add_argument(
        "--list",
        action="store_true",
        help="list selected models and local status without downloading",
    )
    download.set_defaults(func=_cmd_download_models)

    check_models = sub.add_parser(
        "check-models",
        help="validate local model files and SHA-256 hashes",
    )
    check_models.add_argument(
        "--model",
        action="append",
        choices=["base", "retinaface", "arcface", "inswapper", "gfpgan", "scrfd"],
        default=[],
        help="model group or model name; repeatable (default: all)",
    )
    check_models.add_argument(
        "--quick",
        action="store_true",
        help="check existence, size, and cached hash only",
    )
    check_models.set_defaults(func=_cmd_check_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except HeadlessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
