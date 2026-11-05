"""Tests for the headless CLI plumbing.

These run without a GPU, without ONNX models and without OpenCV, so they
can be used as a quick smoke test after a fresh clone:

    python -m faceswapper.tests.test_cli
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from faceswapper.cli import (
    VIDEO_EXTENSIONS,
    _collect_sources,
    _collect_targets,
    _parameter_overrides,
    build_parser,
)
from faceswapper.errors import HeadlessError
from faceswapper.parameters import PARAMETER_BY_NAME, default_values
from faceswapper.parameters_io import _coerce
from faceswapper.parameters_io import load as load_params
from faceswapper.parameters_io import save as save_params


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_parser_subcommands() -> None:
    parser = build_parser()
    actions = parser._subparsers._group_actions[0]
    for name in (
        "swap", "video", "inspect", "params",
        "download-models", "check-models",
    ):
        if name not in actions.choices:
            fail(f"missing subcommand {name!r}")
    print(
        "  ok  parser exposes swap / video / inspect / params / "
        "download-models / check-models"
    )


def test_parser_accepts_documented_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--models", "/data/models",
        "--model-provider", "hf-mirror",
        "--no-auto-download",
        "swap",
        "--source", "a.jpg",
        "--source", "b.jpg",
        "--target", "t.jpg",
        "--output", "o.png",
        "--face-policy", "largest",
        "--backend", "cuda",
        "--params", "saved_parameters.json",
    ])
    assert args.models == "/data/models", args.models
    assert args.model_provider == ["hf-mirror"], args.model_provider
    assert args.no_auto_download is True, args.no_auto_download
    assert args.source == ["a.jpg", "b.jpg"], args.source
    assert args.target == ["t.jpg"], args.target
    assert args.output == "o.png", args.output
    assert args.face_policy == "largest", args.face_policy
    assert args.params == "saved_parameters.json", args.params
    print("  ok  parser accepts the documented flags")


def test_collect_expands_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        faces = root / "faces"
        targets = root / "targets"
        faces.mkdir()
        targets.mkdir()
        (faces / "a.jpg").write_bytes(b"x")
        (faces / "b.png").write_bytes(b"x")
        (faces / "notes.txt").write_bytes(b"x")
        (targets / "t1.jpg").write_bytes(b"x")
        (targets / "t2.webp").write_bytes(b"x")
        (targets / "clip.mp4").write_bytes(b"x")

        sources = _collect_sources([str(faces)])
        assert len(sources) == 2, sources
        assert all(s.endswith((".jpg", ".png")) for s in sources), sources

        tgts = _collect_targets([], str(targets))
        assert len(tgts) == 2, tgts
        assert not any(t.endswith(".mp4") for t in tgts), tgts
    print("  ok  folder expansion keeps images and drops other files")


def test_video_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--models", "/data/models",
        "video",
        "--source", "face.jpg",
        "--target-face", "person_a.jpg",
        "--target-threshold", "60",
        "--target", "in.mp4",
        "--output", "out.mp4",
        "--crf", "20",
        "--preset", "fast",
        "--every-nth", "2",
        "--no-audio",
    ])
    assert args.command == "video", args.command
    assert args.target_face == ["person_a.jpg"], args.target_face
    assert args.target_threshold == 60.0, args.target_threshold
    assert args.target == "in.mp4", args.target
    assert args.output == "out.mp4", args.output
    assert args.crf == 20, args.crf
    assert args.preset == "fast", args.preset
    assert args.every_nth == 2, args.every_nth
    assert args.no_audio is True, args.no_audio
    print("  ok  video subcommand accepts its flags")


def test_video_workers_flag() -> None:
    """--workers 默认串行；>1 时启用多 worker 并行帧处理。"""
    parser = build_parser()
    plain = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
    ])
    assert plain.workers == 1, plain.workers

    parallel = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
        "--workers", "4",
    ])
    assert parallel.workers == 4, parallel.workers
    print("  ok  --workers defaults to 1 and accepts a parallel value")


def test_parallel_path_preserves_order() -> None:
    """并行路径必须按解码顺序消费 future，并启用 Per-Thread session。"""
    src = (Path(__file__).resolve().parents[1] / "headless.py").read_text(
        encoding="utf-8")
    body = src[src.index("def _parallel_swap_frames"):src.index("def _swap_frame")]
    if "popleft()" not in body or "f.result()" not in body:
        fail("parallel frame path must drain futures in decode order")
    if 'set_model_session_mode("Per-Thread")' not in src:
        fail("workers > 1 must switch to Per-Thread sessions")
    print("  ok  parallel frames keep decode order with Per-Thread sessions")


def test_format_elapsed() -> None:
    """完成时的总用时要覆盖秒、分、小时且不能出现 60.0 秒这种进位错误。"""
    from faceswapper.cli import _format_elapsed

    cases = [
        (0.0, "00:00:00.0"),
        (5.26, "00:00:05.3"),
        (154.6, "00:02:34.6"),
        (3661.5, "01:01:01.5"),
        (3599.99, "01:00:00.0"),
    ]
    for seconds, expected in cases:
        got = _format_elapsed(seconds)
        if got != expected:
            fail(f"_format_elapsed({seconds}) = {got!r}, expected {expected!r}")
    print("  ok  elapsed time is formatted as HH:MM:SS.s")


def test_model_command_flags() -> None:
    parser = build_parser()
    download_args = parser.parse_args([
        "--models", "/data/models",
        "download-models",
        "--model", "inswapper",
        "--provider", "hf-mirror",
        "--force",
    ])
    assert download_args.model == ["inswapper"], download_args.model
    assert download_args.provider == ["hf-mirror"], download_args.provider
    assert download_args.force is True, download_args.force

    check_args = parser.parse_args([
        "--models", "/data/models",
        "check-models",
        "--model", "base",
        "--quick",
    ])
    assert check_args.model == ["base"], check_args.model
    assert check_args.quick is True, check_args.quick
    print("  ok  model download and integrity check flags are parsed")


def test_video_extensions_cover_common_containers() -> None:
    for ext in (".mp4", ".mkv", ".mov", ".webm"):
        if ext not in VIDEO_EXTENSIONS:
            fail(f"{ext} missing from VIDEO_EXTENSIONS")
    print(f"  ok  {len(VIDEO_EXTENSIONS)} video container extensions known")


def test_feature_flags_map_to_parameters() -> None:
    """The feature flags set the switches the model resolver reads."""
    parser = build_parser()
    args = parser.parse_args([
        "--models", "/data/models", "swap",
        "--source", "a.jpg", "--target", "t.jpg", "--output", "o.png",
        "--restorer", "CF",
        "--occluder", "--xseg", "--face-parser",
    ])
    overrides = _parameter_overrides(args)
    assert overrides["RestorerSwitch"] is True, overrides
    assert overrides["RestorerTypeTextSel"] == "CF", overrides
    assert overrides["OccluderSwitch"] is True, overrides
    assert overrides["DFLXSegSwitch"] is True, overrides
    assert overrides["FaceParserSwitch"] is True, overrides

    # --restorer with no value defaults to GFPGAN.
    bare = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "t.jpg",
        "--restorer",
    ])
    bare_overrides = _parameter_overrides(bare)
    assert bare_overrides["RestorerTypeTextSel"] == "GFPGAN", bare_overrides

    # Nothing enabled means no optional model switch is set at all.
    plain = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "t.jpg",
    ])
    assert _parameter_overrides(plain) == {}, _parameter_overrides(plain)
    print("  ok  feature flags map onto the parameter switches")


def test_swapper_model_flag_maps_to_parameter() -> None:
    """--swapper-model selects both the model and the output resolution, so
    it lands on the single SwapperTypeTextSel parameter the pipeline reads."""
    parser = build_parser()
    args = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "t.jpg", "--output", "o.png",
        "--swapper-model", "hyperswap-256",
    ])
    assert args.swapper_model == "hyperswap-256", args.swapper_model
    overrides = _parameter_overrides(args)
    assert overrides["SwapperTypeTextSel"] == "hyperswap-256", overrides

    # The old --swapper-size flag is gone, not aliased.
    try:
        parser.parse_args([
            "swap", "--source", "a.jpg", "--target", "t.jpg",
            "--swapper-size", "128",
        ])
    except SystemExit:
        pass
    else:
        fail("--swapper-size should no longer parse")
    print("  ok  --swapper-model maps to SwapperTypeTextSel")


def test_parameters_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved_parameters.json"
        save_params(
            {"RestorerSwitch": True, "BogusKey": 1, "RestorerSlider": 42},
            path,
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "BogusKey" not in raw, raw
        loaded = load_params(path)
        assert loaded["RestorerSwitch"] is True, loaded
        assert loaded["RestorerSlider"] == 42.0, loaded
    print("  ok  saved_parameters.json roundtrip")


def test_legacy_swapper_values_are_mapped() -> None:
    """A saved_parameters.json written before the rename carries bare
    resolutions ('256', '512'); those must keep selecting the same model
    they did before, not silently fall back to the default."""
    for legacy, expected in (
        ("128", "inswapper-128"),
        ("256", "inswapper-256"),
        ("512", "inswapper-512"),
    ):
        got = _coerce("SwapperTypeTextSel", legacy)
        if got != expected:
            fail(f"legacy {legacy!r} mapped to {got!r}, expected {expected!r}")
    # An unknown value still falls back to the default.
    if _coerce("SwapperTypeTextSel", "nonsense") != "inswapper-128":
        fail("unknown swapper value should fall back to the default")
    print("  ok  legacy swapper values map onto the renamed modes")


def test_default_snapshots_have_vm_keys() -> None:
    """The VideoManager reads a fixed set of keys; the defaults must
    contain all of them or the first swap raises KeyError."""
    # VideoManager seeds self.parameters from default_values(), so this is
    # exactly the key set swap_core indexes.
    params = default_values()
    required = [
        "DetectTypeTextSel",
        "DetectScoreSlider",
        "DetectInputSizeTextSel",
        "ThresholdSlider",
        "SwapperTypeTextSel",
    ]
    for key in required:
        if key not in params:
            fail(f"parameter default missing {key!r}")
    # MergeTextSel was removed: headless passes merge_mode straight to
    # EmbeddingMerge.combine(), so the parameter was never read by the
    # pipeline. Guard against it creeping back as a phantom knob.
    if "MergeTextSel" in params:
        fail("MergeTextSel is back in the schema but nothing reads it")
    if len(PARAMETER_BY_NAME) == 0:
        fail("no parameters registered")
    print("  ok  default snapshots carry the keys VideoManager needs")


def test_headless_error_is_raisable() -> None:
    try:
        raise HeadlessError("nope")
    except HeadlessError:
        pass
    print("  ok  HeadlessError is importable and raisable")


def main() -> int:
    print("faceswapper headless CLI tests")
    test_parser_subcommands()
    test_parser_accepts_documented_flags()
    test_collect_expands_folders()
    test_video_parser_flags()
    test_video_workers_flag()
    test_parallel_path_preserves_order()
    test_format_elapsed()
    test_model_command_flags()
    test_video_extensions_cover_common_containers()
    test_feature_flags_map_to_parameters()
    test_swapper_model_flag_maps_to_parameter()
    test_parameters_roundtrip()
    test_legacy_swapper_values_are_mapped()
    test_default_snapshots_have_vm_keys()
    test_headless_error_is_raisable()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
