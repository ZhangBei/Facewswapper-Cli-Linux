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
    _format_avg_fps,
    _format_profile,
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


def test_parallel_path_wiring() -> None:
    """并行帧调度必须委托给可独立测试的 frame_order 模块。

    保序行为本身由 tests/test_frame_order.py 用假 swap_fn 做行为测试；
    这里只用 AST 检查接线，防止 headless 里又留下一份旧实现。
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "headless.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "faceswapper.frame_order"
        and any(alias.name == "ordered_swap_frames" for alias in node.names)
        for node in ast.walk(tree)
    )
    called = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ordered_swap_frames"
        for node in ast.walk(tree)
    )
    if not (imported and called):
        fail("headless must delegate to frame_order.ordered_swap_frames")
    if "_parallel_swap_frames" in src:
        fail("legacy in-headless scheduler is still referenced")
    if not any(
        isinstance(node, ast.Constant) and node.value == "Per-Thread"
        for node in ast.walk(tree)
    ):
        fail("workers > 1 must switch to Per-Thread sessions")
    print("  ok  parallel path delegates to frame_order with Per-Thread sessions")


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


def test_format_avg_fps() -> None:
    """avg fps 用实际帧数 / 耗时，除零与缺帧数一律不输出。"""
    cases = [
        (4731, 154.6, " (avg 30.6 fps)"),
        (100, 10.0, " (avg 10.0 fps)"),
        (0, 10.0, ""),          # 没处理任何帧
        (None, 10.0, ""),       # swap 子命令没有帧数
        (100, 0.0, ""),         # 零耗时
        (100, -1.0, ""),        # 时钟异常
        (-5, 10.0, ""),         # 帧数异常
    ]
    for frames, seconds, expected in cases:
        got = _format_avg_fps(frames, seconds)
        if got != expected:
            fail(
                f"_format_avg_fps({frames}, {seconds}) = {got!r}, "
                f"expected {expected!r}"
            )
    print("  ok  avg fps handles zero frames / zero seconds / rounding")


def test_format_profile() -> None:
    """分段耗时按固定顺序输出，缺的段补 0.00s。"""
    got = _format_profile({"encode": 3.0, "decode": 1.234, "swap": 2.0})
    expected = "profile: decode=1.23s swap=2.00s encode=3.00s"
    if got != expected:
        fail(f"unexpected profile line: {got!r}, expected {expected!r}")
    got = _format_profile({})
    expected = "profile: decode=0.00s swap=0.00s encode=0.00s"
    if got != expected:
        fail(f"missing segments must be zeroed: {got!r}")
    print("  ok  profile line keeps segment order and defaults")


def test_video_profile_flag() -> None:
    """--profile 默认关闭，且只挂在 video 子命令上。"""
    parser = build_parser()
    plain = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
    ])
    assert plain.profile is False, plain.profile
    profiled = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
        "--profile",
    ])
    assert profiled.profile is True, profiled.profile
    print("  ok  --profile defaults off and is accepted by video")


def test_stage3_flags() -> None:
    """阶段 3 两个开关默认关闭，且 --arena-strategy 属于公共参数。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    plain = parser.parse_args(base)
    assert plain.defer_d2h is False, plain.defer_d2h
    assert plain.arena_strategy == "same", plain.arena_strategy

    tuned = parser.parse_args(base + [
        "--defer-d2h", "--arena-strategy", "pow2", "--workers", "4",
    ])
    assert tuned.defer_d2h is True, tuned.defer_d2h
    assert tuned.arena_strategy == "pow2", tuned.arena_strategy
    assert tuned.workers == 4, tuned.workers

    # add_common 参数：图片换脸也能选显存策略。
    image = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "b.jpg",
        "--output", "o.png", "--arena-strategy", "pow2",
    ])
    assert image.arena_strategy == "pow2", image.arena_strategy
    print("  ok  --defer-d2h / --arena-strategy default off and parse correctly")


def test_reuse_detect_buffers_flag() -> None:
    """阶段 6 开关默认关闭，且属于 add_common（swap 也能用）。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    plain = parser.parse_args(base)
    if plain.reuse_detect_buffers is not False:
        fail(f"reuse flag defaults to {plain.reuse_detect_buffers!r}")
    tuned = parser.parse_args(base + ["--reuse-detect-buffers"])
    if tuned.reuse_detect_buffers is not True:
        fail("--reuse-detect-buffers was not accepted by video")
    image = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "b.jpg",
        "--output", "o.png", "--reuse-detect-buffers",
    ])
    if image.reuse_detect_buffers is not True:
        fail("--reuse-detect-buffers should be a common flag")
    print("  ok  --reuse-detect-buffers defaults off and parses for video/swap")


def test_detect_buffer_reuse_wiring() -> None:
    """钉住阶段 6 的闸门与逐帧 fallback，别让复用悄悄变成默认行为。"""
    import ast

    root = Path(__file__).resolve().parents[1]

    def parse(name):
        return ast.parse((root / name).read_text(encoding="utf-8"))

    def called_attrs(nodes):
        return {
            node.func.attr
            for node in ast.walk(ast.Module(body=list(nodes), type_ignores=[]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

    models = parse("Models.py")
    headless = parse("headless.py")

    if "set_reuse_detect_buffers" not in {
        node.func.attr
        for node in ast.walk(headless)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }:
        fail("reuse switch is not wired into the swapper base")

    gates = [
        node.value.value
        for node in ast.walk(models)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "_reuse_detect_buffers"
        and isinstance(node.value, ast.Constant)
    ]
    if gates != [False]:
        fail(f"_reuse_detect_buffers default is {gates!r}, expected [False]")

    fn = next(
        (n for n in ast.walk(models)
         if isinstance(n, ast.FunctionDef)
         and n.name == "_detect_retinaface_inner"),
        None,
    )
    if fn is None:
        fail("_detect_retinaface_inner not found")
    branches = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and "_reuse_detect_buffers" in ast.dump(n.test)
    ]
    if len(branches) != 1:
        fail(f"expected exactly one reuse branch, found {len(branches)}")
    branch = branches[0]
    if "_get_retinaface_output_buffers" not in called_attrs(branch.body):
        fail("reuse branch does not call _get_retinaface_output_buffers")
    if not branch.orelse:
        fail("the per-frame allocation fallback branch is missing")
    empties = sum(
        1 for node in ast.walk(
            ast.Module(body=branch.orelse, type_ignores=[])
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "empty"
    )
    if empties < 3:
        fail(f"fallback branch has only {empties} torch.empty calls")
    print("  ok  reuse is gated off by default and keeps the per-frame fallback")


def test_zero_copy_frames_flag() -> None:
    """阶段 7 开关默认关闭，且属于 add_common（swap 也能用）。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    plain = parser.parse_args(base)
    if plain.zero_copy_frames is not False:
        fail(f"zero-copy flag defaults to {plain.zero_copy_frames!r}")
    tuned = parser.parse_args(base + ["--zero-copy-frames"])
    if tuned.zero_copy_frames is not True:
        fail("--zero-copy-frames was not accepted by video")
    image = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "b.jpg",
        "--output", "o.png", "--zero-copy-frames",
    ])
    if image.zero_copy_frames is not True:
        fail("--zero-copy-frames should be a common flag")
    print("  ok  --zero-copy-frames defaults off and parses for video/swap")


def test_zero_copy_frames_wiring() -> None:
    """钉住阶段 7 的闸门与旧路径：关掉时帧归一化表达式逐字不变。

    H2D（VideoManager）与 D2H / 图片（headless）两侧都要检查：只要哪一侧
    把 astype 直接删了，默认路径的行为就不再是基线，回滚也就没有意义。
    """
    import ast

    root = Path(__file__).resolve().parents[1]

    def parse(name):
        return ast.parse((root / name).read_text(encoding="utf-8"))

    def module(nodes):
        return ast.Module(body=list(nodes), type_ignores=[])

    def defaults(tree, field):
        values = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.AnnAssign)
                    and isinstance(node.value, ast.Constant)):
                continue
            target = node.target
            name = (
                target.id if isinstance(target, ast.Name)
                else target.attr if isinstance(target, ast.Attribute)
                else None
            )
            if name == field:
                values.append(node.value.value)
        return values

    def imports(tree, name):
        return any(
            isinstance(node, ast.ImportFrom)
            and node.module == "faceswapper.frame_upload"
            and any(alias.name == name for alias in node.names)
            for node in ast.walk(tree)
        )

    def is_astype_uint8(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
            and any(
                isinstance(arg, ast.Constant) and arg.value == "uint8"
                for arg in node.args
            )
        )

    headless = parse("headless.py")
    vm = parse("VideoManager.py")

    if defaults(headless, "zero_copy_frames") != [False]:
        fail("SwapOptions.zero_copy_frames must default to False")
    if defaults(vm, "zero_copy_frames") != [False]:
        fail("VideoManager.zero_copy_frames must default to False")
    if not (imports(headless, "normalize_plan")
            and imports(vm, "normalize_plan")):
        fail("both frame paths must delegate to frame_upload.normalize_plan")

    helper = next(
        (n for n in ast.walk(headless)
         if isinstance(n, ast.FunctionDef) and n.name == "_frame_to_uint8"),
        None,
    )
    if helper is None:
        fail("_frame_to_uint8 helper is missing")
    gates = [
        n for n in ast.walk(helper)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.UnaryOp)
        and isinstance(n.test.op, ast.Not)
        and isinstance(n.test.operand, ast.Name)
        and n.test.operand.id == "zero_copy"
    ]
    if len(gates) != 1:
        fail(f"expected one closed-gate branch, found {len(gates)}")
    returns = [n.value for n in gates[0].body if isinstance(n, ast.Return)]
    if len(returns) != 1 or not is_astype_uint8(returns[0]):
        fail("closed gate must keep rgb.astype('uint8') verbatim")

    fn = next(
        (n for n in ast.walk(vm)
         if isinstance(n, ast.FunctionDef) and n.name == "_swap_video_inner"),
        None,
    )
    if fn is None:
        fail("_swap_video_inner not found")
    branches = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and "zero_copy_frames" in ast.dump(n.test)
    ]
    if len(branches) != 1:
        fail(f"expected one zero-copy H2D branch, found {len(branches)}")
    branch = branches[0]
    body_calls = {
        n.func.id for n in ast.walk(module(branch.body))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    if "normalize_plan" not in body_calls:
        fail("zero-copy H2D branch does not consult normalize_plan")
    if not branch.orelse:
        fail("the original astype H2D fallback is missing")
    fallback_values = [
        n.value for n in ast.walk(module(branch.orelse))
        if isinstance(n, ast.Assign)
    ]
    if not any(is_astype_uint8(value) for value in fallback_values):
        fail("H2D fallback must keep target_image.astype('uint8') verbatim")

    wired = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "zero_copy_frames"
            and isinstance(t.value, ast.Attribute)
            and t.value.attr == "vm"
            for t in node.targets
        )
        for node in ast.walk(headless)
    )
    if not wired:
        fail("SwapOptions.zero_copy_frames is not wired into VideoManager")
    print("  ok  zero-copy is gated off and keeps both astype fallbacks")


def test_video_encoder_flag() -> None:
    """--encoder 默认 libx264，nvenc 需显式选择。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    assert parser.parse_args(base).encoder == "libx264"
    assert parser.parse_args(base + ["--encoder", "nvenc"]).encoder == "nvenc"
    print("  ok  --encoder defaults to libx264 and accepts nvenc")


def test_force_recognize_flag() -> None:
    """--force-recognize 默认关闭，是逐脸识别路径的回滚开关。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    assert parser.parse_args(base).force_recognize is False
    assert parser.parse_args(
        base + ["--force-recognize"]
    ).force_recognize is True
    print("  ok  --force-recognize defaults off and restores the old path")


def test_recognize_wiring() -> None:
    """跳过判定必须委托给可测试的 recognize.py，且 face_emb 有 None 兜底。"""
    import ast

    src = (Path(__file__).resolve().parents[1] / "VideoManager.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "faceswapper.recognize"
        and any(alias.name == "should_skip_recognize" for alias in node.names)
        for node in ast.walk(tree)
    )
    called = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "should_skip_recognize"
        for node in ast.walk(tree)
    )
    if not (imported and called):
        fail("VideoManager must delegate the skip decision to recognize.py")
    has_none_default = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "face_emb"
                for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for node in ast.walk(tree)
    )
    if not has_none_default:
        fail("skipped faces must still carry a face_emb placeholder")
    called_attrs = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if "run_recognize" not in called_attrs:
        fail("the recognize path was removed entirely instead of gated")
    print("  ok  recognize policy is delegated, gated and keeps the old path")


def test_sync_red_lines_preserved() -> None:
    """阶段 3 的红线：不能删 syncvec drain、wait_stream 或 stream 绑定。

    用 AST 而不是普通子串匹配，避免注释里的字样把检查骗过去。
    """
    import ast

    root = Path(__file__).resolve().parents[1]

    def parse(name):
        return ast.parse((root / name).read_text(encoding="utf-8"))

    def called_attrs(tree):
        return {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

    models = parse("Models.py")
    syncvec_drains = sum(
        1 for node in ast.walk(models)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cpu"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "syncvec"
    )
    if syncvec_drains < 1:
        fail("Models.syncvec.cpu() drain was removed (red line)")
    if "user_compute_stream" not in {
        node.value for node in ast.walk(models)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }:
        fail("user_compute_stream binding was removed (red line)")
    if "set_arena_strategy" not in called_attrs(parse("headless.py")):
        fail("arena strategy is not wired into the swapper base")
    if not any(
        isinstance(node, ast.Attribute) and node.attr == "_arena_strategy"
        for node in ast.walk(models)
    ):
        fail("_cuda_ep_provider_options still hardcodes the arena strategy")

    vm = parse("VideoManager.py")
    if "wait_stream" not in called_attrs(vm):
        fail("default_stream.wait_stream() was removed (red line)")
    print("  ok  CUDA sync drains / stream binding / arena wiring are intact")


def test_video_encode_thread_flag() -> None:
    """--encode-thread 默认关闭，只挂在 video 子命令上。"""
    parser = build_parser()
    plain = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
    ])
    assert plain.encode_thread is False, plain.encode_thread
    threaded = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
        "--encode-thread",
    ])
    assert threaded.encode_thread is True, threaded.encode_thread
    # 与其它 video 开关可以叠加，不互相覆盖。
    both = parser.parse_args([
        "video", "--source", "face.jpg",
        "--target", "in.mp4", "--output", "out.mp4",
        "--encode-thread", "--workers", "4", "--profile",
    ])
    assert both.workers == 4 and both.profile is True, both
    print("  ok  --encode-thread defaults off and composes with other flags")


def test_main_reports_avg_fps() -> None:
    """main() 只在 video 成功结束后把 avg fps 拼到 elapsed 行上。"""
    import argparse
    import contextlib
    import io
    from unittest import mock

    import faceswapper.cli as cli

    def run(command: str, func) -> tuple[int, str]:
        parser = mock.Mock()
        parser.parse_args.return_value = argparse.Namespace(
            command=command, func=func,
        )
        buf = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(cli, "build_parser", return_value=parser), \
                contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(err):
            rc = cli.main([])
        return rc, buf.getvalue()

    def fake_video(args) -> int:
        args._processed_frames = 100
        return 0

    rc, out = run("video", fake_video)
    if rc != 0 or "elapsed:" not in out or "(avg " not in out:
        fail(f"video should report avg fps, got rc={rc} out={out!r}")

    rc, out = run("swap", lambda args: 0)
    if rc != 0 or "elapsed:" not in out or "avg " in out:
        fail(f"swap must not report avg fps, got rc={rc} out={out!r}")

    def failed_video(args) -> int:
        raise HeadlessError("boom")

    rc, out = run("video", failed_video)
    if rc != 2 or "elapsed:" in out:
        fail(f"a failed run must not report elapsed, got rc={rc} out={out!r}")
    print("  ok  elapsed line carries avg fps for video only, never on failure")


def test_auto_orient_flag() -> None:
    """整帧朝向兜底默认开启，只有 --no-auto-orient 能关；属于 add_common。"""
    parser = build_parser()
    base = ["video", "--source", "face.jpg", "--target", "in.mp4",
            "--output", "out.mp4"]
    plain = parser.parse_args(base)
    if plain.no_auto_orient is not False:
        fail(f"no-auto-orient defaults to {plain.no_auto_orient!r}")
    tuned = parser.parse_args(base + ["--no-auto-orient"])
    if tuned.no_auto_orient is not True:
        fail("--no-auto-orient was not accepted by video")
    image = parser.parse_args([
        "swap", "--source", "a.jpg", "--target", "b.jpg",
        "--output", "o.png", "--no-auto-orient",
    ])
    if image.no_auto_orient is not True:
        fail("--no-auto-orient should be a common flag")

    # 它走 options 字段，不写进 parameter dict——saved_parameters.json 的
    # 磁盘契约与 GUI 时代的 OrientAutoSwitch 保持原样。
    for args in (plain, tuned):
        overrides = _parameter_overrides(args)
        if "OrientAutoSwitch" in overrides or "OrientSwitch" in overrides:
            fail(f"fallback must not touch GUI orient parameters: {overrides!r}")
    from faceswapper.parameters import default_values
    if default_values().get("OrientAutoSwitch") is not False:
        fail("GUI OrientAutoSwitch must stay off by default")
    print("  ok  --no-auto-orient defaults off (fallback on) for video/swap")


def test_auto_orient_wiring() -> None:
    """钉住兜底的默认开启、接线点，以及"不用跨帧 EMA"这条底线。"""
    import ast

    root = Path(__file__).resolve().parents[1]
    vm = (root / "VideoManager.py").read_text(encoding="utf-8")
    headless = (root / "headless.py").read_text(encoding="utf-8")
    cli = (root / "cli.py").read_text(encoding="utf-8")

    # VM 的默认值是 True（AnnAssign 常量），不是靠 --params 打开。
    vals = [
        n.value.value
        for n in ast.walk(ast.parse(vm))
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.target, ast.Attribute)
        and n.target.attr == "auto_orient_fallback"
    ]
    if vals != [True]:
        fail(f"auto_orient_fallback must default to True, got {vals!r}")

    # 三处接线缺一不可：VM 用纯逻辑决策、headless 注入回滚值、CLI 两面传参。
    for needle in (
        "from faceswapper.orientation import",
        "should_probe(",
        "choose_angle(counts)",
        "PROBE_ANGLES",
    ):
        if needle not in vm:
            fail(f"VideoManager.py missing {needle!r}")
    if "self.vm.auto_orient_fallback = not bool(" not in headless:
        fail("headless.py does not inject the fallback flag")
    if 'getattr(options, "no_auto_orient", False)' not in headless:
        fail("headless.py does not read options.no_auto_orient")
    if cli.count("no_auto_orient=args.no_auto_orient") != 2:
        fail("--no-auto-orient must reach both SwapOptions and VideoOptions")

    # 兜底分支只许用 90 倍数常量，不许碰 GUI 的跨帧 EMA 状态——否则
    # workers>1 竞态与 expand=True 尺寸漂移会一起回来。
    branch = vm.split("if should_probe(", 1)[-1].split("漏检重试", 1)[0]
    if "orient_fine_angle" in branch or "detected_orient_angle" in branch:
        fail("fallback branch must not touch the GUI cross-frame orient state")
    if "applied_angle = float(fallback_angle)" not in branch:
        fail("fallback branch must set applied_angle so the frame is rotated back")
    print("  ok  auto-orient fallback wired, stateless, quarter-turns only")


def test_all_new_video_flags_compose() -> None:
    """所有新开关同时给出时不能互相覆盖，也不能被 argparse 判成歧义。"""
    parser = build_parser()
    args = parser.parse_args([
        "--models", "./models",
        "video", "--source", "face.jpg", "--target", "in.mp4",
        "--output", "out.mp4",
        "--workers", "4", "--profile", "--encode-thread", "--defer-d2h",
        "--force-recognize", "--encoder", "nvenc", "--decoder", "gpu",
        "--arena-strategy", "pow2", "--reuse-detect-buffers",
        "--zero-copy-frames", "--no-auto-orient",
        "--crf", "23", "--preset", "fast",
        "--every-nth", "2", "--no-audio", "--backend", "cuda",
    ])
    expected = {
        "workers": 4, "profile": True, "encode_thread": True,
        "defer_d2h": True, "force_recognize": True, "encoder": "nvenc",
        "decoder": "gpu", "arena_strategy": "pow2", "crf": 23, "preset": "fast",
        "every_nth": 2,         "no_audio": True, "backend": "cuda",
        "reuse_detect_buffers": True, "zero_copy_frames": True,
        "no_auto_orient": True,
    }
    for key, value in expected.items():
        got = getattr(args, key)
        if got != value:
            fail(f"{key} = {got!r}, expected {value!r}")
    print("  ok  all new video switches parse together without clobbering")


def test_switch_defaults_match_baseline() -> None:
    """不传开关时所有 CLI 默认值都回到基线。

    两处默认生效的行为变化钉在这里，避免误以为"全部默认关闭"：
    - ``--force-recognize=False``：无读取方的逐脸 ArcFace 推理被跳过
      （像素输出不变，ORT 调用次数变了）。
    - ``--no-auto-orient=False``：整帧朝向兜底默认开启，但只在第一遍
      0 检出的帧上追加推理；正常帧零额外开销、输出逐像素不变。
    """
    parser = build_parser()
    args = parser.parse_args([
        "video", "--source", "face.jpg", "--target", "in.mp4",
        "--output", "out.mp4",
    ])
    defaults = {
        "workers": 1, "profile": False, "encode_thread": False,
        "defer_d2h": False, "force_recognize": False,
        "encoder": "libx264", "decoder": "pyav",
        "arena_strategy": "same", "reuse_detect_buffers": False,
        "zero_copy_frames": False, "no_auto_orient": False,
        "crf": 18, "preset": "medium", "every_nth": 1,
        "no_audio": False, "backend": "cuda",
    }
    for key, value in defaults.items():
        got = getattr(args, key)
        if got != value:
            fail(f"default {key} = {got!r}, expected baseline {value!r}")
    print("  ok  switch defaults match baseline (two default-on changes documented)")


def test_docs_stay_in_sync() -> None:
    """README 必须提到每个 video flag；本地 CLAUDE.md 要列全测试模块。

    这是收尾防漂移检查：新增开关却忘了写文档，比代码本身出错更难发现。
    CLAUDE.md 被 .gitignore 忽略、克隆环境里可能不存在，所以只在存在时查。
    """
    root = Path(__file__).resolve().parents[2]
    parser = build_parser()
    sub = parser._subparsers._group_actions[0]
    video = sub.choices["video"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for action in video._actions:
        for opt in action.option_strings:
            if opt in ("-h", "--help"):
                continue
            if opt not in readme:
                fail(f"{opt} is missing from README.md")

    claude_path = root / "CLAUDE.md"
    if claude_path.exists():
        claude = claude_path.read_text(encoding="utf-8")
        for path in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
            dotted = f"faceswapper.tests.{path.stem}"
            if dotted not in claude:
                fail(f"{dotted} is missing from CLAUDE.md")
    print("  ok  README covers every video flag and the test list is complete")


def test_headless_module_syntax() -> None:
    """headless.py 需要 GPU 栈才能 import，但改动后语法必须仍可解析。

    这里只做 ast.parse，不触发任何第三方依赖，用来兜住 _encode_video 这类
    本地无法实际执行的核心循环改动。
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "headless.py").read_text(
        encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        fail(f"headless.py has a syntax error: {exc}")
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in (
        "_encode_video", "_encode_via_thread", "_swap_frame",
        "_finalize_frame",
    ):
        if required not in names:
            fail(f"headless.py is missing {required}()")

    # CLI 会把这些 flag 传进 dataclass；字段缺失要到运行时才 TypeError，
    # 这里用 AST 提前兜住。
    classes = {
        node.name: node for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    expected_fields = {
        "SwapOptions": (
            "arena_strategy", "reuse_detect_buffers", "zero_copy_frames",
        ),
        "VideoOptions": (
            "profile", "encode_thread", "defer_d2h", "force_recognize",
            "encoder", "decoder", "workers",
        ),
    }
    for cls_name, required in expected_fields.items():
        cls = classes.get(cls_name)
        if cls is None:
            fail(f"headless.py is missing {cls_name}")
        fields = {
            node.target.id
            for node in cls.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }
        for field in required:
            if field not in fields:
                fail(f"{cls_name} is missing field {field!r}")
    print("  ok  headless.py parses and keeps the frame-pipeline wiring")


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
    test_parallel_path_wiring()
    test_format_elapsed()
    test_format_avg_fps()
    test_format_profile()
    test_video_profile_flag()
    test_video_encode_thread_flag()
    test_stage3_flags()
    test_reuse_detect_buffers_flag()
    test_detect_buffer_reuse_wiring()
    test_zero_copy_frames_flag()
    test_zero_copy_frames_wiring()
    test_force_recognize_flag()
    test_video_encoder_flag()
    test_recognize_wiring()
    test_sync_red_lines_preserved()
    test_main_reports_avg_fps()
    test_headless_module_syntax()
    test_auto_orient_flag()
    test_auto_orient_wiring()
    test_all_new_video_flags_compose()
    test_switch_defaults_match_baseline()
    test_docs_stay_in_sync()
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
