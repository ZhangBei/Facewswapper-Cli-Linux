"""Tests for the per-frame ArcFace recognize policy.

The policy is stdlib-only, so these are real behaviour tests: no GPU, no
torch, no OpenCV, no ONNX Runtime.

    python -m faceswapper.tests.test_recognize
"""

from __future__ import annotations

import sys

from faceswapper.recognize import should_skip_recognize


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


# 用一个普通对象冒充 embedding；判定只看有没有读取方，不看它的内容。
SOURCE = object()
TARGET = object()


def test_headless_video_without_target_face_skips() -> None:
    """headless video、无 --target-face：embedding 无读取方，可以跳过。"""
    if not should_skip_recognize(force_source=SOURCE, targets=None):
        fail("headless video with no target-face should skip recognition")
    if not should_skip_recognize(force_source=SOURCE, targets=[]):
        fail("an empty target list is the same as no target-face")
    print("  ok  unused per-face embeddings are skipped")


def test_paths_that_read_embeddings_never_skip() -> None:
    """图片/GUI 匹配路径与 --target-face 筛选都要真读 embedding。"""
    cases = [
        ("image/gui path", dict(force_source=None, targets=None)),
        ("image/gui with targets", dict(force_source=None, targets=[TARGET])),
        ("target-face filter", dict(force_source=SOURCE, targets=[TARGET])),
        ("multiple references", dict(force_source=SOURCE,
                                     targets=[TARGET, TARGET])),
    ]
    for label, kwargs in cases:
        if should_skip_recognize(**kwargs):
            fail(f"{label} must still compute per-face embeddings")
    print("  ok  every path that reads embeddings keeps recognition")


def test_opt_out_always_wins() -> None:
    """--force-recognize 是回滚开关，任何组合下都必须保留旧路径。"""
    combos = [
        dict(force_source=SOURCE, targets=None),
        dict(force_source=SOURCE, targets=[TARGET]),
        dict(force_source=None, targets=None),
    ]
    for kwargs in combos:
        if should_skip_recognize(**kwargs, opt_out=True):
            fail(f"opt_out must disable skipping for {kwargs!r}")
    print("  ok  --force-recognize always restores the old path")


def test_truthiness_matches_cli_semantics() -> None:
    """target_face_embeddings 在 VM 里是 None 或 list，空容器必须视为无筛选。"""
    for empty in (None, [], (), {}):
        if not should_skip_recognize(force_source=SOURCE, targets=empty):
            fail(f"empty targets {empty!r} should not force recognition")
    if should_skip_recognize(force_source=SOURCE, targets=[TARGET]):
        fail("a non-empty target list must force recognition")
    print("  ok  empty vs non-empty target collections behave correctly")


def main() -> int:
    print("faceswapper recognize policy tests")
    test_headless_video_without_target_face_skips()
    test_paths_that_read_embeddings_never_skip()
    test_opt_out_always_wins()
    test_truthiness_matches_cli_semantics()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
