"""Tests for target-person matching (no GPU / models needed).

Exercises the similarity filter used by video mode to decide whether a
detected face belongs to the person the user asked to replace.

    python -m faceswapper.tests.test_target_match
"""

from __future__ import annotations

import sys

import numpy as np


def _cosine_similarity_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Copy of faceswapper.headless._cosine_similarity_pct.

    Duplicated here so this test runs without OpenCV / torch installed.
    If the formula in faceswapper/headless.py changes, update it here too.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cos_dist = 1.0 - float(np.dot(a, b)) / denom
    return 100.0 - cos_dist * 50.0


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_identical_is_100() -> None:
    v = np.array([0.3, -0.2, 0.9], dtype=np.float32)
    score = _cosine_similarity_pct(v, v)
    if abs(score - 100.0) > 1e-4:
        fail(f"identical vectors scored {score}, expected 100")
    print("  ok  identical embeddings score 100")


def test_orthogonal_is_50() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    score = _cosine_similarity_pct(a, b)
    if abs(score - 50.0) > 1e-4:
        fail(f"orthogonal vectors scored {score}, expected 50")
    print("  ok  orthogonal embeddings score 50")


def test_opposite_is_0() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    score = _cosine_similarity_pct(a, b)
    if abs(score - 0.0) > 1e-4:
        fail(f"opposite vectors scored {score}, expected 0")
    print("  ok  opposite embeddings score 0")


def test_zero_vector_is_safe() -> None:
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    score = _cosine_similarity_pct(a, b)
    if score != 0.0:
        fail(f"zero vector scored {score}, expected 0")
    print("  ok  zero embedding does not divide by zero")


def test_threshold_separates_identities() -> None:
    """A 55 threshold must accept the same identity and reject a different one."""
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    same_person = np.array([0.98, 0.19, 0.05], dtype=np.float32)
    other_person = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    same_score = _cosine_similarity_pct(same_person, target)
    other_score = _cosine_similarity_pct(other_person, target)

    if same_score < 55.0:
        fail(f"same identity scored {same_score:.2f}, should pass 55")
    if other_score >= 55.0:
        fail(f"different identity scored {other_score:.2f}, should fail 55")
    print(
        f"  ok  threshold separates identities "
        f"(same={same_score:.1f}, other={other_score:.1f})"
    )


def main() -> int:
    print("faceswapper target-person matching tests")
    test_identical_is_100()
    test_orthogonal_is_50()
    test_opposite_is_0()
    test_zero_vector_is_safe()
    test_threshold_separates_identities()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

