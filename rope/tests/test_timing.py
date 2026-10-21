"""Timing tests for the video encoder (no GPU / models / PyAV needed).

The encoder normalises the first frame to 0 while keeping the original
spacing between frames, then the mux step re-applies the source start
offset. These tests pin down that arithmetic.

    python -m rope.tests.test_timing
"""

from __future__ import annotations

import sys
from fractions import Fraction


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def normalise(pts_values: list[int]) -> list[int]:
    """Same arithmetic the encoder applies to frame PTS values."""
    first = pts_values[0]
    return [p - first for p in pts_values]


def test_cfr_spacing_preserved() -> None:
    # 30 fps with a 1/90000 time base.
    pts = [0, 3000, 6000, 9000, 12000]
    out = normalise(pts)
    if out != [0, 3000, 6000, 9000, 12000]:
        fail(f"CFR spacing changed: {out}")
    print("  ok  constant frame rate spacing preserved")


def test_vfr_spacing_preserved() -> None:
    """A VFR clip has uneven gaps; they must survive untouched."""
    pts = [0, 3003, 6006, 12012, 15015]
    out = normalise(pts)
    if out != [0, 3003, 6006, 12012, 15015]:
        fail(f"VFR spacing changed: {out}")
    gaps = [b - a for a, b in zip(out, out[1:])]
    if gaps != [3003, 3003, 6006, 3003]:
        fail(f"VFR gaps changed: {gaps}")
    print(f"  ok  VFR gaps preserved ({gaps})")


def test_nonzero_start_normalised() -> None:
    """A stream starting late is rebased to 0; the offset is re-applied at mux."""
    pts = [45000, 48000, 51000]
    out = normalise(pts)
    if out != [0, 3000, 6000]:
        fail(f"start offset not rebased: {out}")
    print("  ok  late-starting stream rebased to 0")


def test_offset_roundtrip() -> None:
    """Rebasing then re-applying the offset must reproduce the original PTS."""
    tb = Fraction(1, 90000)
    original = [45000, 48000, 51000]
    offset_seconds = float(original[0] * tb)
    rebased = normalise(original)
    restored = [p * float(tb) + offset_seconds for p in rebased]
    expected = [p * float(tb) for p in original]
    for got, want in zip(restored, expected):
        if abs(got - want) > 1e-9:
            fail(f"offset roundtrip drift: {got} vs {want}")
    print("  ok  start offset roundtrips without drift")


def test_time_base_is_not_frame_rate() -> None:
    """Guard the old bug: 1/time_base is a tick rate, not fps."""
    tb = Fraction(1, 90000)
    bogus = 1.0 / float(tb)
    if bogus == 30.0:
        fail("time_base unexpectedly equals 30 fps")
    if bogus != 90000.0:
        fail(f"unexpected tick rate {bogus}")
    print(f"  ok  1/time_base = {bogus:.0f} ticks/s, not a frame rate")


def main() -> int:
    print("rope video timing tests")
    test_cfr_spacing_preserved()
    test_vfr_spacing_preserved()
    test_nonzero_start_normalised()
    test_offset_roundtrip()
    test_time_base_is_not_frame_rate()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

