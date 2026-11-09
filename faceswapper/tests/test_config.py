"""config.SwapConfig 的离线等价性测试（无 GPU / numpy / ORT）。

风格化重构的守门测试：只证明「形状变了、值没变」，不证明任何 GPU 行为。
逐 key 往返是重点——映射表里写错一个目标字段，全覆盖测试会立刻抓到。

    python -m faceswapper.tests.test_config
"""

from __future__ import annotations

import re
import sys
from dataclasses import FrozenInstanceError, fields
from typing import Any

from faceswapper.config import FIELD_MAP, SwapConfig
from faceswapper.parameters import PARAMETER_BY_NAME, default_values


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _probe_value(name: str) -> Any:
    """造一个与默认值不同的值，用来验证这条映射真的被走到。"""
    p = PARAMETER_BY_NAME[name]
    default = p.default
    if p.kind == "switch":
        return not default
    if p.kind == "slider":
        return default + 1
    if p.kind == "select":
        try:
            modes = [m for m in p.modes if m != default]
        except (KeyError, AttributeError):
            modes = []
        # 这里测的是映射而非 coerce；类型/取值强转是 parameters_io.load 的职责，
        # 所以用哨兵串也没问题。
        return modes[0] if modes else "__probe_select__"
    return "__probe_entry__"


def test_map_covers_schema() -> None:
    schema_keys = set(PARAMETER_BY_NAME)
    if len(schema_keys) != 54:
        fail(f"schema changed: expected 54 keys, got {len(schema_keys)}")
    if set(FIELD_MAP) != schema_keys:
        missing = sorted(schema_keys - set(FIELD_MAP))
        extra = sorted(set(FIELD_MAP) - schema_keys)
        fail(f"map drift: missing={missing} extra={extra}")
    targets = list(FIELD_MAP.values())
    if len(set(targets)) != len(targets):
        fail("two old keys map to the same field")
    if set(targets) != {f.name for f in fields(SwapConfig)}:
        fail("FIELD_MAP targets do not match SwapConfig fields")
    print("  ok  FIELD_MAP covers all 54 keys, bijective, matches dataclass")


def test_default_roundtrip() -> None:
    got = SwapConfig.default().to_parameters()
    expected = default_values()
    if got != expected:
        fail(f"default round-trip changed values: {got!r}")
    if list(got) != list(expected):
        fail("parameter order changed")
    print("  ok  default() -> to_parameters() equals default_values()")


def test_every_key_roundtrips() -> None:
    baseline = default_values()
    for old, field in FIELD_MAP.items():
        value = _probe_value(old)
        got = SwapConfig.from_parameters({old: value}).to_parameters()
        if got[old] != value:
            fail(f"{old} -> {field}: expected {value!r}, got {got[old]!r}")
        for other, expected in baseline.items():
            if other != old and got[other] != expected:
                fail(f"setting {old} perturbed {other}")
    print("  ok  all 54 keys round-trip individually without cross-talk")


def test_unknown_keys_are_ignored() -> None:
    got = SwapConfig.from_parameters({
        "SwapFacesButton": 1,
        "GhostKey": "x",
    }).to_parameters()
    if got != default_values():
        fail("unknown keys leaked into the projected dict")
    print("  ok  unknown / removed GUI keys stay ignored")


def test_field_names_follow_cli_style() -> None:
    bad = [f for f in FIELD_MAP.values() if not re.fullmatch(r"[a-z][a-z0-9_]*", f)]
    if bad:
        fail(f"non snake_case field names: {bad}")
    print("  ok  all field names are lowercase snake_case")


def test_config_is_immutable() -> None:
    cfg = SwapConfig.default()
    try:
        cfg.threshold = 99.0  # type: ignore[misc]
    except FrozenInstanceError:
        print("  ok  SwapConfig is frozen")
        return
    fail("SwapConfig must be frozen")


def main() -> int:
    print("faceswapper config tests")
    test_map_covers_schema()
    test_default_roundtrip()
    test_every_key_roundtrips()
    test_unknown_keys_are_ignored()
    test_field_names_follow_cli_style()
    test_config_is_immutable()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
