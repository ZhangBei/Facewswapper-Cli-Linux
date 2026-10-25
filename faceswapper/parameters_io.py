"""saved_parameters.json 的读写辅助函数。

磁盘格式是扁平的 {name: value} JSON——和最初的桌面 GUI 完全一致，所以从它
导出的文件在这里照样能加载。因此"迁移"的含义就是"校验 key 是否已知，其余
丢掉"：属于已移除的 GUI-only parameter 的 key 会被静默忽略。

对外接口：

    load(path) -> dict[str, Any]
        读 JSON，只保留 PARAMETER_BY_NAME 里有的 key，并按 param 的 kind
        把值强转成相应的 Python 类型。返回的 dict 可直接喂给 VideoManager。

    save(values, path) -> None
        写 {name: value} JSON。不在 PARAMETER_BY_NAME 里的 key 会被静默丢弃
        （防御性的——与旧版 load 的行为一致）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faceswapper.parameters import PARAMETER_BY_NAME, SelectParam


# 桌面 GUI 在 swapper 模式改名之前写下的值——那时只有裸分辨率
# （'128'/'256'/'512'），后来才改成带模型名的形式。没有这张表的话，一份
# 写着 "256" 的旧 saved_parameters.json 会被静默降级成 128 基线，而不是
# polyphase-256。
_LEGACY_SWAPPER_MODES = {
    "128": "inswapper-128",
    "256": "inswapper-256",
    "512": "inswapper-512",
}


def _coerce(name: str, value: Any) -> Any:
    """把从 JSON 读到的值强转成 param 期望的 kind。

    JSON 给我们的是 str/int/float/bool/list，而 schema 期望的是：
    - slider：float（现在的 GUI 有时存 int，别处存 float）
    - switch：bool
    - select：str（必须在 modes 里）
    - entry：str
    """
    p = PARAMETER_BY_NAME.get(name)
    if p is None:
        return value
    if p.kind == "switch":
        return bool(value)
    if p.kind == "slider":
        try:
            return float(value)
        except (TypeError, ValueError):
            return p.default
    if p.kind == "select":
        # 必须匹配某个已知 mode，否则回退到默认值。
        if isinstance(p, SelectParam) and value in p.modes:
            return value
        if name == "SwapperTypeTextSel" and value in _LEGACY_SWAPPER_MODES:
            return _LEGACY_SWAPPER_MODES[value]
        return p.default
    if p.kind == "entry":
        return "" if value is None else str(value)
    return value


def load(path: str | Path) -> dict[str, Any]:
    """读一份 saved_parameters.json，返回清理过的 {name: value} dict。

    未知 key 会被丢掉，值会做类型强转。文件不存在时返回 {}。
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for name, value in raw.items():
        if name in PARAMETER_BY_NAME:
            out[name] = _coerce(name, value)
    return out


def save(values: dict[str, Any], path: str | Path) -> None:
    """写 {name: value} JSON。防御性地丢掉未知 key。"""
    filtered = {k: v for k, v in values.items() if k in PARAMETER_BY_NAME}
    Path(path).write_text(json.dumps(filtered, indent=2), encoding="utf-8")
