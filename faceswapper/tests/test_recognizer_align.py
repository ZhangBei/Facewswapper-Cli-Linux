"""B3/B4 wiring tests (no GPU, models, torch or OpenCV needed).

B3 adds an alternative recognizer weight (glintr100, ResNet100) selected by
RecognizerTypeTextSel / --recognizer; B4 adds a 68-point alignment path
selected by AlignTypeTextSel / --align: 2dfan4 detects 68 landmarks, a
FaceFusion-style 68->5 reduction (eye means, nose tip, mouth corners) feeds
the SAME arcface_dst template, so the downstream similarity transform is
unchanged.

Both default to the old behaviour (w600k_r50 + 5-point) and both must keep
that guarantee. The pure part of the 68->5 reduction lives in
frontal_match-free stdlib form here; the GPU-side landmark inference is
wired but not executed.

    python -m faceswapper.tests.test_recognizer_align
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from faceswapper.model_store import (
    MODEL_BY_NAME,
    build_download_url,
    specs_for_names,
    specs_for_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def _source(relative: str, pattern: str) -> str:
    """Return the source of the first def/class matching `pattern`."""
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.S)
    if match is None:
        fail(f"{pattern!r} not found in {relative}")
    return match.group(0)


def _strip_docstrings(body: str) -> str:
    return re.sub(r'""".*?"""', '', body, flags=re.S)


# FaceFusion convert_to_face_landmark_5 的索引，直接抄过来作为基准：左眼
# 36-41 均值，右眼 42-47 均值，鼻尖 30，左嘴角 48，右嘴角 54。_models.py
# 里的换算必须与之等价（均值/直取的选取逻辑），所以这里用一个手工梯子
# 重放同样的计算，绑定 68 点布局的语义位。
def _reduce_68_like_facesfusion(lm68) -> list:
    return [
        lm68[36:42].mean(axis=0),
        lm68[42:48].mean(axis=0),
        lm68[30],
        lm68[48],
        lm68[54],
    ]


def test_modes_are_selectable() -> None:
    from faceswapper.parameters import PARAMETER_BY_NAME

    rec = PARAMETER_BY_NAME["RecognizerTypeTextSel"]
    if rec.modes != ["w600k_r50", "glintr100"]:
        fail(f"RecognizerTypeTextSel modes wrong: {rec.modes}")
    if rec.default != "w600k_r50":
        fail(f"recognizer default changed: {rec.default!r}")

    align = PARAMETER_BY_NAME["AlignTypeTextSel"]
    if align.modes != ["5-point", "68-point"]:
        fail(f"AlignTypeTextSel modes wrong: {align.modes}")
    if align.default != "5-point":
        fail(f"align default changed: {align.default!r}")
    print("  ok  RecognizerTypeTextSel / AlignTypeTextSel selectable, "
          "defaults keep old behaviour")


def test_specs_verify() -> None:
    for name, local, checksum in (
        ("arcface100", "glintr100.onnx", "sha256"),
        ("landmark68", "2dfan4.onnx", "crc32"),
    ):
        spec = MODEL_BY_NAME.get(name)
        if spec is None:
            fail(f"{name} missing from MODEL_BY_NAME")
        if spec.local_name != local:
            fail(f"{name} local_name is {spec.local_name!r}, expected {local!r}")
        if not getattr(spec, checksum):
            fail(f"{name} spec carries no {checksum}")
        if getattr(spec, checksum) and checksum == "crc32":
            if not re.fullmatch(r"[0-9a-f]{8}", spec.crc32):
                fail(f"{name} crc32 malformed: {spec.crc32!r}")
        if getattr(spec, checksum) and checksum == "sha256":
            if len(spec.sha256) != 64:
                fail(f"{name} sha256 malformed: {spec.sha256!r}")
    # arcface100 has no GitHub release upstream — its remote_url must
    # route through {mirror}, and github must be rejected loudly.
    spec = MODEL_BY_NAME["arcface100"]
    hf = build_download_url("huggingface", spec)
    if "glintr100.onnx" not in hf or "{mirror}" in hf:
        fail(f"arcface100 hf url wrong: {hf}")
    try:
        build_download_url("github", spec)
        fail("arcface100 on github should raise")
    except Exception:
        pass
    gh = build_download_url("github", MODEL_BY_NAME["landmark68"])
    if "2dfan4.onnx" not in gh:
        fail(f"landmark68 github url wrong: {gh}")
    print("  ok  arcface100 (sha256, mirror-only) and landmark68 "
          "(crc32, release) specs verify")


def test_selection_pulls_optional_models() -> None:
    """Recognizer/align 参数必须决定按需下载哪个可选模型，默认值必须
    什么额外模型都不拉。"""
    base = {s.name for s in specs_for_parameters({})}
    if "arcface100" in base or "landmark68" in base:
        fail(f"defaults must not pull optional models: {sorted(base)}")

    chosen = {s.name for s in specs_for_parameters({
        "RecognizerTypeTextSel": "glintr100",
        "AlignTypeTextSel": "68-point",
    })}
    if "arcface100" not in chosen or "landmark68" not in chosen:
        fail(f"glintr100/68-point must add arcface100+landmark68, got "
             f"{sorted(chosen)}")
    print("  ok  specs_for_parameters pulls arcface100/landmark68 only "
          "when selected")


def test_models_wiring() -> None:
    """set_recognizer / set_align_type 校验回退；landmark68 session 进每
    一个注册表；recognize() 的 68 点分支换算与 FaceFusion 索引一致。"""
    models_src = (REPO_ROOT / "faceswapper" / "Models.py").read_text(
        encoding="utf-8")

    setter = _source("faceswapper/Models.py",
                     r"def set_recognizer\(.*?(?=\n    def )")
    setter = _strip_docstrings(setter)
    if "_RECOGNIZER_FILES" not in setter or "'w600k_r50'" not in setter:
        fail("set_recognizer does not validate against _RECOGNIZER_FILES")
    files = _source("faceswapper/Models.py",
                    r"_RECOGNIZER_FILES = \{.*?\}")
    if "'glintr100'" not in files:
        fail("_RECOGNIZER_FILES does not list glintr100")

    setter = _source("faceswapper/Models.py",
                     r"def set_align_type\(.*?(?=\n    def )")
    if "'68-point'" not in _strip_docstrings(setter):
        fail("set_align_type does not validate 68-point")

    for registry in ("_VRAM_TRACKED_ATTRS", "_ATTR_TO_TLS_NAME",
                     "_BACKEND_PREF_ATTRS"):
        block = _source("faceswapper/Models.py", rf"{registry} = .*?\n\n")
        if "landmark68" not in block:
            fail(f"landmark68 missing from Models.{registry}")
    for needle in (
        "self.landmark68_model = []",
        "def _create_landmark68_session(",
        "def _get_landmark68_session(",
        "def _landmark68_to_kps(",
        "def _landmark68_to_5(",  # placeholder removed below if absent
    ):
        if needle == "def _landmark68_to_5(":
            continue  # no separate helper; reduction is inline
        if needle not in models_src:
            fail(f"Models.py missing {needle!r}")
    delete = _source("faceswapper/Models.py",
                     r"def delete_models\(.*?(?=\n    def )")
    if "landmark68_model" not in delete:
        fail("delete_models does not release landmark68_model")

    recognize = _source("faceswapper/Models.py",
                        r"def recognize\(.*?(?=\n    def )")
    code = _strip_docstrings(recognize)
    if "self._align_type == '68-point'" not in code:
        fail("recognize() does not branch on 68-point")
    if "ARCFACE_DST_106" in models_src:
        fail("abandoned ARCFACE_DST_106 template still in Models.py")

    lmk = _source("faceswapper/Models.py",
                  r"def _landmark68_to_kps\(.*?(?=\n    def )")
    lmk = _strip_docstrings(lmk)
    # The 68->5 reduction must match FaceFusion's indices exactly.
    if "lm68[36:42].mean(axis=0)" not in lmk:
        fail("left eye is not the 36-41 mean")
    if "lm68[42:48].mean(axis=0)" not in lmk:
        fail("right eye is not the 42-47 mean")
    for idx in ("lm68[30]", "lm68[48]", "lm68[54]"):
        if idx not in lmk:
            fail(f"reduction missing {idx}")
    # Coordinate output contract: 'landmarks' (1,68,3) in 64-grid units.
    if "/ 64.0 * 256.0" not in lmk:
        fail("landmark coords are not rescaled from the 64 grid to 256")
    if "(1, 3, 256, 256)" not in lmk:
        fail("input binding shape is not (1, 3, 256, 256)")
    print("  ok  Models.py: recognizer/align setters validate, landmark68 "
          "session registered, 68->5 reduction matches FaceFusion indices")


def test_headless_and_cli_wiring() -> None:
    """SwapOptions 有 recognizer/align_type 字段并注入 Models；CLI 有
    --recognizer/--align 并折叠进 overrides（供按需下载用）。"""
    base = _source("faceswapper/headless.py",
                   r"class SwapOptions:.*?(?=\n@|\nclass |\ndef )")
    for field in ("recognizer:", "align_type:"):
        if field not in base:
            fail(f"SwapOptions missing {field}")
    init = _source("faceswapper/headless.py",
                   r"class _SwapperBase:.*?(?=\nclass |\ndef )")
    init = _strip_docstrings(init)
    if "set_recognizer(" not in init:
        fail("_SwapperBase.__init__ never calls models.set_recognizer")
    if "set_align_type(" not in init:
        fail("_SwapperBase.__init__ never calls models.set_align_type")

    cli_src = (REPO_ROOT / "faceswapper" / "cli.py").read_text(
        encoding="utf-8")
    for needle in (
        '"--recognizer"',
        '"--align"',
        'overrides["RecognizerTypeTextSel"] = args.recognizer',
        'overrides["AlignTypeTextSel"] = args.align',
    ):
        if needle not in cli_src:
            fail(f"cli.py missing {needle!r}")
    print("  ok  headless injects recognizer/align_type, CLI flags fold "
          "into overrides")


def test_defaults_are_inert() -> None:
    """默认 snapshot 下 headless 注入的值必须与旧行为逐字等价：w600k_r50
    走原 session 路径，5-point 不进 landmark 分支。"""
    from faceswapper.parameters import default_values

    snap = default_values()
    if snap["RecognizerTypeTextSel"] != "w600k_r50":
        fail("default recognizer must stay w600k_r50")
    if snap["AlignTypeTextSel"] != "5-point":
        fail("default align must stay 5-point")

    # _SwapperBase 注入的默认值必须与 schema 默认一致。
    init = _source("faceswapper/headless.py",
                   r"class _SwapperBase:.*?(?=\nclass |\ndef )")
    init = _strip_docstrings(init)
    if '"w600k_r50"' not in init or '"5-point"' not in init:
        fail("injection defaults drifted from the schema defaults")
    print("  ok  defaults are inert (w600k_r50 + 5-point = old behaviour)")


def main() -> int:
    print("faceswapper recognizer/align tests")
    test_modes_are_selectable()
    test_specs_verify()
    test_selection_pulls_optional_models()
    test_models_wiring()
    test_headless_and_cli_wiring()
    test_defaults_are_inert()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
