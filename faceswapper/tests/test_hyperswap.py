"""HyperSwap-256 wiring tests (no GPU, models, torch or OpenCV needed).

HyperSwap differs from the inswapper family in three ways the pipeline has
to honour, and each one is pinned down here:

  1. its latent is the raw L2-normalized 512-d embedding, NOT the
     emap-projected one the inswapper consumes;
  2. its crop is normalized to [-1, 1] via (x/255 - 0.5) / 0.5, not 0-1;
  3. it runs natively at 256x256 in a single pass, no polyphase tiling.

The mode string that selects all three is 'hyperswap-256'
(SwapperTypeTextSel / --swapper-model).

    python -m faceswapper.tests.test_hyperswap
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from faceswapper.model_store import (
    MODEL_SPECS,
    build_download_url,
    specs_for_names,
)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str, pattern: str) -> str:
    """Return the source of the first def/class matching `pattern`."""
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.S)
    if match is None:
        fail(f"{pattern!r} not found in {relative}")
    return match.group(0)


def test_mode_is_selectable() -> None:
    from faceswapper.parameters import PARAMETER_BY_NAME

    param = PARAMETER_BY_NAME["SwapperTypeTextSel"]
    if "hyperswap-256" not in param.modes:
        fail(f"hyperswap-256 missing from modes: {param.modes}")
    if param.default != "inswapper-128":
        fail(f"default swapper changed to {param.default!r}; existing "
             "setups must keep inswapper_128")
    print(f"  ok  SwapperTypeTextSel offers {param.modes}")


def test_model_is_downloadable() -> None:
    specs = specs_for_names(["hyperswap"])
    if [s.name for s in specs] != ["hyperswap"]:
        fail(f"unexpected specs for hyperswap: {[s.name for s in specs]}")
    spec = specs[0]
    if spec.local_name != "hyperswap_1a_256.onnx":
        fail(f"unexpected local name: {spec.local_name}")
    if spec.size <= 0:
        fail(f"unexpected size: {spec.size}")
    # Upstream publishes no sha256 for models-3.3.0, only the CRC32
    # FaceFusion's own hash_helper uses — the spec must carry one of them
    # or the file can never be verified.
    if not spec.sha256 and not spec.crc32:
        fail("hyperswap spec has neither sha256 nor crc32")
    if spec.sha256 and len(spec.sha256) != 64:
        fail(f"malformed sha256: {spec.sha256!r}")
    if spec.crc32 and not re.fullmatch(r"[0-9a-f]{8}", spec.crc32):
        fail(f"malformed crc32: {spec.crc32!r}")
    url = build_download_url("huggingface", spec)
    if "hyperswap_1a_256.onnx" not in url:
        fail(f"download URL does not name the file: {url}")
    print(f"  ok  hyperswap spec verifies ({spec.local_name}, "
          f"{'crc32 ' + spec.crc32 if spec.crc32 else 'sha256'})")


def test_latent_mode_is_raw() -> None:
    """The dispatch must ask for the un-projected embedding."""
    body = _source(
        "faceswapper/VideoManager.py",
        r"def swap_core\(.*?(?=\n    def )",
    )
    if "use_hyperswap = (swapper_type == 'hyperswap-256')" not in body:
        fail("swap_core does not recognize hyperswap-256")
    if "latent_mode = 'raw' if use_hyperswap else 'emap'" not in body:
        fail("latent_mode does not switch to raw for HyperSwap")
    if "poly_pass = self._hyperswap_pass" not in body:
        fail("swap_core never dispatches to _hyperswap_pass")
    # The other swapper paths must be untouched.
    for branch in ("_native_512_pass", "_native_256_pass", "_polyphase_pass_v1"):
        if f"poly_pass = self.{branch}" not in body:
            fail(f"dispatch to {branch} was lost")
    print("  ok  hyperswap-256 dispatches to _hyperswap_pass with raw latent, "
          "other branches intact")


def test_pass_normalizes_both_ways() -> None:
    """_hyperswap_pass must present [-1, 1] to the model and hand 0-1 back,
    so the strength loop and every downstream consumer see the same
    contract the inswapper passes already provide."""
    body = _source(
        "faceswapper/VideoManager.py",
        r"def _hyperswap_pass\(.*?(?=\n    def |\n    # @profile)",
    )
    if "self.models.run_hyperswap" not in body:
        fail("_hyperswap_pass does not call run_hyperswap")
    if "(normalized - 0.5) / 0.5" not in body:
        fail("input is not normalized to [-1, 1]")
    if "mul(0.5).add(0.5)" not in body:
        fail("output is not de-normalized back to 0-1")
    # The pass is size-agnostic; swap_core is what pins it to 256.
    core = _source(
        "faceswapper/VideoManager.py",
        r"def swap_core\(.*?(?=\n    def )",
    )
    if "elif use_native_256 or use_hyperswap:" not in core:
        fail("HyperSwap is not routed to the 256 crop size")
    print("  ok  _hyperswap_pass maps 0-1 in -> [-1, 1] model -> 0-1 out at 256")


def test_normalization_roundtrips() -> None:
    """The 0-1 -> [-1,1] -> 0-1 round trip must be lossless, otherwise a
    second Strength iteration compounds the error and the paste-back
    lands on the wrong brightness."""
    forward = lambda x: (x - 0.5) / 0.5      # noqa: E731
    backward = lambda x: x * 0.5 + 0.5       # noqa: E731
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        restored = backward(forward(value))
        if abs(restored - value) > 1e-9:
            fail(f"roundtrip drift at {value}: {restored}")
    # And the untransformed path really would be wrong, so the test above
    # is not vacuous: a dark pixel normalizes negative, and scaling that
    # straight to 255 lands below 0, where the downstream clamp flattens
    # every shadow to black.
    naive = forward(0.1) * 255.0
    if 0.0 <= naive <= 255.0:
        fail(f"naive scaling unexpectedly in range: {naive}")
    print(f"  ok  [-1, 1] normalization roundtrips losslessly "
          f"(naive x255 of 0.1 gives {naive:.1f}, which would clamp to 0)")


def test_session_is_wired() -> None:
    """The new model must join every registry the other swappers use, or
    Per-Thread mode and VRAM tracking silently skip it."""
    models_src = (REPO_ROOT / "faceswapper" / "Models.py").read_text(
        encoding="utf-8")
    for registry in (
        "_VRAM_TRACKED_ATTRS",
        "_ATTR_TO_TLS_NAME",
        "_BACKEND_PREF_ATTRS",
    ):
        block = _source("faceswapper/Models.py", rf"{registry} = .*?\n\n")
        if "hyperswap" not in block:
            fail(f"hyperswap missing from Models.{registry}")
    if "self.hyperswap_model = []" not in models_src:
        fail("hyperswap_model attribute is never initialised")
    if "def run_hyperswap(" not in models_src:
        fail("Models.run_hyperswap is missing")
    if "def _create_hyperswap_session(" not in models_src:
        fail("Models._create_hyperswap_session is missing")
    if "def _get_hyperswap_session(" not in models_src:
        fail("Models._get_hyperswap_session is missing")

    body = _source(
        "faceswapper/Models.py",
        r"def run_hyperswap\(.*?(?=\n    def )",
    )
    code = re.sub(r'""".*?"""', '', body, flags=re.S)
    bound = re.findall(r"bind_(?:input|output)\(\s*name='([^']+)'", code)
    if bound != ["target", "source", "output"]:
        fail(f"unexpected io binding: {bound} (expected the exported "
             "source/target/output tensors)")
    if "(1, 3, 256, 256)" not in code or "(1, 512)" not in code:
        fail("io binding shapes do not match the exported model")

    # delete_models must release it too, or the session leaks past the run.
    delete = _source(
        "faceswapper/Models.py",
        r"def delete_models\(.*?(?=\n    def )",
    )
    if "hyperswap_model" not in delete:
        fail("delete_models does not release the hyperswap session")
    print("  ok  hyperswap session wired into registries, io binding and teardown")


def main() -> int:
    print("faceswapper HyperSwap tests")
    test_mode_is_selectable()
    test_model_is_downloadable()
    test_latent_mode_is_raw()
    test_pass_normalizes_both_ways()
    test_normalization_roundtrips()
    test_session_is_wired()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
