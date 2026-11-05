"""Model inventory and download tests (no network required)."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import zlib
from dataclasses import replace
from pathlib import Path

from faceswapper.errors import HeadlessError
from faceswapper.model_store import (
    BASE_MODELS,
    DEFAULT_PROVIDER,
    MODEL_SPECS,
    build_download_url,
    default_providers,
    download_model,
    ensure_models,
    inspect_model,
    normalize_providers,
    specs_for_names,
    specs_for_parameters,
)


def crc32_bytes(content: bytes) -> str:
    return format(zlib.crc32(content) & 0xFFFFFFFF, "08x")


def fail(message: str) -> None:
    print(f"  FAIL: {message}")
    raise SystemExit(1)


def test_inventory_is_complete() -> None:
    names = [spec.name for spec in MODEL_SPECS]
    if len(names) != len(set(names)):
        fail("duplicate model names in inventory")
    if [spec.name for spec in BASE_MODELS] != [
        "retinaface", "arcface", "inswapper"
    ]:
        fail(f"unexpected base model set: {[s.name for s in BASE_MODELS]}")
    for spec in MODEL_SPECS:
        # A spec with no size is user-supplied (no upstream release
        # publishes it), so it is exempt from the checksum rules.
        if not spec.size:
            continue
        if spec.sha256:
            if len(spec.sha256) != 64:
                fail(f"invalid SHA-256 for {spec.name}")
        elif not spec.crc32:
            fail(f"{spec.name} has neither sha256 nor crc32")
    print("  ok  model inventory has unique base models and SHA-256 values")


def test_provider_urls() -> None:
    spec = MODEL_SPECS[0]
    urls = {
        provider: build_download_url(provider, spec)
        for provider in ("github", "huggingface", "hf-mirror")
    }
    if "retinaface_10g.onnx" not in urls["github"]:
        fail("GitHub URL does not contain remote file name")
    if "huggingface.co" not in urls["huggingface"]:
        fail("Hugging Face URL host is wrong")
    if "hf-mirror.com" not in urls["hf-mirror"]:
        fail("mirror URL host is wrong")
    print("  ok  GitHub, Hugging Face, and mirror URLs are constructed")


def test_default_source_is_huggingface_with_env_override() -> None:
    """默认 huggingface；$HF_ENDPOINT 设置后整体覆盖，命令行参数优先级更高。"""
    import os

    saved = os.environ.pop("HF_ENDPOINT", None)
    try:
        if default_providers() != (DEFAULT_PROVIDER,):
            fail(f"expected huggingface as default source: {default_providers()}")
        try:
            normalize_providers(())
        except HeadlessError as exc:
            if "--provider" not in str(exc) or "HF_ENDPOINT" not in str(exc):
                fail(f"unexpected no-source error: {exc}")
        else:
            fail("empty provider list should be rejected")

        os.environ["HF_ENDPOINT"] = "https://mirror.example.com/"
        providers = default_providers()
        if providers != ("https://mirror.example.com",):
            fail(f"HF_ENDPOINT was not picked up: {providers}")
        url = build_download_url(providers[0], MODEL_SPECS[0])
        if "mirror.example.com" not in url or not url.endswith(".onnx"):
            fail(f"URL was not built from HF_ENDPOINT: {url}")
    finally:
        os.environ.pop("HF_ENDPOINT", None)
        if saved is not None:
            os.environ["HF_ENDPOINT"] = saved
    print("  ok  default source is huggingface, overridable by $HF_ENDPOINT")


def test_missing_and_corrupt_detection() -> None:
    spec = replace(
        MODEL_SPECS[0],
        size=4,
        sha256=hashlib.sha256(b"good").hexdigest(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        if inspect_model(models_dir, spec) != "missing":
            fail("empty directory should report a missing model")

        path = models_dir / spec.local_name
        path.write_bytes(b"bad!")
        if inspect_model(models_dir, spec, verify=True) != "corrupt":
            fail("wrong content should be reported corrupt")
        path.write_bytes(b"good")
        if inspect_model(models_dir, spec, verify=True) != "valid":
            fail("correct content should be reported valid")
    print("  ok  missing, corrupt, and valid model states are detected")


def test_download_falls_back_between_providers() -> None:
    content = b"good"
    spec = replace(
        MODEL_SPECS[0],
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    attempted: list[str] = []

    def downloader(url: str, destination: Path, expected_size: int) -> None:
        attempted.append(url)
        if url.startswith("https://huggingface.co/"):
            raise HeadlessError("simulated first-provider failure")
        destination.write_bytes(content[:expected_size])

    with tempfile.TemporaryDirectory() as tmp:
        status = download_model(
            tmp,
            spec,
            ("huggingface", "hf-mirror"),
            on_event=None,
            downloader=downloader,
        )
        if status != "downloaded":
            fail(f"unexpected download status: {status}")
        if len(attempted) != 2:
            fail(f"expected two provider attempts, got {attempted}")
        if inspect_model(tmp, spec, verify=True) != "valid":
            fail("downloaded model did not pass verification")
    print("  ok  provider fallback and post-download verification work")


def test_ensure_allows_offline_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ensure_models(
                tmp,
                BASE_MODELS,
                providers=(),
                allow_download=False,
                on_event=None,
            )
        except HeadlessError as exc:
            if "automatic download is disabled" not in str(exc):
                fail(f"unexpected offline error: {exc}")
        else:
            fail("offline mode should fail when models are missing")
    print("  ok  --no-auto-download fails clearly before inference")


def test_ensure_skips_download_when_models_are_valid() -> None:
    """模型齐全时不触发下载，也不联网，即使默认源是 huggingface。"""
    import os

    content = b"good"
    spec = replace(
        MODEL_SPECS[0],
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )

    def must_not_download(url: str, destination: Path, expected_size: int) -> None:
        fail(f"valid models must not trigger a download: {url}")

    saved = os.environ.pop("HF_ENDPOINT", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / spec.local_name).write_bytes(content)
            statuses = ensure_models(
                tmp, (spec,), providers=None,
                allow_download=True, on_event=None,
                downloader=must_not_download,
            )
            if statuses != {spec.name: "valid"}:
                fail(f"unexpected statuses: {statuses}")
    finally:
        os.environ.pop("HF_ENDPOINT", None)
        if saved is not None:
            os.environ["HF_ENDPOINT"] = saved
    print("  ok  valid local models run without any download")


def test_missing_model_uses_default_huggingface_source() -> None:
    """不指定来源时，缺模型自动走默认的 Hugging Face 官方源。"""
    import os

    content = b"good"
    spec = replace(
        MODEL_SPECS[0],
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    attempted: list[str] = []

    def downloader(url: str, destination: Path, expected_size: int) -> None:
        attempted.append(url)
        destination.write_bytes(content[:expected_size])

    saved = os.environ.pop("HF_ENDPOINT", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            statuses = ensure_models(
                tmp, (spec,), providers=None,
                allow_download=True, on_event=None,
                downloader=downloader,
            )
            if statuses != {spec.name: "downloaded"}:
                fail(f"unexpected statuses: {statuses}")
            if len(attempted) != 1 or "huggingface.co" not in attempted[0]:
                fail(f"default source was not huggingface: {attempted}")
    finally:
        os.environ.pop("HF_ENDPOINT", None)
        if saved is not None:
            os.environ["HF_ENDPOINT"] = saved
    print("  ok  missing models download from the default huggingface source")


def test_ensure_rejects_explicitly_empty_provider_list() -> None:
    """调用方显式清空来源时仍给出可操作的报错，不会静默联网。"""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ensure_models(
                tmp, BASE_MODELS[:1], providers=(),
                allow_download=True, on_event=None,
            )
        except HeadlessError as exc:
            message = str(exc)
            if "no model download source given" not in message:
                fail(f"unexpected no-source error: {message}")
            if "--model-provider" not in message or "$HF_ENDPOINT" not in message:
                fail(f"no-source error lacks actionable flags: {message}")
        else:
            fail("an explicitly empty provider list should be rejected")
    print("  ok  explicitly empty provider list is rejected")


def test_specs_for_parameters_tracks_enabled_features() -> None:
    """A run downloads the base set plus only the features it enabled."""
    base = {spec.name for spec in BASE_MODELS}

    cases = [
        ({}, base),
        ({"RestorerSwitch": True, "RestorerTypeTextSel": "CF"},
         base | {"codeformer"}),
        ({"RestorerSwitch": True, "RestorerTypeTextSel": "GPEN256"},
         base | {"gpen256"}),
        ({"RestorerSwitch": True, "RestorerTypeTextSel": "GPEN512"},
         base | {"gpen512"}),
        ({"RestorerSwitch": True}, base | {"gfpgan"}),
        ({"OccluderSwitch": True}, base | {"xseg"}),
        ({"DFLXSegSwitch": True}, base | {"dflxseg"}),
        ({"FaceParserSwitch": True}, base | {"faceparser"}),
    ]
    for parameters, expected in cases:
        got = {spec.name for spec in specs_for_parameters(parameters)}
        if got != expected:
            fail(f"unexpected model set for {parameters}: {got}")

    scrfd = {
        spec.name for spec in
        specs_for_parameters({"DetectTypeTextSel": "SCRDF"})
    }
    if scrfd != base | {"scrfd"}:
        fail(f"SCRDF detection should pull scrfd: {scrfd}")

    hyper = {
        spec.name for spec in
        specs_for_parameters({"SwapperTypeTextSel": "hyperswap-256"})
    }
    if hyper != base | {"hyperswap"}:
        fail(f"hyperswap-256 should pull hyperswap: {hyper}")

    from_snapshot = {
        spec.name for spec in
        specs_for_parameters({"DetectTypeTextSel": "SCRDF"})
    }
    if "scrfd" not in from_snapshot:
        fail("DetectTypeTextSel from the snapshot was ignored")
    print("  ok  parameter switches select exactly the models they need")


def test_user_supplied_models_are_reported_not_downloaded() -> None:
    """res50.onnx has no upstream release, so it is never fetched."""
    spec = next(s for s in MODEL_SPECS if s.name == "resnet50")
    if spec.size or spec.sha256 or spec.crc32:
        fail("resnet50 should carry no size and no checksum")

    with tempfile.TemporaryDirectory() as tmp:
        if inspect_model(tmp, spec) != "missing":
            fail("absent user-supplied model should report missing")

        path = Path(tmp) / spec.local_name
        path.write_bytes(b"anything")
        if inspect_model(tmp, spec) != "present":
            fail("present user-supplied model should report present")

        # verify=True must not hash a file we have no reference for.
        if inspect_model(tmp, spec, verify=True) != "present":
            fail("verify must not reject a user-supplied model")

        try:
            download_model(tmp, spec, ("huggingface",), on_event=None)
        except HeadlessError as exc:
            if "user-supplied" not in str(exc):
                fail(f"unexpected error: {exc}")
        else:
            fail("download_model should refuse a user-supplied model")

        # ensure_models reports it without trying to download it, even
        # when it is the only requested spec.
        statuses = ensure_models(
            tmp, (spec,), providers=("huggingface",),
            allow_download=True, on_event=None,
        )
        if statuses.get("resnet50") != "present":
            fail(f"unexpected status: {statuses}")
    print("  ok  user-supplied models are reported, never downloaded")


def test_all_group_covers_every_spec() -> None:
    selected = specs_for_names(["all"])
    if {spec.name for spec in selected} != {
        spec.name for spec in MODEL_SPECS
    }:
        fail("the 'all' group should cover every model in the inventory")
    print("  ok  'all' group covers the whole inventory")


def test_crc32_only_model_verifies() -> None:
    """A spec with no sha256 falls back to its CRC32 (the HyperSwap case:
    upstream publishes no sha256 for models-3.3.0)."""
    spec = next(s for s in MODEL_SPECS if s.name == "hyperswap")
    if spec.sha256:
        fail("hyperswap spec should carry no sha256")
    if len(spec.crc32) != 8:
        fail(f"hyperswap crc32 should be 8 hex chars: {spec.crc32!r}")

    content = b"fake onnx payload"
    good = replace(spec, size=len(content), crc32=crc32_bytes(content))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / spec.local_name
        path.write_bytes(content)
        if inspect_model(tmp, good, verify=True) != "valid":
            fail("crc32 match should read as valid")
        bad = replace(good, crc32="deadbeef")
        if inspect_model(tmp, bad, verify=True) != "corrupt":
            fail("crc32 mismatch should read as corrupt")
    print("  ok  crc32-only specs verify without a sha256")


def main() -> int:
    print("faceswapper model store tests")
    test_inventory_is_complete()
    test_provider_urls()
    test_default_source_is_huggingface_with_env_override()
    test_missing_and_corrupt_detection()
    test_download_falls_back_between_providers()
    test_ensure_allows_offline_failure()
    test_ensure_skips_download_when_models_are_valid()
    test_missing_model_uses_default_huggingface_source()
    test_ensure_rejects_explicitly_empty_provider_list()
    test_crc32_only_model_verifies()
    test_specs_for_parameters_tracks_enabled_features()
    test_user_supplied_models_are_reported_not_downloaded()
    test_all_group_covers_every_spec()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
