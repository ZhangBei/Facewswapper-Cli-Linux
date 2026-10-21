"""Model inventory and download tests (no network required)."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from rope.errors import HeadlessError
from rope.model_store import (
    BASE_MODELS,
    MODEL_SPECS,
    build_download_url,
    download_model,
    ensure_models,
    inspect_model,
    specs_for_names,
)


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
        if len(spec.sha256) != 64:
            fail(f"invalid SHA-256 for {spec.name}")
        if spec.size <= 0:
            fail(f"invalid size for {spec.name}")
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


def main() -> int:
    print("rope model store tests")
    test_inventory_is_complete()
    test_provider_urls()
    test_missing_and_corrupt_detection()
    test_download_falls_back_between_providers()
    test_ensure_allows_offline_failure()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
