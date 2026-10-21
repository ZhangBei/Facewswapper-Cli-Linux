"""Model inventory, integrity checks, and multi-provider downloads.

This module deliberately depends only on the Python standard library so
``download-models`` works before the GPU inference stack is installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from rope.errors import HeadlessError


CACHE_FILE = ".rope-models.json"
DEFAULT_PROVIDERS = ("huggingface", "hf-mirror", "github")
PROVIDER_LABELS = {
    "github": "GitHub Release",
    "huggingface": "Hugging Face",
    "hf-mirror": "Hugging Face mirror",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    local_name: str
    remote_name: str
    release: str
    size: int
    sha256: str
    role: str
    required: bool = False


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="retinaface",
        local_name="det_10g.onnx",
        remote_name="retinaface_10g.onnx",
        release="models-3.0.0",
        size=16926877,
        sha256=(
            "808399c3d0b40ed340634ed590ae564a7caa09e5659916923a95d4470b54ff5a"
        ),
        role="RetinaFace detector",
        required=True,
    ),
    ModelSpec(
        name="arcface",
        local_name="w600k_r50.onnx",
        remote_name="arcface_w600k_r50.onnx",
        release="models-3.0.0",
        size=174388474,
        sha256=(
            "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"
        ),
        role="ArcFace recognition",
        required=True,
    ),
    ModelSpec(
        name="inswapper",
        local_name="inswapper_128.fp16.onnx",
        remote_name="inswapper_128_fp16.onnx",
        release="models-3.0.0",
        size=277680829,
        sha256=(
            "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420"
        ),
        role="Inswapper 128 face swapper",
        required=True,
    ),
    ModelSpec(
        name="gfpgan",
        local_name="GFPGANv1.4.onnx",
        remote_name="gfpgan_1.4.onnx",
        release="models-3.0.0",
        size=340299087,
        sha256=(
            "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"
        ),
        role="GFPGAN v1.4 face restorer",
    ),
    ModelSpec(
        name="scrfd",
        local_name="scrfd_2.5g_bnkps.onnx",
        remote_name="scrfd_2.5g.onnx",
        release="models-3.0.0",
        size=3295067,
        sha256=(
            "2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada"
        ),
        role="SCRFD alternative face detector",
    ),
)

MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
BASE_MODELS = tuple(spec for spec in MODEL_SPECS if spec.required)


def build_download_url(provider: str, spec: ModelSpec) -> str:
    """Build one provider URL for a model."""
    if provider == "github":
        return (
            "https://github.com/facefusion/facefusion-assets/releases/download/"
            f"{spec.release}/{spec.remote_name}"
        )
    if provider == "huggingface":
        return (
            "https://huggingface.co/facefusion/"
            f"{spec.release}/resolve/main/{spec.remote_name}"
        )
    if provider == "hf-mirror":
        return (
            "https://hf-mirror.com/facefusion/"
            f"{spec.release}/resolve/main/{spec.remote_name}"
        )
    raise HeadlessError(f"unknown model download provider: {provider}")


def normalize_providers(providers: Iterable[str] | None) -> tuple[str, ...]:
    values = tuple(providers or DEFAULT_PROVIDERS)
    if not values:
        raise HeadlessError("at least one model download provider is required")
    unknown = [value for value in values if value not in PROVIDER_LABELS]
    if unknown:
        raise HeadlessError(
            "unknown model download provider(s): " + ", ".join(unknown)
        )
    return values


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cache(models_dir: str | os.PathLike[str]) -> dict:
    cache_path = Path(models_dir) / CACHE_FILE
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(models_dir: str | os.PathLike[str], cache: dict) -> None:
    cache_path = Path(models_dir) / CACHE_FILE
    try:
        cache_path.write_text(
            json.dumps(cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # A read-only or full model volume must not block inference when
        # the model itself is valid; the cache is only an optimization.
        pass


def _cache_matches(
    cache: dict,
    spec: ModelSpec,
    path: Path,
) -> bool:
    entry = cache.get(spec.name)
    if not isinstance(entry, dict):
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        entry.get("size") == spec.size
        and entry.get("sha256") == spec.sha256
        and entry.get("mtime_ns") == stat.st_mtime_ns
    )


def _record_cache(
    models_dir: str | os.PathLike[str],
    cache: dict,
    spec: ModelSpec,
    path: Path,
) -> None:
    try:
        stat = path.stat()
    except OSError:
        return
    cache[spec.name] = {
        "file": spec.local_name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": spec.sha256,
    }
    _write_cache(models_dir, cache)


def inspect_model(
    models_dir: str | os.PathLike[str],
    spec: ModelSpec,
    *,
    verify: bool = False,
) -> str:
    """Return ``valid``, ``missing``, or ``corrupt`` for one model."""
    path = Path(models_dir) / spec.local_name
    if not path.is_file():
        return "missing"
    try:
        actual_size = path.stat().st_size
    except OSError:
        return "missing"
    if actual_size != spec.size:
        return "corrupt"

    cache = _read_cache(models_dir)
    if not verify and _cache_matches(cache, spec, path):
        return "valid"
    if sha256_file(path) != spec.sha256:
        return "corrupt"
    _record_cache(models_dir, cache, spec, path)
    return "valid"


def inspect_models(
    models_dir: str | os.PathLike[str],
    specs: Iterable[ModelSpec] = MODEL_SPECS,
    *,
    verify: bool = False,
) -> dict[str, str]:
    return {
        spec.name: inspect_model(models_dir, spec, verify=verify)
        for spec in specs
    }


def _download_with_curl(
    url: str,
    destination: Path,
    expected_size: int,
) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise HeadlessError("curl not found")

    if destination.exists() and destination.stat().st_size > expected_size:
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resume_args = ["--continue-at", "-"] if destination.exists() else []
    command = [
        curl,
        "--location",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "15",
        *resume_args,
        "--output",
        str(destination),
        url,
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip()
        raise HeadlessError(
            f"download failed ({process.returncode})"
            + (f": {detail}" if detail else "")
        )


def _download_with_urllib(
    url: str,
    destination: Path,
    expected_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        destination.unlink()
        existing_size = 0

    headers = {
        "User-Agent": "Rope-CLI-model-downloader/1.0",
    }
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HeadlessError(f"download failed: {exc}") from exc

    mode = "ab" if existing_size and response.status == 206 else "wb"
    if mode == "wb":
        existing_size = 0
    with response, open(destination, mode) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def download_model(
    models_dir: str | os.PathLike[str],
    spec: ModelSpec,
    providers: Sequence[str],
    *,
    force: bool = False,
    on_event: Callable[[str], None] | None = print,
    downloader: Callable[[str, Path, int], None] | None = None,
) -> str:
    """Download and atomically install one model, trying providers in order."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    final_path = models_path / spec.local_name
    partial_path = models_path / f"{spec.local_name}.part"

    if not force and inspect_model(models_path, spec) == "valid":
        if on_event:
            on_event(f"[model] {spec.local_name}: valid")
        return "valid"

    errors: list[str] = []
    for provider in providers:
        url = build_download_url(provider, spec)
        if on_event:
            on_event(
                f"[model] {spec.local_name}: downloading via "
                f"{PROVIDER_LABELS[provider]}"
            )
        try:
            if downloader is not None:
                downloader(url, partial_path, spec.size)
            elif shutil.which("curl") is not None:
                _download_with_curl(url, partial_path, spec.size)
            else:
                _download_with_urllib(url, partial_path, spec.size)

            if partial_path.stat().st_size != spec.size:
                raise HeadlessError(
                    f"downloaded size {partial_path.stat().st_size}, "
                    f"expected {spec.size}"
                )
            actual_sha256 = sha256_file(partial_path)
            if actual_sha256 != spec.sha256:
                raise HeadlessError(
                    f"SHA-256 mismatch: {actual_sha256}, "
                    f"expected {spec.sha256}"
                )

            os.replace(partial_path, final_path)
            cache = _read_cache(models_path)
            _record_cache(models_path, cache, spec, final_path)
            if on_event:
                on_event(f"[model] {spec.local_name}: installed and verified")
            return "downloaded"
        except (
            HeadlessError,
            OSError,
            TimeoutError,
            urllib.error.URLError,
        ) as exc:
            errors.append(f"{provider}: {exc}")
            try:
                partial_path.unlink()
            except FileNotFoundError:
                pass

    if final_path.exists() and inspect_model(models_path, spec) != "valid":
        try:
            final_path.unlink()
        except OSError:
            pass
    raise HeadlessError(
        f"could not download {spec.local_name}:\n  "
        + "\n  ".join(errors)
    )


def ensure_models(
    models_dir: str | os.PathLike[str],
    specs: Iterable[ModelSpec] = BASE_MODELS,
    *,
    providers: Iterable[str] | None = DEFAULT_PROVIDERS,
    force: bool = False,
    verify: bool = False,
    allow_download: bool = True,
    on_event: Callable[[str], None] | None = print,
    downloader: Callable[[str, Path, int], None] | None = None,
) -> dict[str, str]:
    """Ensure all requested models are valid, downloading when permitted."""
    selected = tuple(specs)
    provider_values = (
        DEFAULT_PROVIDERS if providers is None else tuple(providers)
    )
    if provider_values:
        provider_values = normalize_providers(provider_values)
    results: dict[str, str] = {}

    for spec in selected:
        status = inspect_model(models_dir, spec, verify=verify or force)
        if status == "valid" and not force:
            if on_event:
                on_event(f"[model] {spec.local_name}: valid")
            results[spec.name] = status
            continue
        if not allow_download or not provider_values:
            raise HeadlessError(
                f"model {spec.local_name} is {status}; automatic download "
                "is disabled"
            )
        results[spec.name] = download_model(
            models_dir,
            spec,
            provider_values,
            force=force,
            on_event=on_event,
            downloader=downloader,
        )
    return results


def specs_for_names(names: Iterable[str]) -> tuple[ModelSpec, ...]:
    values = tuple(names)
    if not values or "base" in values:
        return BASE_MODELS
    unknown = [value for value in values if value not in MODEL_BY_NAME]
    if unknown:
        raise HeadlessError("unknown model(s): " + ", ".join(unknown))
    selected: dict[str, ModelSpec] = {}
    for value in values:
        selected[MODEL_BY_NAME[value].name] = MODEL_BY_NAME[value]
    return tuple(selected.values())


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            break
        value /= 1024.0
    return f"{value:.1f} {unit}"
