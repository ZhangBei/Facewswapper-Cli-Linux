"""模型清单、完整性校验与多 provider 下载。

本模块刻意只依赖 Python 标准库，这样在 GPU 推理栈还没装好之前
``download-models`` 也能用。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from faceswapper.errors import HeadlessError


CACHE_FILE = ".faceswapper-models.json"
# 默认下载源是 Hugging Face 官方仓库。HF_ENDPOINT 是 huggingface_hub 自己
# 认的变量，设置后整体覆盖默认源，所以给系统配一次 mirror，本工具也一并
# 覆盖；命令行 --model-provider / --provider 的优先级最高。
ENDPOINT_ENV = "HF_ENDPOINT"
DEFAULT_PROVIDER = "huggingface"
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
    # crc32（8 位十六进制）是上游 FaceFusion 发布里给较新模型提供的唯一
    # checksum；有 sha256 的地方仍以它为准，crc32 是在 size 之外额外查的。
    crc32: str = ""
    # 上游不在 FaceFusion release 里的模型：完整的下载 URL 模板，{name}
    # 替换为 remote_name。设置后所有 provider 路径都走它（hf-mirror 域名
    # 替换 huggingface 的），而不是拼 facefusion/<release> 布局。
    remote_url: str = ""


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
    # B3（候选已落地、默认不启用）：ResNet100 ArcFace（InsightFace glintr100，
    # 512 维输出与 w600k_r50 同接口）。侧脸/难样本相似度更稳，识别推理约
    # x4。上游（FaceFusion / insightface 官方仓库）没有发布这个转出文件，
    # sha256 取自社区 ONNX 转出（HammadNaseer/arcface-glintr100-onnx）。
    ModelSpec(
        name="arcface100",
        local_name="glintr100.onnx",
        remote_name="arcface_glintr100.onnx",
        release="models-3.0.0",
        size=260665334,
        sha256=(
            "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf"
        ),
        role="ArcFace ResNet100 recognition (alternative to w600k_r50)",
        remote_url="https://{mirror}/HammadNaseer/arcface-glintr100-onnx/resolve/main/glintr100.onnx",
    ),
    # B4（候选已落地、默认不启用）：2dfan4 68 点 landmark。68 → 5 点换算
    # （两眼各取均值、鼻尖 30、两嘴角 48/54）后再过同一 5 点模板，对极端
    # 侧脸的裁剪比检测器 5 点更准。官方 FaceFusion release（crc32 校验），
    # 93.4 MB。
    ModelSpec(
        name="landmark68",
        local_name="2dfan4.onnx",
        remote_name="2dfan4.onnx",
        release="models-3.0.0",
        size=97904803,
        sha256="",
        crc32="a948738e",
        role="68-point face landmark detector (optional alignment)",
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
    ModelSpec(
        name="hyperswap",
        local_name="hyperswap_1a_256.onnx",
        remote_name="hyperswap_1a_256.onnx",
        release="models-3.3.0",
        size=402742682,
        # 上游这个 release 没有发布 sha256，只有 FaceFusion 自己的
        # hash_helper 用的那个 CRC32（zlib.crc32，8 位十六进制）。
        sha256="",
        crc32="79e50d4b",
        role="HyperSwap 256 face swapper (ResearchRAIL, non-commercial)",
    ),
    # 1b / 1c 与 1a 同架构同接口（同 shape、同 latent、同归一化），只是
    # 上游的另外两轮训练权重；CRC32 同样来自上游发布的 .hash 文件。
    ModelSpec(
        name="hyperswap1b",
        local_name="hyperswap_1b_256.onnx",
        remote_name="hyperswap_1b_256.onnx",
        release="models-3.3.0",
        size=402742682,
        sha256="",
        crc32="36312901",
        role="HyperSwap 1b 256 face swapper (ResearchRAIL, non-commercial)",
    ),
    ModelSpec(
        name="hyperswap1c",
        local_name="hyperswap_1c_256.onnx",
        remote_name="hyperswap_1c_256.onnx",
        release="models-3.3.0",
        size=402742682,
        sha256="",
        crc32="83c5ce36",
        role="HyperSwap 1c 256 face swapper (ResearchRAIL, non-commercial)",
    ),
    # ---- 可选功能模型。普通换脸一个都不需要；每个都对应一个 CLI 已经暴露
    # 出来的 parameter switch（--restorer / --occluder / --xseg / --face-parser）。
    ModelSpec(
        name="gpen256",
        local_name="GPEN-BFR-256.onnx",
        remote_name="gpen_bfr_256.onnx",
        release="models-3.0.0",
        size=75792988,
        sha256=(
            "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25"
        ),
        role="GPEN-BFR-256 face restorer (--restorer gpen256)",
    ),
    ModelSpec(
        name="gpen512",
        local_name="GPEN-BFR-512.onnx",
        remote_name="gpen_bfr_512.onnx",
        release="models-3.0.0",
        size=284340240,
        sha256=(
            "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54"
        ),
        role="GPEN-BFR-512 face restorer (--restorer gpen512)",
    ),
    ModelSpec(
        name="codeformer",
        local_name="codeformer_fp16.onnx",
        remote_name="codeformer.onnx",
        release="models-3.0.0",
        size=376951650,
        sha256=(
            "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f"
        ),
        role="CodeFormer face restorer (--restorer codeformer)",
    ),
    ModelSpec(
        name="xseg",
        local_name="occluder.onnx",
        remote_name="xseg_2.onnx",
        release="models-3.1.0",
        size=70324286,
        sha256=(
            "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042"
        ),
        role="XSeg occlusion mask (--occluder)",
    ),
    ModelSpec(
        name="faceparser",
        local_name="faceparser_resnet34.onnx",
        remote_name="bisenet_resnet_34.onnx",
        release="models-3.0.0",
        size=93632546,
        sha256=(
            "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96"
        ),
        role="BiSeNet face parser (--face-parser)",
    ),
    ModelSpec(
        name="dflxseg",
        local_name="dfl_xseg.onnx",
        remote_name="dfl_xseg.onnx",
        release="models-3.0.0",
        size=70343569,
        sha256=(
            "af105ae257170fdbc6a03460327b88d5c0b9a659aa4384fb8686ceead7294ad8"
        ),
        role="DeepFaceLab XSeg occlusion mask (--xseg)",
    ),
    ModelSpec(
        name="resnet50",
        local_name="res50.onnx",
        remote_name="res50.onnx",
        release="models-3.0.0",
        # FaceFusion 没有单独托管 res50，所以这个文件始终由用户自备：
        # 没有 checksum 意味着它只会被报告，不会被下载。
        size=0,
        sha256="",
        role="ResNet50 detector (--restorer-align reference, user-supplied)",
    ),
)

MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
BASE_MODELS = tuple(spec for spec in MODEL_SPECS if spec.required)


def build_download_url(provider: str, spec: ModelSpec) -> str:
    """为某个模型拼出一个 provider URL。

    provider 要么是已知的名字，要么是一个完整的 http(s) URL——它会被当作
    Hugging Face endpoint 的基址，所以任何 mirror（包括 $HF_ENDPOINT 指向的
    那个）不用在这里登记也能用。
    """
    # 上游不在 FaceFusion release 里的模型：专用 URL 模板，{mirror} 在
    # 镜像下载时替换为 hf-mirror.com；这类模型没有 GitHub release。
    if spec.remote_url:
        if provider == "github":
            raise HeadlessError(
                f"model {spec.name!r} is not published on the GitHub release; "
                "use huggingface / hf-mirror / a URL provider"
            )
        mirror = "hf-mirror.com" if provider == "hf-mirror" else "huggingface.co"
        return spec.remote_url.format(name=spec.remote_name, mirror=mirror)
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
    if provider.startswith(("http://", "https://")):
        return f"{provider}/facefusion/{spec.release}/resolve/main/{spec.remote_name}"
    raise HeadlessError(f"unknown model download provider: {provider}")


def default_providers() -> tuple[str, ...]:
    """调用方没指定来源时：$HF_ENDPOINT 优先，否则用 Hugging Face 官方源。"""
    endpoint = (os.environ.get(ENDPOINT_ENV) or "").strip().rstrip("/")
    return (endpoint,) if endpoint else (DEFAULT_PROVIDER,)


def no_source_message() -> str:
    """调用方显式清空来源时的提示（CLI 路径下有默认源，一般不会触发）。

    自动下载走全局 `--model-provider`，`download-models` 走子命令的
    `--provider`，两个名字都要给，否则用户会按提示去写一个当前命令不存在
    的参数。
    """
    return (
        "no model download source given: pass --model-provider (global) or "
        "--provider (download-models) with a URL or a known source "
        f"({', '.join(sorted(PROVIDER_LABELS))}), or set ${ENDPOINT_ENV}; "
        f"the default is {DEFAULT_PROVIDER}"
    )


def normalize_providers(providers: Iterable[str]) -> tuple[str, ...]:
    values = tuple(providers)
    if not values:
        raise HeadlessError(no_source_message())
    unknown = [
        value for value in values
        if value not in PROVIDER_LABELS
        and not value.startswith(("http://", "https://"))
    ]
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


def crc32_file(path: str | os.PathLike[str]) -> str:
    """CRC32，8 位小写十六进制——上游拿不出 sha256 的那些模型，FaceFusion
    发布的就是这个 checksum。"""
    checksum = 0
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return format(checksum & 0xFFFFFFFF, "08x")


def _checksum_mismatch(path: str | os.PathLike[str], spec: ModelSpec) -> str | None:
    """文件的 checksum 与 spec 相符时返回 None，否则返回一段描述。

    上游发布了 sha256 的地方以它为准；只带 CRC32 的模型退回到 CRC32。
    两者都没有的 spec（用户自备）永远不会判定为不匹配。
    """
    if spec.sha256:
        actual = sha256_file(path)
        if actual != spec.sha256:
            return f"SHA-256 mismatch: {actual}, expected {spec.sha256}"
        return None
    if spec.crc32:
        actual = crc32_file(path)
        if actual != spec.crc32:
            return f"CRC32 mismatch: {actual}, expected {spec.crc32}"
    return None


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
        # 模型卷只读或写满时，不该在模型本身有效的情况下挡住推理；cache 只是
        # 个优化手段。
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
        and entry.get("crc32") == spec.crc32
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
        "crc32": spec.crc32,
    }
    _write_cache(models_dir, cache)


def inspect_model(
    models_dir: str | os.PathLike[str],
    spec: ModelSpec,
    *,
    verify: bool = False,
) -> str:
    """对单个模型返回 ``valid``、``missing`` 或 ``corrupt``。"""
    # 既没有上游 checksum 也没有已知 size 的 spec，按定义就是用户自备的
    # （FaceFusion 的 asset 仓库里没有对应文件）。它只被如实报告、从不下载，
    # 所以 `res50.onnx` 显示为 missing，而不会让整个清单检查失败。
    if not spec.size:
        path = Path(models_dir) / spec.local_name
        return "present" if path.is_file() else "missing"
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
    if _checksum_mismatch(path, spec) is not None:
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
        "User-Agent": "Faceswapper-CLI-model-downloader/1.0",
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
    """按顺序尝试各个 provider，下载并原子地安装一个模型。"""
    if not spec.size:
        # 没什么可下的：没有任何上游 release 发布这个文件，所以既拼不出 URL，
        # 也没有 checksum 可校验。
        raise HeadlessError(
            f"{spec.local_name} is user-supplied; no upstream release "
            "publishes it, so it cannot be downloaded"
        )
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
            label = PROVIDER_LABELS.get(provider, provider)
            on_event(
                f"[model] {spec.local_name}: downloading via {label}"
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
            mismatch = _checksum_mismatch(partial_path, spec)
            if mismatch is not None:
                raise HeadlessError(mismatch)

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
    providers: Iterable[str] | None = None,
    force: bool = False,
    verify: bool = False,
    allow_download: bool = True,
    on_event: Callable[[str], None] | None = print,
    downloader: Callable[[str, Path, int], None] | None = None,
) -> dict[str, str]:
    """确保所有请求的模型都有效，允许时就地下载。"""
    selected = tuple(specs)
    # providers=None 表示"没有指定来源"：用默认源（huggingface 或
    # $HF_ENDPOINT）；显式传空列表则是调用方刻意禁用下载源。
    provider_values = (
        tuple(providers) if providers is not None else default_providers()
    )
    # 模型齐全时整个运行不联网，也不会碰到下载源。非空来源在这里校验
    # 合法性；真的要下载却没有来源时，再到下面的分支报错。
    if provider_values:
        provider_values = normalize_providers(provider_values)
    results: dict[str, str] = {}

    for spec in selected:
        status = inspect_model(models_dir, spec, verify=verify or force)
        # 用户自备的文件（上游没有 release）只报告、不下载。缺一个只是警告，
        # 不是致命错误，因为只有对应的 parameter switch 打开时它才要紧。
        if not spec.size:
            results[spec.name] = status
            continue
        if status == "valid" and not force:
            if on_event:
                on_event(f"[model] {spec.local_name}: valid")
            results[spec.name] = status
            continue
        if not allow_download:
            raise HeadlessError(
                f"model {spec.local_name} is {status}; automatic download "
                "is disabled"
            )
        if not provider_values:
            raise HeadlessError(no_source_message())
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
    if "all" in values:
        return MODEL_SPECS
    if not values or "base" in values:
        return BASE_MODELS
    unknown = [value for value in values if value not in MODEL_BY_NAME]
    if unknown:
        raise HeadlessError("unknown model(s): " + ", ".join(unknown))
    selected: dict[str, ModelSpec] = {}
    for value in values:
        selected[MODEL_BY_NAME[value].name] = MODEL_BY_NAME[value]
    return tuple(selected.values())


# parameter switch -> 它会拉进来的模型文件。只有对应功能真的启用时 CLI 才
# 需要某个模型，所以推理类命令是从 parameter snapshot 推导这个集合的，而不是
# 一律要求所有可选文件都在。
FEATURE_MODELS: dict[str, tuple[str, ...]] = {
    "OccluderSwitch": ("xseg",),
    "DFLXSegSwitch": ("dflxseg",),
    "FaceParserSwitch": ("faceparser",),
}

# RestorerTypeTextSel -> 它选中的模型文件。四个 restorer 共用同一个 switch，
# 只需要被选中那一个。
RESTORER_MODELS: dict[str, str] = {
    "GFPGAN": "gfpgan",
    "CF": "codeformer",
    "GPEN256": "gpen256",
    "GPEN512": "gpen512",
}


def specs_for_parameters(
    parameters: dict[str, object],
) -> tuple[ModelSpec, ...]:
    """从 parameter snapshot 推出一次运行需要的模型集合。

    始终包含 base 模型，再加上被启用的功能所选中的那些。
    """
    selected: dict[str, ModelSpec] = {
        spec.name: spec for spec in BASE_MODELS
    }

    def add(name: str) -> None:
        spec = MODEL_BY_NAME.get(name)
        if spec is not None and spec.name not in selected:
            selected[spec.name] = spec

    if str(parameters.get("DetectTypeTextSel") or "") == "SCRDF":
        add("scrfd")

    swapper = str(parameters.get("SwapperTypeTextSel") or "")
    if swapper == "hyperswap-256":
        add("hyperswap")
    elif swapper == "hyperswap-1b-256":
        add("hyperswap1b")
    elif swapper == "hyperswap-1c-256":
        add("hyperswap1c")

    # B3/B4：识别器权重与 68 点对齐模型（均默认关闭——选中才拉取）。
    if str(parameters.get("RecognizerTypeTextSel") or "") == "glintr100":
        add("arcface100")
    if str(parameters.get("AlignTypeTextSel") or "") == "68-point":
        add("landmark68")

    # RestorerSwitch 决定用哪一族，RestorerTypeTextSel 决定具体哪个文件。
    if parameters.get("RestorerSwitch"):
        add(RESTORER_MODELS.get(
            str(parameters.get("RestorerTypeTextSel") or ""), "gfpgan"
        ))
    for switch, names in FEATURE_MODELS.items():
        if parameters.get(switch):
            for name in names:
                add(name)
    return tuple(selected.values())


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            break
        value /= 1024.0
    return f"{value:.1f} {unit}"
