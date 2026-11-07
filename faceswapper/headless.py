"""Headless（无界面）换脸管线。

桌面 GUI 是通过一个交互循环驱动 VideoManager 的: 拖到某一帧、"Find
Faces"、点一张源脸把它绑到某个 slot，然后播放/录制。这些在命令行上
都无法表达，所以本模块提供等效的批量路径:

    image:  检测人脸 -> 分配源脸 -> swap_video -> 写出 PNG
    video:  解码帧 -> 换掉每一张脸 -> 编码 -> 混入音频

重活是共享的: 检测、识别和真正的换脸都走 faceswapper.Models 与
VideoManager，所以相同参数下 CLI 产出与桌面应用一致的像素。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import cv2
import numpy as np
import torch

from faceswapper.EmbeddingMerge import combine
from faceswapper.decoder import DecoderPlan, frame_fields, resolve_decoder
from faceswapper.encoder_probe import EncoderPlan, resolve_encoder
from faceswapper.errors import HeadlessError
from faceswapper.frame_order import ordered_swap_frames
from faceswapper.frame_upload import normalize_plan
from faceswapper.Models import Models
from faceswapper.VideoManager import VideoManager


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class SwapOptions:
    """图片换脸所需的一切，从 CLI 参数解析而来。"""

    models_folder: str = "./models"
    # 哪些目标脸被换。"all" 换掉每一张检测到的脸，"largest" 只换最大的
    # 那一张。
    face_policy: str = "all"
    detect_type: str = "Retinaface"
    detect_input_size: int = 640
    detect_score: float = 0.5
    # "Mean" | "Median" | "Sph" | "Geo" | "Qual"——多张源脸如何合并成
    # 一个身份向量。
    merge_mode: str = "Mean"
    # 叠加在默认值之上的额外参数（若同时提供了 saved_parameters.json，
    # 则叠加在它之上）。
    overrides: dict[str, Any] = field(default_factory=dict)
    # 强制走 ONNX/CUDA 后端而不是 TensorRT。GUI 里 TRT 是默认，但它的
    # Linux engine 缓存无法从 Windows 安装直接移植过来，所以除非明确要求，
    # CLI 保持用 CUDA EP。
    backend: str = "cuda"
    # CUDA EP 的 VRAM arena 扩展策略：same（默认，kSameAsRequested）或 pow2
    # （kNextPowerOfTwo）。后者只减少重复 cudaMalloc，不是提速手段。
    arena_strategy: str = "same"
    # 阶段 6：复用 retinaface 的 9 个 io_binding 输出缓冲，省掉每帧 9 次
    # torch.empty。默认关闭；TRT-EP 后端下会被判定逻辑自动关掉。
    reuse_detect_buffers: bool = False
    # 识别精度（RecognizeMirrorSwitch）：识别时把裁剪图水平镜像再各跑一次
    # ArcFace、取平均，压低侧脸 embedding 的方差。识别 ×2，默认关闭。
    recognize_mirror: bool = False
    # 识别精度（RecognizeDimSwitch）：识别用 dim=2 的网格平均（4 个裁剪
    # 各跑一次 ArcFace 取平均）。识别 ×4，默认关闭。
    recognize_dim2: bool = False
    # B3（RecognizerTypeTextSel）：识别权重。'w600k_r50'（默认）或
    # 'glintr100'（ResNet100，侧脸/难样本相似度更稳，推理约 ×4）。必须在
    # 任何识别调用前设置。
    recognizer: str = "w600k_r50"
    # B4（AlignTypeTextSel）：识别对齐。'5-point'（默认）或 '68-point'
    # （2dfan4 landmark 对齐，每次识别前多一次 landmark 推理）。
    align_type: str = "5-point"
    # 阶段 7：帧已是 uint8 C contiguous 时跳过 astype 的整帧复制。默认关闭，
    # 关掉时所有帧路径逐字保留原来的 astype 表达式。
    zero_copy_frames: bool = False


def _read_image(path: str) -> np.ndarray:
    """把图片读成 RGB uint8。失败时报错。"""
    if not os.path.isfile(path):
        raise HeadlessError(f"image not found: {path}")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HeadlessError(f"could not decode image (unsupported format?): {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _frame_to_uint8(rgb: np.ndarray, zero_copy: bool) -> np.ndarray:
    """阶段 7：按 plan 把一帧归一化成 uint8，能不复制就不复制。

    闸门关闭时逐字返回 ``rgb.astype("uint8")``——这是所有调用点改造前的
    表达式。开启后只有「已是 uint8 且 C contiguous」的帧直接复用原数组；
    dtype 或布局不满足时仍要转换，用 ``ascontiguousarray`` 一次完成，数值
    与 astype 等价。

    跳过复制会返回与输入共享内存的视图，调用方只把它交给同步的
    ``torch.from_numpy(...).to("cuda")`` 或 ``VideoFrame.from_ndarray``，
    不会原地写；PyAV 每帧 ``to_ndarray`` 都产出新数组，没有跨帧复用。
    """
    if not zero_copy:
        return rgb.astype("uint8")
    plan = normalize_plan(
        rgb.dtype.name,
        bool(rgb.flags.c_contiguous),
        rgb.shape[0],
        rgb.shape[1],
        zero_copy=True,
    )
    if not plan.needs_copy:
        return rgb
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _cosine_similarity_pct(a: np.ndarray, b: np.ndarray) -> float:
    """0..100 的相似度，100 = 完全相同。与 GUI 里的 helper 一致。"""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cos_dist = 1.0 - float(np.dot(a, b)) / denom
    return 100.0 - cos_dist * 50.0


def _kps_scale(kps: np.ndarray) -> float:
    """估算一张脸在原帧里有多大。

    检测器返回 5 个像素坐标的 landmark。双眼间距随人脸大小增长，所以它
    是给脸按屏幕上大小排序的一个好代理。
    """
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(kps[0] - kps[1]))


class _SwapperBase:
    """图片与视频路径共享的模型/VM 配置与源脸处理。"""

    def __init__(self, options: SwapOptions) -> None:
        self.options = options
        self.models = Models()
        self.models.set_models_folder(options.models_folder)
        # 阶段 3 的 6.2：必须在任何 session 懒加载之前设置，否则已构建的
        # session 仍用旧策略。默认 same，只有显式 pow2 才切换。
        self.models.set_arena_strategy(
            "kNextPowerOfTwo"
            if getattr(options, "arena_strategy", "same") == "pow2"
            else "kSameAsRequested"
        )
        # 阶段 6：默认关闭；resolve_reuse_buffers 会在 trt / 未知 backend 下
        # 再关一次，不猜 TRT-EP 的 io_binding 输出行为。
        self.models.set_reuse_detect_buffers(
            bool(getattr(options, "reuse_detect_buffers", False)),
            backend=options.backend,
        )
        if options.backend == "cuda":
            # 这里的 "onnx" 指的是 faceswapper.Models 里那条纯 CUDA
            # ExecutionProvider 路径，不是 CPU provider。这个命名继承自
            # GUI 的 per-model 后端开关。必须覆盖全部可切换模型：漏掉
            # 任何一个（例如 hyperswap）都会让它仍去初始化 TensorRT EP，
            # 在没有 libnvinfer.so.10 的机器上打印 EP Error 再回退。
            self.models.set_backend_preference_for_all("onnx", unload=False)
        if getattr(options, "workers", 1) > 1:
            # Shared 会把多个 worker 串行化在同一份 session 上；Per-Thread
            # 才能让它们各自占用 GPU（代价是每 worker 多一份显存）。
            self.models.set_model_session_mode("Per-Thread")
        self.vm = VideoManager(self.models)
        # 识别精度（RecognizeMirrorSwitch）：镜像平均，压低侧脸 embedding
        # 的方差。默认关闭。必须在任何识别调用前设置——即下一次
        # _apply_parameters() 读到参数之前。
        self.models.set_recognize_mirror(bool(
            getattr(options, "recognize_mirror", False)
        ))
        # 识别精度（RecognizeDimSwitch）：dim=2 网格平均。默认关闭。
        self.vm.recognize_dim2 = bool(
            getattr(options, "recognize_dim2", False)
        )
        # B3/B4：识别权重与对齐方式。默认值 = 旧行为，必须在任何识别调用
        # 前设置。set_recognizer/set_align_type 对未知值回退默认。
        self.models.set_recognizer(
            str(getattr(options, "recognizer", "w600k_r50"))
        )
        self.models.set_align_type(
            str(getattr(options, "align_type", "5-point"))
        )
        self.vm.force_recognize = bool(
            getattr(options, "force_recognize", False)
        )
        # 阶段 7：闸门默认关闭；开启后才在 VM 的 H2D 分支跳过整帧复制。
        self.vm.zero_copy_frames = bool(
            getattr(options, "zero_copy_frames", False)
        )
        self._apply_parameters()

    # -- 生命周期 ------------------------------------------------------------

    def _apply_parameters(self) -> None:
        """根据选项构建 VM 的参数/控制快照。"""
        params = dict(self.vm.parameters)
        params["DetectTypeTextSel"] = self.options.detect_type
        params["DetectInputSizeTextSel"] = str(self.options.detect_input_size)
        params["DetectScoreSlider"] = self.options.detect_score * 100
        params.update(self.options.overrides)
        self.vm.parameters = params

    def close(self) -> None:
        """释放由 CUDA 支撑的模型 session。"""
        try:
            self.models.delete_models()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "ImageSwapper":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # -- source embedding -----------------------------------------------------

    def source_embedding(self, path: str, *, return_kps: bool = False):
        """检测并识别 `path` 里的那一张脸。

        return_kps=True 时连同那张脸的 5 点关键点一起返回
        (embedding, kps)——姿态补偿需要参考图自身的姿态。"""
        rgb = _read_image(path)
        img_chw = torch.from_numpy(
            _frame_to_uint8(rgb, self.options.zero_copy_frames)
        ).to("cuda").permute(2, 0, 1)
        kpss = self.models.run_detect(
            img_chw,
            self.options.detect_type,
            max_num=1,
            score=self.options.detect_score,
            input_size=self.options.detect_input_size,
        )
        if len(kpss) == 0:
            raise HeadlessError(f"no face detected in source image: {path}")
        emb, _cropped = self.models.run_recognize(
            img_chw, kpss[0],
            dim=2 if self.options.recognize_dim2 else 1,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return (emb, kpss[0]) if return_kps else emb

    def source_embedding_from_many(
        self, paths: list[str],
    ) -> np.ndarray:
        """把多张源脸的 embedding 合并起来。"""
        embs = [self.source_embedding(p) for p in paths]
        if not embs:
            raise HeadlessError("no source faces supplied")
        return np.asarray(combine(embs, self.options.merge_mode), dtype=np.float32)

    # -- target 人脸检测 ------------------------------------------------------

    def _detect_face_slots(self, rgb: np.ndarray) -> list[dict]:
        """检测 `rgb` 里的每一张脸，每个不同的人返回一个 slot。

        复刻 GUI 的 "Find Faces" 步骤: 检测、识别，并用相似度阈值剔除近乎
        重复的检测结果。
        """
        threshold = float(self.vm.parameters.get("ThresholdSlider", 55))
        img_chw = torch.from_numpy(
            _frame_to_uint8(rgb, self.options.zero_copy_frames)
        ).to("cuda").permute(2, 0, 1)
        kpss = self.models.run_detect(
            img_chw,
            self.options.detect_type,
            # 必须与 VideoManager.swap_video 用的 max_num 一致，这样这里
            # 构建的 slot 才能与它重新检测到的脸对上号。
            max_num=10,
            score=self.options.detect_score,
            input_size=self.options.detect_input_size,
        )
        slots: list[dict] = []
        for kps in kpss:
            emb, _cropped = self.models.run_recognize(
                img_chw, kps,
                dim=2 if self.options.recognize_dim2 else 1,
            )
            emb = np.asarray(emb, dtype=np.float32)
            if any(
                _cosine_similarity_pct(emb, s["Embedding"]) >= threshold
                for s in slots
            ):
                continue
            slots.append({
                "Embedding": emb,
                "SourceFaceAssignments": [],
                "AssignedEmbedding": None,
                "HFCorrectionGap": None,
                "HFCorrectionGapSamples": 0,
                "HFRefinePending": False,
                # 一个 landmark 单位在原帧里跨多少个源像素。对齐后的裁剪图
                # 恒为 112x112，所以裁剪尺寸无法用来给脸排序。
                "Scale": _kps_scale(kps),
            })
        return slots


class ImageSwapper(_SwapperBase):
    """在静态图片中换脸。

    用法:

        with ImageSwapper(SwapOptions(models_folder="/data/models")) as swapper:
            swapper.swap(["source.jpg"], "target.jpg", "out.png")
    """

    # -- target 人脸选择 ------------------------------------------------------

    def _select_faces(self, slots: list[dict]) -> list[dict]:
        """把脸部策略应用到检测出的 slot 上。"""
        policy = (self.options.face_policy or "all").lower()
        if policy == "all":
            return slots
        if policy == "largest":
            if not slots:
                return []
            return [max(slots, key=lambda s: s["Scale"])]
        raise HeadlessError(
            f"unknown face policy: {self.options.face_policy!r} "
 "(expected 'all' or 'largest')"
        )

    # -- 换脸 -----------------------------------------------------------------

    def swap(
        self,
        source_paths: list[str],
        target_path: str,
        output_path: str,
        embedding: np.ndarray | None = None,
    ) -> dict:
        """在一张目标图片里换脸并写出结果。

        `embedding` 让调用方在整批处理中复用一个预先算好的源 embedding，
        而不必对每个 target 重新识别一遍源脸。

        返回一个小报告 dict（检测到的脸数、换掉的脸数……）。
        """
        if embedding is None:
            embedding = self.source_embedding_from_many(source_paths)
        rgb = _read_image(target_path)
        slots = self._detect_face_slots(rgb)
        if not slots:
            raise HeadlessError(f"no face detected in target image: {target_path}")
        selected = self._select_faces(slots)
        if not selected:
            raise HeadlessError(
                f"no target face selected by policy {self.options.face_policy!r}"
            )
        for slot in selected:
            slot["SourceFaceAssignments"] = list(source_paths)
            slot["AssignedEmbedding"] = embedding
        self.vm.found_faces = selected

        # swap_video 接收 RGB HxWx3 uint8（numpy）的帧，返回同形状的
        # CUDA tensor。
        result = self.vm.swap_video(rgb, 0)
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        out = np.ascontiguousarray(
            _frame_to_uint8(result, self.options.zero_copy_frames)
        )

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return {
            "detected_faces": len(slots),
            "swapped_faces": len(selected),
            "output": output_path,
        }

    def swap_batch(
        self,
        source_paths: list[str],
        target_paths: list[str],
        output_dir: str,
        suffix: str = "_swapped",
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[dict]:
        """把同一张（些）源脸换进许多张目标图片。"""
        embedding = self.source_embedding_from_many(source_paths)
        results: list[dict] = []
        total = len(target_paths)
        for index, target in enumerate(target_paths, start=1):
            stem, ext = os.path.splitext(os.path.basename(target))
            out_path = os.path.join(output_dir, f"{stem}{suffix}{ext or '.png'}")
            report = self.swap(source_paths, target, out_path, embedding=embedding)
            report["target"] = target
            results.append(report)
            if on_progress is not None:
                on_progress(index, total, target)
        return results


@dataclass
class VideoOptions(SwapOptions):
    """在共享换脸选项之上的视频专有选项。"""

    # 要替换掉**视频里那个人**的参考图。设置后，只有匹配这些身份的脸会被
    # 换，其余一概不动。为空时每一张检测到的脸都会被换。
    target_person_paths: list[str] = field(default_factory=list)
    # 一张检测脸要被认定为目标人物的最低相似度（0-100）。越高越严格。
    target_threshold: float = 55.0
    # 姿态自适应匹配阈值（PoseCompensationSlider）：全侧脸时判定线比
    # --target-threshold 低多少分。0 = 关闭，判定与旧行为逐字等价。
    pose_compensation: float = 0.0
    # H.264 质量: 数值越低越好。18 在视觉上接近无损，23 是交付时常用的
    # "够好就行" 默认值。
    crf: int = 18
    preset: str = "medium"
    # 保留源音频轨。纯视频源本来就没有音频流可映射。
    keep_audio: bool = True
    # 除了每第 N 帧之外都跳过换脸，原样透传其余帧。用于快速计时测试；
    # 输出仍保留每一帧，时长与音频因此保持同步。
    every_nth: int = 1
    # 并行处理帧的 worker 数。1 = 原来的单线程串行路径；>1 时每个 worker
    # 各持有一份 ORT session 和一条 CUDA stream，GPU 与显存占用随之上升。
    workers: int = 1
    # 打开分段计时，结束时输出 decode / swap / encode 的累计耗时。
    # 关闭时计时代码只多一次 bool 判断，不改变任何推理与编码行为。
    profile: bool = False
    # 把编码搬进独立线程，让 CPU 编码与 GPU 推理解耦。默认关闭；关闭时
    # _encode_video 走原来的主线程串行编码路径，输出与基线逐字一致。
    encode_thread: bool = False
    # 把 D2H（.cpu().numpy()）从 worker 线程推迟到帧的消费侧。只在
    # workers > 1 时有意义：串行路径没有可解放的 worker。默认关闭。
    # 不删除 swap_video 里的 wait_stream / Models 里的 syncvec 同步。
    defer_d2h: bool = False
    # True = 保留每帧逐脸 run_recognize 的旧路径，便于 A/B 与回滚。
    # 默认 False：无用的 ArcFace 推理会被跳过（像素输出不变）。
    force_recognize: bool = False
    # 视频编码器：libx264（默认）或 nvenc（需探测 h264_nvenc 可用）。
    encoder: str = "libx264"
    # 解码后端：pyav（默认，CPU 逐帧）或 gpu（torchcodec 常驻 CUDA，实验性）。
    decoder: str = "pyav"


@dataclass
class _EncodeResult:
    """``_encode_video`` 的结果。

    ``first_pts`` 供音频 mux 恢复视频流的起始偏移；``frames`` 是编码循环
    实际消费的帧数（容器声明的 stream.frames 可能为 0 或只是估算值，不能
    用来算吞吐）；``profile`` 未开启计时时为 None。
    """

    first_pts: int
    frames: int
    profile: dict[str, float] | None = None


def _find_ffmpeg() -> str:
    """返回系统 ffmpeg 路径，找不到就给出清晰的报错。"""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise HeadlessError(
            "ffmpeg not found on PATH. Install it with: "
            "sudo apt install -y ffmpeg"
        )
    return exe


class VideoSwapper(_SwapperBase):
    """把一个源身份换进一段视频。

    用法:

        with VideoSwapper(VideoOptions(models_folder="/data/models")) as sw:
            sw.swap(["source.jpg"], "in.mp4", "out.mp4")

    设置了 `options.target_person_paths` 时，只有匹配那些参考身份的脸会被
    换——视频里的其他人一概不动。不设置的话每一张检测到的脸都会被换。

    解码用 PyAV（CPU），编码用系统的 ffmpeg 可执行文件。音频从源直接拷贝
    过来，不重新编码。
    """

    def _target_embeddings(self) -> list[np.ndarray]:
        """要在视频里被替换掉的人的 embedding。"""
        return [
            self.source_embedding(path)
            for path in self.options.target_person_paths
        ]

    def _target_refs_with_kps(self) -> list[tuple[np.ndarray, object]]:
        """参考 embedding 连同检测时的 kps，供姿态补偿配对用。"""
        return [
            self.source_embedding(path, return_kps=True)
            for path in self.options.target_person_paths
        ]

    def swap(
        self,
        source_paths: list[str],
        video_path: str,
        output_path: str,
        embedding: np.ndarray | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        if not os.path.isfile(video_path):
            raise HeadlessError(f"video not found: {video_path}")
        ffmpeg = _find_ffmpeg()
        if embedding is None:
            embedding = self.source_embedding_from_many(source_paths)

        try:
            import av
        except ImportError as exc:
            raise HeadlessError(
                f"PyAV is not installed ({exc}). Install it with: pip install av"
            ) from exc

        # 开了姿态补偿才需要参考图的 kps（多一趟检测+识别）；关闭时与
        # 旧路径完全一致，不多跑推理。
        if self.options.pose_compensation > 0 and self.options.target_person_paths:
            refs = self._target_refs_with_kps()
            targets = [emb for emb, _kps in refs]
            ref_kps = [kps for _emb, kps in refs]
        else:
            targets = self._target_embeddings()
            ref_kps = None
        # 只有匹配参考图的脸会被换。未提供参考图时每一张脸都会被换。
        self.vm.force_source_embedding = embedding
        self.vm.target_face_embeddings = targets or None
        self.vm.target_face_ref_kps = ref_kps
        self.vm.target_face_threshold = self.options.target_threshold
        self.vm.target_face_pose_compensation = float(
            getattr(self.options, "pose_compensation", 0.0)
        )

        # 阶段 5b：解析 --decoder。默认 pyav 分支不调用 probe，既不 import
        # torchcodec 也不碰 CUDA；gpu 探测失败会明确报错，不静默回退。
        decoder_plan = resolve_decoder(
            getattr(self.options, "decoder", "pyav"),
            self._gpu_decode_probe,
        )

        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            fps = self._fps(stream)
            total = int(stream.frames or 0)
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            audio_stream = (
                container.streams.audio[0] if container.streams.audio else None
            )
            has_audio = audio_stream is not None and self.options.keep_audio
            if total <= 0:
                # 有些容器不声明帧数；回退到 duration，好让进度输出仍有
                # 一个分母。
                duration = float(stream.duration or container.duration or 0)
                time_base = float(stream.time_base or 0)
                total = int(duration * time_base * fps) if duration else 0

            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="faceswapper-video-") as tmp:
                silent = os.path.join(tmp, "silent.mp4")
                encoded = self._encode_video(
                    av, container, stream, silent, width, height, fps,
                    total, on_progress, video_path, decoder_plan,
                )
                # 源视频流的时间偏移，单位为秒。重建出的视频从 0 开始，
                # 所以要把它挪回原视频相对音频所在的位置。
                video_offset = (
                    float(encoded.first_pts * stream.time_base)
                    if encoded.first_pts and stream.time_base else 0.0
                )
                audio_offset = self._stream_start_seconds(audio_stream)
                self._mux_audio(
                    ffmpeg, silent, video_path, output_path, has_audio,
                    video_offset=video_offset,
                    audio_offset=audio_offset,
                )

            report = {
                "output": output_path,
                # 容器声明或按 duration 估算的帧数：done 行沿用原语义，不改。
                "frames": total,
                # 编码循环实际消费的帧数：容器不声明帧数时仍准确，供 avg fps 用。
                "processed_frames": encoded.frames,
                "fps": fps,
                "width": width,
                "height": height,
                "audio": has_audio,
            }
            if encoded.profile is not None:
                report["profile"] = encoded.profile
            return report
        finally:
            container.close()
            self.vm.force_source_embedding = None
            self.vm.target_face_embeddings = None

    @staticmethod
    def _fps(stream) -> float:
        rate = stream.average_rate or stream.base_rate
        if rate:
            return float(rate)
        # time_base 是时间戳粒度（常见为 1/90000），**不是**帧率——
        # 绝不能由它推导 fps。rate 元数据缺失时，回退到声明的帧数与
        # duration。
        duration = stream.duration
        frames = stream.frames
        tb = stream.time_base
        if duration and frames and tb:
            return float(frames) / (float(duration) * float(tb))
        return 30.0

    @staticmethod
    def _stream_start_seconds(stream) -> float:
        """一条流相对于容器的起始时间，单位为秒。

        PyAV 以流自身的 time base 暴露 `start_time`。起始较晚的流该值为
        正；两条轨重新合回来时这个偏移必须保留。
        """
        if stream is None:
            return 0.0
        start = getattr(stream, "start_time", None)
        tb = getattr(stream, "time_base", None)
        if start is None or tb is None:
            return 0.0
        try:
            return float(start * tb)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _gpu_decode_probe() -> tuple[bool, str]:
        """真机探测 torchcodec + CUDA 是否可用，供 resolve_decoder 注入。"""
        import importlib.util
        import inspect

        if importlib.util.find_spec("torchcodec") is None:
            return False, "torchcodec>=0.4 is not installed"
        try:
            from torchcodec.decoders import VideoDecoder
            params = inspect.signature(VideoDecoder.__init__).parameters
            if "device" not in params:
                return False, "VideoDecoder has no device= parameter"
        except Exception as exc:
            return False, f"torchcodec import failed: {exc}"
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        return True, "ok"

    @staticmethod
    def _normalize_gpu_rgb(data):
        """把 torchcodec 的 frame data 归一成 HWC uint8 CUDA tensor。

        兼容 CHW/HWC 与 float 0-1 / uint8 两种常见布局；非 tensor 直接报错，
        不猜。
        """
        if not isinstance(data, torch.Tensor):
            raise HeadlessError(
                "GPU decoder frame data is not a torch.Tensor; "
                "use the default --decoder pyav"
            )
        rgb = data
        if rgb.device.type != "cuda":
            rgb = rgb.to("cuda")
        if rgb.dtype != torch.uint8:
            rgb = (rgb.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        if rgb.ndim == 3 and rgb.shape[0] == 3:  # CHW -> HWC
            rgb = rgb.permute(1, 2, 0)
        return rgb.contiguous()

    def _decoded_gpu(self, video_path: str, in_tb: Fraction, gpu_state: dict):
        """实验性 GPU 常驻解码，产出 ``(index, rgb, pts)``。

        **API 假设（需真机核对）**：``VideoDecoder(path, device="cuda")`` 可
        迭代；frame 的 data 是 CUDA tensor；pts 是与 ``in_tb`` 同单位的整数
        ticks。任何一条不满足都明确抛错，绝不静默用等间隔 PTS 兜底——那会
        悄悄破坏 VFR 时序与音画同步。
        """
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path, device="cuda")
        index = 0
        first_source_pts = None
        for raw in decoder:
            data, pts = frame_fields(raw)
            if pts is None:
                raise HeadlessError(
                    "GPU decoder frame has no PTS; VFR-safe decode requires "
                    "explicit timestamps — use --decoder pyav"
                )
            if isinstance(pts, float) and not pts.is_integer():
                raise HeadlessError(
                    "GPU decoder returned fractional PTS (likely seconds); "
                    "unit conversion is not implemented — use --decoder pyav"
                )
            source_pts = int(pts)
            if first_source_pts is None:
                first_source_pts = source_pts
                gpu_state["first_pts"] = source_pts
            yield index, self._normalize_gpu_rgb(data), source_pts - first_source_pts
            index += 1

    def _encode_video(
        self,
        av,
        container,
        in_stream,
        silent_path: str,
        width: int,
        height: int,
        fps: float,
        total: int,
        on_progress: Callable[[int, int], None] | None,
        video_path: str,
        decoder_plan: DecoderPlan,
    ) -> _EncodeResult:
        """解码每一帧、换脸，并按原始 PTS 编码。

        与按固定 `-r` 把原始帧灌给 ffmpeg 不同，这里保留每一帧的呈现
        时间戳。因此可变帧率源能保持自己的原始时序，拷贝过来的音频轨也与
        视频保持对齐。

        ``--profile`` 打开时累加 decode / swap / encode 三段耗时。串行路径
        下三段近似串行、之和接近总用时；workers > 1 时它们在不同线程上
        重叠，之和会大于总用时，只能用来判断瓶颈落在哪一段，不要相加。
        """
        profile = bool(getattr(self.options, "profile", False))
        times: dict[str, float] = {"decode": 0.0, "swap": 0.0, "encode": 0.0}
        times_lock = threading.Lock()
        processed = 0
        workers = min(8, max(1, int(getattr(self.options, "workers", 1))))
        # GPU 解码路径由自己的生成器负责记录首帧 PTS（PyAV 路径写在
        # decoded() 的 nonlocal 里），两条路径的音画偏移都要能恢复。
        gpu_state: dict = {"first_pts": None}

        def resolved_first_pts() -> int:
            if decoder_plan.backend == "gpu":
                return gpu_state["first_pts"] or 0
            return first_pts if first_pts is not None else 0

        parallel = workers > 1
        # 阶段 3 的 6.1：只有并行路径才有可解放的 worker；串行时推迟 D2H
        # 没有意义，自动退回原行为。同步本身一个都不删。
        defer_d2h = bool(getattr(self.options, "defer_d2h", False)) and parallel

        def add_time(key: str, seconds: float) -> None:
            # swap 由 ThreadPoolExecutor 的多个 worker 并发记账，累加必须
            # 串行化；另两段只在主线程调用，共用这把锁的开销可忽略。
            if not profile:
                return
            with times_lock:
                times[key] += seconds

        def timed_swap(rgb):
            # 串行路径在 decoded() 里调用，并行路径由 ordered_swap_frames
            # 调用；包一层让两条路径共用同一个计时入口，后者签名不用改。
            if not profile:
                return self._swap_frame(rgb, defer=defer_d2h)
            t0 = time.monotonic()
            try:
                return self._swap_frame(rgb, defer=defer_d2h)
            finally:
                add_time("swap", time.monotonic() - t0)

        in_tb = in_stream.time_base
        if in_tb is None:
            in_tb = Fraction(1, 1000)

        # 第一帧的时间戳，以源 time-base 为单位。返回它，好让 mux 步骤
        # 能把重建视频放到原视频流所处的同一偏移上。
        first_pts: int | None = None

        # yuv420p 需要偶数尺寸；尺寸为奇数的源回退到 4:4:4，而不是悄悄
        # 改掉分辨率。
        pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"

        # 阶段 5 的编码侧：解析 --encoder。libx264 分支不会触碰 av.codec，
        # 只有显式 nvenc 才通过惰性 lambda 去探测 h264_nvenc；探测失败会
        # 抛 HeadlessError，不静默回退成 libx264。
        plan = resolve_encoder(
            getattr(self.options, "encoder", "libx264"),
            lambda name, mode: av.codec.Codec(name, mode),
            crf=self.options.crf,
            preset=self.options.preset,
        )

        # output 由实际使用的那条编码路径创建：默认路径在主线程创建，
        # --encode-thread 路径在编码线程里创建（PyAV 对象不跨线程）。
        output = None
        try:
            # 给极少数不带 PTS 的帧准备的兜底步长。
            fallback_step = max(
                1, int(round((1.0 / max(fps, 1e-6)) / float(in_tb)))
            )
            # 输出被归一化为从 0 开始。源视频原本的起始偏移只在 mux 时
            # 通过 -itsoffset 补回一次；在这里也补一次会让位移翻倍。
            first_source_pts: int | None = None
            next_pts: int | None = None
            step = max(1, int(self.options.every_nth))
            encode_thread = bool(
                getattr(self.options, "encode_thread", False)
            )

            def decoded():
                """按解码顺序产出 (index, rgb, pts)。

                PTS 算法与串行路径完全一致；串行时换脸也在这里做，并行时
                交给 ordered_swap_frames 提前提交给 worker 线程池。
                """
                nonlocal first_pts, first_source_pts, next_pts
                index = 0
                for frame in container.decode(in_stream):
                    t0 = time.monotonic() if profile else 0.0
                    rgb = frame.to_ndarray(format="rgb24")
                    if not parallel and index % step == 0:
                        # 串行换脸归 swap 段：先结清解码耗时，换完再重新起表，
                        # 后面的 PTS 计算仍记回 decode，两段不会重复计数。
                        if profile:
                            add_time("decode", time.monotonic() - t0)
                        rgb = timed_swap(rgb)
                        if profile:
                            t0 = time.monotonic()
                    if frame.pts is not None:
                        source_pts = int(frame.pts)
                        if first_source_pts is None:
                            first_source_pts = source_pts
                            first_pts = source_pts
                        # 相对时间戳: 第一帧 -> 0，其余帧保持精确的间距
                        # （因而也保留了 VFR 时序）。
                        pts = source_pts - first_source_pts
                        next_pts = source_pts + fallback_step
                    elif next_pts is not None:
                        source_pts = next_pts
                        pts = source_pts - (first_source_pts or 0)
                        next_pts = source_pts + fallback_step
                    else:
                        # 连第一帧也没有 PTS；直接从 0 开始。
                        pts = 0
                        first_source_pts = 0
                        next_pts = fallback_step
                    if profile:
                        add_time("decode", time.monotonic() - t0)
                    yield index, rgb, pts
                    index += 1

            if decoder_plan.backend == "gpu":
                # 实验性 GPU 常驻路径；默认 pyav 完全不触碰 torchcodec。
                frames = self._decoded_gpu(video_path, in_tb, gpu_state)
            else:
                frames = decoded()
            if parallel:
                # 调度器刻意放在 stdlib-only 的 frame_order.py，保序行为
                # 由 tests/test_frame_order.py 用假 swap_fn 做行为测试。
                frames = ordered_swap_frames(
                    frames, workers, step, timed_swap,
                )

            if defer_d2h:
                # 阶段 3 的 6.1：把 D2H 从 worker 线程挪到帧的消费侧，
                # worker 不再被 .cpu().numpy() 阻塞，可立刻接下一帧。
                #
                # 顺序仍由 VideoManager.swap_video 里的
                # default_stream().wait_stream(worker_stream) 保证：消费侧的
                # .cpu() 跑在默认 stream 上，一定读到已写完的缓冲。
                # Models 里的 syncvec.cpu() 与 user_compute_stream 一个都没动，
                # 这是文档 6.1 的红线。
                #
                # 这段 D2H 耗时不计入 --profile 的三段（既非解码、非 swap、
                # 也非编码），开启后三段之和会更小于总用时。
                frames = (
                    (index, self._finalize_frame(rgb), pts)
                    for index, rgb, pts in frames
                )

            if encode_thread:
                # 编码线程自己创建/关闭 PyAV 对象；主线程只按序投递帧。
                # decoded() 与 PTS 算法共用上面这一份，不复制第二套。
                processed = self._encode_via_thread(
                    av, frames, silent_path, width, height, fps, in_tb,
                    pix_fmt, total, on_progress, workers,
                    add_time if profile else None, plan,
                )
                return _EncodeResult(
                    first_pts=resolved_first_pts(),
                    frames=processed,
                    profile=dict(times) if profile else None,
                )

            output = av.open(
                silent_path, mode="w", format="mp4",
                options={"movflags": "+faststart"},
            )
            # PyAV >= 14 要求 `rate` 是 Fraction，不能是 float。
            out_stream = output.add_stream(
                plan.codec, rate=Fraction(fps).limit_denominator(1000000)
            )
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = pix_fmt
            out_stream.options = dict(plan.stream_options)
            # 保留源 time base，这样 PTS 值可以直接照抄，不累积舍入
            # 误差。
            out_stream.time_base = in_tb
            out_tb = out_stream.time_base or in_tb
            pts_scale = float(in_tb / out_tb)

            for index, rgb, pts in frames:
                t0 = time.monotonic() if profile else 0.0
                new_frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(rgb), format="rgb24",
                )
                new_frame.pts = int(round(pts * pts_scale))
                new_frame.time_base = out_tb
                for packet in out_stream.encode(new_frame):
                    output.mux(packet)
                if profile:
                    add_time("encode", time.monotonic() - t0)

                processed += 1
                index += 1
                if on_progress is not None and (
                    index % 10 == 0 or index == total
                ):
                    on_progress(index, total or index)

            for packet in out_stream.encode(None):
                output.mux(packet)
        finally:
            # --encode-thread 路径没有创建 output，PyAV 的关闭由编码线程负责。
            if output is not None:
                output.close()
        return _EncodeResult(
            first_pts=resolved_first_pts(),
            frames=processed,
            profile=dict(times) if profile else None,
        )

    def _encode_via_thread(
        self,
        av,
        frames,
        silent_path: str,
        width: int,
        height: int,
        fps: float,
        in_tb: Fraction,
        pix_fmt: str,
        total: int,
        on_progress: Callable[[int, int], None] | None,
        workers: int,
        add_time: Callable[[str, float], None] | None,
        plan: EncoderPlan,
    ) -> int:
        """把编码放到独立线程；返回实际编码的帧数。

        主线程按解码顺序把 ``(display_index, rgb, pts)`` 投进有界队列，编码
        线程按 FIFO 取用，因此 PTS / VFR 时序与串行路径一致。队列
        ``maxsize=2*workers`` 提供背压，主机内存不会随视频长度膨胀。

        异常路径不投 sentinel，而是设 ``stop_event`` 让编码线程在 get 超时
        后自行退出；非 daemon 线程一定 join，保证 output.close() 发生在
        音频 mux 之前。
        """
        import queue

        from faceswapper.encoder import (
            EncoderFailure,
            encoder_worker,
            submit_frames,
        )

        items: queue.Queue = queue.Queue(maxsize=max(2, 2 * workers))
        stop_event = threading.Event()
        errors: list[BaseException] = []
        encoded = [0]

        def on_encoded() -> None:
            encoded[0] += 1

        thread = threading.Thread(
            target=encoder_worker,
            kwargs={
                "av": av,
                "silent_path": silent_path,
                "width": width,
                "height": height,
                "fps": fps,
                "in_tb": in_tb,
                "pix_fmt": pix_fmt,
                "crf": self.options.crf,
                "preset": self.options.preset,
                "items": items,
                "stop_event": stop_event,
                "errors": errors,
                "on_progress": on_progress,
                "total": total,
                "on_encoded": on_encoded,
                "add_time": add_time,
                "codec": plan.codec,
                "stream_options": dict(plan.stream_options),
            },
            name="faceswapper-encoder",
            daemon=False,
        )
        thread.start()
        finished = False
        try:
            for index, rgb, pts in frames:
                # PyAV 要求 contiguous。放在主线程做，编码线程因此不依赖
                # numpy，也便于用假 av 桩做离线测试。
                submit_frames(
                    items, (index + 1, np.ascontiguousarray(rgb), pts),
                    stop_event, errors,
                )
            submit_frames(items, None, stop_event, errors)
            finished = True
        finally:
            if not finished:
                stop_event.set()
            thread.join(timeout=30.0)
            if thread.is_alive():
                errors.append(RuntimeError("encoder thread did not stop"))

        if errors:
            raise EncoderFailure(errors[0])
        return encoded[0]

    def _swap_frame(self, rgb: np.ndarray, *, defer: bool = False):
        """跑一次逐帧换脸。

        ``defer=True`` 时把 CUDA tensor 原样交回调用方，由 ``_finalize_frame``
        在消费侧取回 CPU。默认 False，行为与改造前逐字一致。
        """
        result = self.vm.swap_video(rgb, 0)
        if defer:
            return result
        return self._finalize_frame(result)

    def _finalize_frame(self, rgb) -> np.ndarray:
        """把换脸结果取回 CPU，做成 PyAV 能直接吃的 contiguous uint8 数组。

        对 CUDA tensor 走 ``detach().cpu().numpy()`` 再转 uint8，与原
        ``_swap_frame`` 一致；对已经是 numpy 的帧（例如 ``--every-nth`` 跳过
        换脸的帧）只保证 contiguous，不额外 astype，避免改变像素。
        """
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
            return np.ascontiguousarray(
                _frame_to_uint8(rgb, self.options.zero_copy_frames)
            )
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _mux_audio(
        ffmpeg: str,
        silent_path: str,
        source_path: str,
        output_path: str,
        has_audio: bool,
        video_offset: float = 0.0,
        audio_offset: float = 0.0,
    ) -> None:
        """把源音频轨拷到编码好的视频上。

        重建出的视频从 0 开始，而拷贝的音频保留自己的时间戳，所以两个
        原始起始偏移都要用 `-itsoffset` 补回。少了这一步，一个视频与音频
        并非同一时刻开始的源就会恰好相差这个差值而错位。
        """
        video_args = (
            ["-itsoffset", f"{video_offset:.6f}"] if video_offset else []
        )
        audio_args = (
            ["-itsoffset", f"{audio_offset:.6f}"] if audio_offset else []
        )
        if has_audio:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                *video_args,
                "-i", silent_path,
                *audio_args,
                "-i", source_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "copy",
                "-shortest",
                "-avoid_negative_ts", "make_zero",
                output_path,
            ]
        else:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                *video_args,
                "-i", silent_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                output_path,
            ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise HeadlessError(f"ffmpeg mux failed:\n{err}")
