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
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import cv2
import numpy as np
import torch

from faceswapper.EmbeddingMerge import combine
from faceswapper.errors import HeadlessError
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


def _read_image(path: str) -> np.ndarray:
    """把图片读成 RGB uint8。失败时报错。"""
    if not os.path.isfile(path):
        raise HeadlessError(f"image not found: {path}")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HeadlessError(f"could not decode image (unsupported format?): {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
        if options.backend == "cuda":
            # 这里的 "onnx" 指的是 faceswapper.Models 里那条纯 CUDA
            # ExecutionProvider 路径，不是 CPU provider。这个命名继承自
            # GUI 的 per-model 后端开关。
            for attr in ("swapper_model", "retinaface_model", "recognition_model"):
                self.models.set_backend_preference(attr, "onnx", unload=False)
        self.vm = VideoManager(self.models)
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

    def source_embedding(self, path: str) -> np.ndarray:
        """检测并识别 `path` 里的那一张脸。"""
        rgb = _read_image(path)
        img_chw = torch.from_numpy(rgb.astype("uint8")).to("cuda").permute(2, 0, 1)
        kpss = self.models.run_detect(
            img_chw,
            self.options.detect_type,
            max_num=1,
            score=self.options.detect_score,
            input_size=self.options.detect_input_size,
        )
        if len(kpss) == 0:
            raise HeadlessError(f"no face detected in source image: {path}")
        emb, _cropped = self.models.run_recognize(img_chw, kpss[0])
        return np.asarray(emb, dtype=np.float32)

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
        img_chw = torch.from_numpy(rgb.astype("uint8")).to("cuda").permute(2, 0, 1)
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
            emb, _cropped = self.models.run_recognize(img_chw, kps)
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
        out = np.ascontiguousarray(result.astype(np.uint8))

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
    # H.264 质量: 数值越低越好。18 在视觉上接近无损，23 是交付时常用的
    # "够好就行" 默认值。
    crf: int = 18
    preset: str = "medium"
    # 保留源音频轨。纯视频源本来就没有音频流可映射。
    keep_audio: bool = True
    # 除了每第 N 帧之外都跳过换脸，原样透传其余帧。用于快速计时测试；
    # 输出仍保留每一帧，时长与音频因此保持同步。
    every_nth: int = 1


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

        targets = self._target_embeddings()
        # 只有匹配参考图的脸会被换。未提供参考图时每一张脸都会被换。
        self.vm.force_source_embedding = embedding
        self.vm.target_face_embeddings = targets or None
        self.vm.target_face_threshold = self.options.target_threshold

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
                first_pts = self._encode_video(
                    av, container, stream, silent, width, height, fps,
                    total, on_progress,
                )
                # 源视频流的时间偏移，单位为秒。重建出的视频从 0 开始，
                # 所以要把它挪回原视频相对音频所在的位置。
                video_offset = (
                    float(first_pts * stream.time_base)
                    if first_pts and stream.time_base else 0.0
                )
                audio_offset = self._stream_start_seconds(audio_stream)
                self._mux_audio(
                    ffmpeg, silent, video_path, output_path, has_audio,
                    video_offset=video_offset,
                    audio_offset=audio_offset,
                )

            return {
                "output": output_path,
                "frames": total,
                "fps": fps,
                "width": width,
                "height": height,
                "audio": has_audio,
            }
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
    ) -> int:
        """解码每一帧、换脸，并按原始 PTS 编码。

        与按固定 `-r` 把原始帧灌给 ffmpeg 不同，这里保留每一帧的呈现
        时间戳。因此可变帧率源能保持自己的原始时序，拷贝过来的音频轨也与
        视频保持对齐。
        """
        in_tb = in_stream.time_base
        if in_tb is None:
            in_tb = Fraction(1, 1000)

        # 第一帧的时间戳，以源 time-base 为单位。返回它，好让 mux 步骤
        # 能把重建视频放到原视频流所处的同一偏移上。
        first_pts: int | None = None

        # yuv420p 需要偶数尺寸；尺寸为奇数的源回退到 4:4:4，而不是悄悄
        # 改掉分辨率。
        pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"

        output = av.open(
            silent_path, mode="w", format="mp4",
            options={"movflags": "+faststart"},
        )
        try:
            # PyAV >= 14 要求 `rate` 是 Fraction，不能是 float。
            out_stream = output.add_stream(
                "libx264", rate=Fraction(fps).limit_denominator(1000000)
            )
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = pix_fmt
            out_stream.options = {
                "crf": str(int(self.options.crf)),
                "preset": str(self.options.preset),
            }
            # 保留源 time base，这样 PTS 值可以直接照抄，不累积舍入
            # 误差。
            out_stream.time_base = in_tb
            out_tb = out_stream.time_base or in_tb
            pts_scale = float(in_tb / out_tb)

            # 给极少数不带 PTS 的帧准备的兜底步长。
            fallback_step = max(
                1, int(round((1.0 / max(fps, 1e-6)) / float(in_tb)))
            )
            # 输出被归一化为从 0 开始。源视频原本的起始偏移只在 mux 时
            # 通过 -itsoffset 补回一次；在这里也补一次会让位移翻倍。
            first_source_pts: int | None = None
            next_pts: int | None = None
            index = 0
            step = max(1, int(self.options.every_nth))

            for frame in container.decode(in_stream):
                rgb = frame.to_ndarray(format="rgb24")
                if index % step == 0:
                    rgb = self._swap_frame(rgb)

                new_frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(rgb), format="rgb24",
                )
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
                new_frame.pts = int(round(pts * pts_scale))
                new_frame.time_base = out_tb
                for packet in out_stream.encode(new_frame):
                    output.mux(packet)

                index += 1
                if on_progress is not None and (
                    index % 10 == 0 or index == total
                ):
                    on_progress(index, total or index)

            for packet in out_stream.encode(None):
                output.mux(packet)
        finally:
            output.close()
        return first_pts if first_pts is not None else 0

    def _swap_frame(self, rgb: np.ndarray) -> np.ndarray:
        result = self.vm.swap_video(rgb, 0)
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        return np.ascontiguousarray(result.astype(np.uint8))

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
