"""Headless (no-GUI) face-swap pipelines.

The desktop GUI drove VideoManager through an interactive loop: scrub to
a frame, "Find Faces", click a source face to bind it to a slot, then
play/record. None of that is expressible on a command line, so this
module provides the equivalent batch paths:

    image:  detect faces -> assign source -> swap_video -> write PNG
    video:  decode frame -> swap every face -> encode -> mux audio

The heavy lifting is shared: detection, recognition and the actual swap
all go through rope.Models and VideoManager, so the CLI produces the
same pixels as the desktop app for the same parameters.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import torch

from rope.EmbeddingMerge import combine
from rope.errors import HeadlessError
from rope.Models import Models
from rope.VideoManager import VideoManager


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class SwapOptions:
    """Everything the image swap needs, resolved from CLI arguments."""

    models_folder: str = "./models"
    # Which target faces get swapped. "all" swaps every detected face,
    # "largest" only the biggest one.
    face_policy: str = "all"
    detect_type: str = "Retinaface"
    detect_input_size: int = 640
    detect_score: float = 0.5
    # "Mean" | "Median" | "Sph" | "Geo" | "Qual" - how several source
    # faces are combined into one identity vector.
    merge_mode: str = "Mean"
    # Extra parameters layered on top of the defaults (and on top of
    # saved_parameters.json when one is supplied).
    overrides: dict[str, Any] = field(default_factory=dict)
    # Force the ONNX/CUDA backend instead of TensorRT. TRT is the
    # default in the GUI, but its Linux engine cache is not portable
    # from a Windows install, so the CLI stays on CUDA EP unless asked.
    backend: str = "cuda"


def _read_image(path: str) -> np.ndarray:
    """Read an image as RGB uint8. Returns None-ish error on failure."""
    if not os.path.isfile(path):
        raise HeadlessError(f"image not found: {path}")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HeadlessError(f"could not decode image (unsupported format?): {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _cosine_similarity_pct(a: np.ndarray, b: np.ndarray) -> float:
    """0..100 similarity, 100 = identical. Mirrors the GUI's helper."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cos_dist = 1.0 - float(np.dot(a, b)) / denom
    return 100.0 - cos_dist * 50.0


def _kps_scale(kps: np.ndarray) -> float:
    """Estimate how large a face is in the original frame.

    The detector returns 5 landmarks in pixel coordinates. The distance
    between the eyes grows with the face size, so it is a good proxy for
    ranking faces by how big they are on screen.
    """
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(kps[0] - kps[1]))


class _SwapperBase:
    """Model/VM setup and source-face handling shared by image and video."""

    def __init__(self, options: SwapOptions) -> None:
        self.options = options
        self.models = Models()
        self.models.set_models_folder(options.models_folder)
        if options.backend == "cuda":
            # "onnx" here means the plain CUDA ExecutionProvider path in
            # rope.Models, not the CPU provider. Naming is inherited from
            # the GUI's per-model backend toggle.
            for attr in ("swapper_model", "retinaface_model", "recognition_model"):
                self.models.set_backend_preference(attr, "onnx", unload=False)
        self.vm = VideoManager(self.models)
        self._apply_parameters()

    # -- lifecycle ------------------------------------------------------------

    def _apply_parameters(self) -> None:
        """Build the VM parameter/control snapshots from the options."""
        params = dict(self.vm.parameters)
        params["DetectTypeTextSel"] = self.options.detect_type
        params["DetectInputSizeTextSel"] = str(self.options.detect_input_size)
        params["DetectScoreSlider"] = self.options.detect_score * 100
        params["MergeTextSel"] = self.options.merge_mode
        params.update(self.options.overrides)
        self.vm.parameters = params

        control = dict(self.vm.control)
        # SwapFacesButton gates whether swap_video actually swaps; MaskView
        # would render a side-by-side mask preview instead of the result.
        control["SwapFacesButton"] = True
        control["MaskViewButton"] = False
        self.vm.control = control

    def close(self) -> None:
        """Release the CUDA-backed model sessions."""
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
        """Detect + recognize the single face in `path`."""
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
        """Combine the embeddings of several source faces."""
        embs = [self.source_embedding(p) for p in paths]
        if not embs:
            raise HeadlessError("no source faces supplied")
        return np.asarray(combine(embs, self.options.merge_mode), dtype=np.float32)

    # -- target face detection ------------------------------------------------

    def _detect_face_slots(self, rgb: np.ndarray) -> list[dict]:
        """Detect every face in `rgb` and return one slot per distinct face.

        Mirrors the GUI's "Find Faces" step: detect, recognize, and drop
        near-duplicate detections using the similarity threshold.
        """
        threshold = float(self.vm.parameters.get("ThresholdSlider", 55))
        img_chw = torch.from_numpy(rgb.astype("uint8")).to("cuda").permute(2, 0, 1)
        kpss = self.models.run_detect(
            img_chw,
            self.options.detect_type,
            # Must match the max_num VideoManager.swap_video uses, so the
            # slots built here line up with the faces it re-detects.
            max_num=10,
            score=self.options.detect_score,
            input_size=self.options.detect_input_size,
        )
        slots: list[dict] = []
        for kps in kpss:
            emb, cropped = self.models.run_recognize(img_chw, kps)
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
                "Thumbnail": cropped.cpu().numpy().astype(np.uint8),
                "HFCorrectionGap": None,
                "HFCorrectionGapSamples": 0,
                "HFRefinePending": False,
                # How many source pixels one landmark unit spans in the
                # original frame. The aligned crop is always 112x112, so
                # the crop size can't be used to rank faces.
                "Scale": _kps_scale(kps),
            })
        return slots


class ImageSwapper(_SwapperBase):
    """Swaps faces in still images.

    Usage:

        with ImageSwapper(SwapOptions(models_folder="/data/models")) as swapper:
            swapper.swap(["source.jpg"], "target.jpg", "out.png")
    """

    # -- target face selection ------------------------------------------------

    def _select_faces(self, slots: list[dict]) -> list[dict]:
        """Apply the face policy to the detected slots."""
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

    # -- the swap -------------------------------------------------------------

    def swap(
        self,
        source_paths: list[str],
        target_path: str,
        output_path: str,
        embedding: np.ndarray | None = None,
    ) -> dict:
        """Swap faces in one target image and write the result.

        `embedding` lets a caller reuse a pre-computed source embedding
        across a batch instead of re-recognizing the source faces per
        target.

        Returns a small report dict (detected faces, swapped faces, ...).
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

        # swap_video takes the frame as RGB HxWx3 uint8 (numpy) and
        # returns the same shape as a CUDA tensor.
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
        """Swap the same source face(s) into many target images."""
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
    """Video-specific options on top of the shared swap options."""

    # Reference images of the person to replace IN THE VIDEO. When set,
    # only faces matching these identities are swapped; everyone else is
    # left untouched. When empty, every detected face is swapped.
    target_person_paths: list[str] = field(default_factory=list)
    # Minimum similarity (0-100) for a detected face to count as the
    # target person. Higher = stricter.
    target_threshold: float = 55.0
    # H.264 quality: lower is better. 18 is visually near-lossless,
    # 23 is the usual "good enough" default for delivery.
    crf: int = 18
    preset: str = "medium"
    # Keep the source audio track. Video-only sources simply have no
    # audio stream to map.
    keep_audio: bool = True
    # Skip the swap on all but every Nth frame, passing the original
    # frames through instead. Useful for a quick timing test; the output
    # keeps every frame so duration and audio stay in sync.
    every_nth: int = 1


def _find_ffmpeg() -> str:
    """Return the system ffmpeg path or raise a clear error."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise HeadlessError(
            "ffmpeg not found on PATH. Install it with: "
            "sudo apt install -y ffmpeg"
        )
    return exe


class VideoSwapper(_SwapperBase):
    """Swaps one source identity into a video.

    Usage:

        with VideoSwapper(VideoOptions(models_folder="/data/models")) as sw:
            sw.swap(["source.jpg"], "in.mp4", "out.mp4")

    When `options.target_person_paths` is set, only faces matching those
    reference identities are swapped - the other people in the video are
    left untouched. Without it, every detected face is swapped.

    Decoding uses PyAV (CPU), encoding uses the system ffmpeg binary.
    Audio is copied through from the source rather than re-encoded.
    """

    def _target_embeddings(self) -> list[np.ndarray]:
        """Embeddings of the person(s) to replace in the video."""
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
        # Only faces matching the reference images are swapped. When no
        # reference is supplied, every face is swapped.
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
                # Some containers do not declare a frame count; fall back to
                # duration so progress output still has a denominator.
                duration = float(stream.duration or container.duration or 0)
                time_base = float(stream.time_base or 0)
                total = int(duration * time_base * fps) if duration else 0

            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="rope-video-") as tmp:
                silent = os.path.join(tmp, "silent.mp4")
                first_pts = self._encode_video(
                    av, container, stream, silent, width, height, fps,
                    total, on_progress,
                )
                # Offset of the source video stream, in seconds. The
                # rebuilt video starts at 0, so shift it back to where the
                # original sat relative to the audio.
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
        # time_base is the timestamp granularity (often 1/90000), NOT the
        # frame rate - never derive fps from it. Fall back to the declared
        # frame count and duration when the rate metadata is missing.
        duration = stream.duration
        frames = stream.frames
        tb = stream.time_base
        if duration and frames and tb:
            return float(frames) / (float(duration) * float(tb))
        return 30.0

    @staticmethod
    def _stream_start_seconds(stream) -> float:
        """Start time of a stream in seconds, relative to the container.

        PyAV exposes `start_time` in the stream's own time base. A stream
        that begins late carries a positive value; that offset must be
        preserved when the two tracks are recombined.
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
        """Decode every frame, swap it, and encode with its original PTS.

        Unlike piping raw frames into ffmpeg at a fixed `-r`, this keeps
        each frame's presentation timestamp. Variable-frame-rate sources
        therefore keep their original timing, and the copied audio track
        stays aligned with the video.
        """
        in_tb = in_stream.time_base
        if in_tb is None:
            from fractions import Fraction
            in_tb = Fraction(1, 1000)

        # Timestamp of the first frame, in source time-base units. It is
        # returned so the mux step can place the rebuilt video at the same
        # offset the original video stream had.
        first_pts: int | None = None

        # yuv420p needs even dimensions; fall back to 4:4:4 for odd-sized
        # sources rather than silently changing the resolution.
        pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"

        output = av.open(
            silent_path, mode="w", format="mp4",
            options={"movflags": "+faststart"},
        )
        try:
            out_stream = output.add_stream("libx264", rate=fps)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = pix_fmt
            out_stream.options = {
                "crf": str(int(self.options.crf)),
                "preset": str(self.options.preset),
            }
            # Preserve the source time base so PTS values can be copied
            # through without accumulating rounding error.
            out_stream.time_base = in_tb
            out_tb = out_stream.time_base or in_tb
            pts_scale = float(in_tb / out_tb)

            # Fallback cadence for the rare frames that carry no PTS.
            fallback_step = max(
                1, int(round((1.0 / max(fps, 1e-6)) / float(in_tb)))
            )
            # The output is normalised to start at 0. The source's original
            # start offset is re-applied once, at mux time, via
            # -itsoffset; applying it here too would double the shift.
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
                    # Relative timestamp: first frame -> 0, preserving the
                    # exact spacing (and therefore VFR timing) of the rest.
                    pts = source_pts - first_source_pts
                    next_pts = source_pts + fallback_step
                elif next_pts is not None:
                    source_pts = next_pts
                    pts = source_pts - (first_source_pts or 0)
                    next_pts = source_pts + fallback_step
                else:
                    # No PTS on the very first frame either; start at zero.
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
        """Copy the source audio track onto the encoded video.

        The rebuilt video starts at 0 and the copied audio keeps its own
        timestamps, so both original start offsets are re-applied with
        `-itsoffset`. Without that, a source whose video and audio do not
        begin at the same instant would be misaligned by exactly that
        difference.
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
