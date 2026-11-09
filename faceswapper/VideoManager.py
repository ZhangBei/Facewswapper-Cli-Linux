import os
import cv2
import threading
import time
import numpy as np
from skimage import transform as trans
from math import floor, ceil

from faceswapper.parameters import default_values
from faceswapper._nvtx import nvtx_range
from faceswapper.cross_frame_batch import (
    EXPERIMENTAL_CROSS_FRAME_BATCH,
    cross_frame_batch_eligibility,
    run_cross_frame_batch,
)
from faceswapper.face_batch import FaceCandidate, batch_eligibility
from faceswapper.detect_retry import (
    align_kps_for_swap, kps_quality, retry_plan,
)
from faceswapper.frame_upload import normalize_plan
from faceswapper.frontal_match import effective_threshold_pair
from faceswapper.orientation import (
    PROBE_ANGLES, choose_angle, should_probe,
    invert_affine, map_kps_back, normalized_grid_matrix, rotation_affine,
    merge_probe_landmarks, parse_probe_angles, should_probe_on_miss,
)
from faceswapper.recognize import should_skip_recognize


import onnxruntime
import torchvision
from torchvision.transforms.functional import normalize # 迁移到 v2
import torch
import torch.nn.functional
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
torch.set_grad_enabled(False)
onnxruntime.set_default_logger_severity(1)

device = 'cuda'

# 阶段 4 的 7.1 单帧多脸 batch 的闸门。代码已编写，但**默认不启用**：
# _swap_video_inner 不调用 swap_core_batched，闸门关闭时该方法直接拒绝执行。
# 解冻前必须在 GPU 环境用多人脸视频做与逐脸路径的 PSNR 比对，确认像素一致。
# 完整阻断条件见 docs/gpu-utilization-plan.md 7.1。
EXPERIMENTAL_FACE_BATCH = False


def _rgb_chw_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """把 (3, H, W) 的 RGB tensor（取值名义上在 [0, 255]）转成同一设备上
    (3, H, W) 的 float32 CIE 1976 L*a*b* tensor。sRGB 原色，D65 白点。
    与 cv2 的 RGB->LAB 等价，只差 cv2 为 uint8 输出所做的那步 8 bit 缩放
    （我们不需要——要做均值/标准差迁移，要的是原始 LAB）。
    """
    rgb = rgb.to(torch.float32).clamp_(0, 255) / 255.0
    # sRGB -> 线性光（分段 gamma）
    rgb = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    R, G, B = rgb[0], rgb[1], rgb[2]
    # 线性 RGB -> CIE XYZ（sRGB 矩阵），按 D65 白点归一化
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B            # Yn = 1
    Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
    # XYZ -> L*a*b*
    EPS = 216.0 / 24389.0                                         # (6/29)**3
    KAPPA = 24389.0 / 27.0
    def _f(t):
        return torch.where(
            t > EPS,
            t.clamp(min=1e-12).pow(1.0 / 3.0),
            (KAPPA * t + 16) / 116,
        )
    fx, fy, fz = _f(X), _f(Y), _f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=0)


def _lab_to_rgb_chw_uint8(lab: torch.Tensor) -> torch.Tensor:
    """_rgb_chw_to_lab 的逆操作。返回 clamp 到 [0, 255] 的 (3, H, W) uint8
    RGB。超出色域的 LAB 在线性 RGB 那一步之后就 clamp 掉，让 sRGB 编码始终
    在实数范围内进行。"""
    L, a, b = lab[0], lab[1], lab[2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    DELTA = 6.0 / 29.0                                            # EPS**(1/3)
    KAPPA = 24389.0 / 27.0
    def _inv_f(ft):
        return torch.where(ft > DELTA, ft.pow(3), (116 * ft - 16) / KAPPA)
    X = _inv_f(fx) * 0.95047
    Y = _inv_f(fy)
    Z = _inv_f(fz) * 1.08883
    # XYZ -> 线性 RGB（sRGB 矩阵求逆）
    R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    Bc = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    R = R.clamp_(0, 1); G = G.clamp_(0, 1); Bc = Bc.clamp_(0, 1)
    # 线性 -> sRGB
    def _to_srgb(x):
        return torch.where(
            x > 0.0031308,
            1.055 * x.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055,
            12.92 * x,
        )
    rgb = torch.stack([_to_srgb(R), _to_srgb(G), _to_srgb(Bc)], dim=0)
    return (rgb.clamp_(0, 1) * 255).to(torch.uint8)

class VideoManager():  
    def __init__(self, models):
        self.models = models
        # 模型相关
        self.swapper_model = []             # insightface swapper 模型
        # self.faceapp_model = []             # insight faceapp model
        self.input_names = []               # inswapper.onnx 各输入的名字
        self.input_size = []                # inswapper.onnx 各输入的尺寸

        self.output_names = []              # inswapper.onnx 各输出的名字    
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) #112
        # self.arcface_dst = self.arcface_dst * 128.0/112.0
        self.arcface_scale = 128.0/np.linalg.norm(self.arcface_dst[0]-self.arcface_dst[1])

        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        self.FFHQ_kps /= 4.0

        # 参数 / 控制快照。用 schema 默认值打底，让这个类开箱即可用；CLI
        # 随后会用用户保存的值覆盖它们。
        self.parameters = default_values()
        # per-worker CUDA stream——在每个线程首次调用 swap_video 时懒填充。
        # 把每帧的工作包进 `with torch.cuda.stream(worker_stream):`，就会
        # 把该 worker 的所有 torch 操作排到它自己的 stream 上，而不是全局
        # 默认那条。再配上以
        # `user_compute_stream=worker_stream.cuda_stream` 配置的 per-thread
        # ORT session，整个每帧管线（torch 操作 + ORT 调用）就在每个 worker
        # 各自的一条 stream 上运行——worker 不再在全局默认 stream 的队列上
        # 串行化，那正是 2026-05-21 的 nsys trace 揭示的瓶颈，即便在
        # Per-Thread session 已经把 inswapper 调用本身并行化之后依然存在。
        self._worker_streams_tls = threading.local()

        # 当前图片的目标人脸。每项是一个 dict，含 "Embedding"（检测到的人脸的
        # ArcFace embedding）、"AssignedEmbedding"（要换入的源身份）以及
        # 可选的 High Fidelity 记账字段。
        self.found_faces = []

        # Headless 批量模式: 设置后，一帧里检测到的每张脸都换成这个
        # embedding，不做逐脸匹配。这就是 CLI 暴露的"一个源身份、替换所有
        # 脸"策略。
        self.force_source_embedding: np.ndarray | None = None

        # Headless 目标选择: 设置后，只有 embedding 匹配这些身份之一的
        # 检测脸才会被换。None 表示"换掉每一张检测到的脸"。
        self.target_face_embeddings: list[np.ndarray] | None = None
        self.target_face_threshold: float = 55.0
        # 姿态自适应匹配阈值（PoseCompensationSlider）：全侧脸时判定线
        # 下调的百分数，按每张检测脸的关键点正脸度线性插值。0 = 关闭，
        # 判定与旧代码逐字等价。frontal_match.effective_threshold 计算。
        self.target_face_pose_compensation: float = 0.0
        # 与 target_face_embeddings 一一对应的参考图 kps（B1b）。补偿 >0
        # 时阈值按（检测脸, 参考图）这对姿态配对计算；None / 长度不符
        # 时只看检测脸。参考图姿态未知不报错——降级为 B1a 行为。
        self.target_face_ref_kps: list | None = None
        # True = 显式保留每帧逐脸 run_recognize 的旧路径（--force-recognize）。
        # 默认 False：headless video 且未指定 --target-face 时，那些 embedding
        # 没有任何读取方，跳过可省掉每帧 N 次 ArcFace 推理。
        self.force_recognize: bool = False
        # 识别精度：True = 每次识别把裁剪图水平镜像再各跑一次 ArcFace、
        # 取平均，压低侧脸 embedding 的方差。默认 False（单次，行为不变）。
        # 由 headless._SwapperBase 按 RecognizeMirrorSwitch 注入。
        self.recognize_mirror: bool = False
        # 识别精度：True = 逐脸识别用 dim=2 的网格平均（4 个裁剪各跑一次
        # ArcFace 取平均）。只影响"识别裁剪"那次调用；source 特征提取也
        # 同样受益。默认 False（dim=1，行为不变）。
        self.recognize_dim2: bool = False
        # 阶段 7：帧已是 uint8 C contiguous 时跳过 ``astype('uint8')`` 的整帧
        # 复制，直接把原数组交给 torch.from_numpy。默认 False，关掉时
        # _swap_video_inner 的表达式逐字不变。
        self.zero_copy_frames: bool = False

        # 自动朝向状态（OrientAutoSwitch）。`detected_orient_angle` 是 4 向
        # 探测缓存下来的胜出旋转（None = 下一帧重新探测）。
        # `orient_fine_angle` 是叠加其上、用于亚 90 度精修的眼轴倾斜 EMA。
        # `orient_miss_streak` 在连续数次检测失败后（换镜头 / 切场景）触发
        # 重新探测。
        self.detected_orient_angle = None
        self.orient_fine_angle = 0.0
        self.orient_miss_streak = 0
        self.orient_miss_threshold = 5

        # 逐帧无状态的整帧朝向兜底，默认开启（--no-auto-orient 关闭）。
        # 与上面的 GUI 时代状态机互不依赖：只在第一遍 0 检出时试 90/180/270，
        # 命中后整帧转正再检测，最后逆旋回来。无跨帧状态，因此
        # workers > 1 下不会竞态；只用 90 倍数，expand=True 可逆、尺寸不变。
        # 由 headless._SwapperBase 按 options.no_auto_orient 注入。
        self.auto_orient_fallback: bool = True

        # --probe-angles 的探测轴。None = 旧的整帧旋转路径（默认，逐像素
        # 行为不变）；显式传参后由 headless 注入 (45,90,...,315) 之类，
        # 走「只旋转检测副本」路径。无跨帧状态，workers > 1 下安全。
        self.probe_angles: tuple[float, ...] | None = None

        # --probe-on-target-miss：指定 --target-face 且本帧有脸却没匹配上
        # 目标人物时，补做多向探测并合并去重。默认关闭，正常帧零额外推理。
        # 由 headless._SwapperBase 按 options 注入。
        self.probe_on_target_miss: bool = False

        # --detect-retry-quality：解析后的 (min_conf, min_kps_quality)；
        # None = 关闭，旧行为逐字不变。由 headless._SwapperBase 注入。
        self.detect_retry_quality: tuple[float, float] | None = None

        # --align-for-swap：大 yaw 帧启用 68 点派生 kps 的侧度阈值。
        # None = 关闭，旧行为逐字不变。由 headless._SwapperBase 注入。
        self.align_for_swap: float | None = None

        # 每帧性能缓存。_blur_cache 在所有 worker 线程间共享
        # （transforms.GaussianBlur 一旦构造完就是无状态的；dict.get +
        # 赋值是 GIL 原子的，所以重复构建的竞争无害）。
        # _scratch 保存按线程复用的 tensor，避开每帧的 cudaMalloc 抖动
        # ——每个 worker 通过 threading.local 拿到自己的那套缓冲，这样
        # 进行中的 swap_core 调用不会互相踩踏。
        self._blur_cache = {}
        self._scratch = threading.local()

        # swapper conditioning latent 的缓存，以 source-embedding 的字节为
        # 键。latent 只取决于 s_e（对某个指定人脸而言基本跨帧不变），所以
        # 这省掉了每次 swap 调用一次的 CPU matmul + host->device 上传。
        # 每条约 2 KB；即便一个会话里有上百个不同源也可以忽略。
        self._latent_cache: dict[bytes, torch.Tensor] = {}


    def _warp_grid_sample(self, src, sample_matrix, output_size, padding_mode='zeros'):
        """用一个像素空间的 2x3 矩阵构造采样网格，把 `src` 仿射变形到
        `output_size`。

        sample_matrix 把**输出**像素坐标 (x, y) 映射到**输入**像素坐标
        (x_in, y_in)——也就是 GPU 填充每个输出像素时查阅的那个矩阵
        （"逆向 warp"）。在我们这里它直接来自
        `skimage.SimilarityTransform.inverse.params`（输入仿射的情形）或
        `tform.params`（贴回的情形）。

        为什么它胜过 v2.functional.affine + crop: torchvision 的 affine
        输出与输入同尺寸——一张 1080p 的帧被 warp 到 256×256 的人脸裁剪
        时，affine 仍然要算约 600 万像素，而我们只留角落里那一块。用
        target 尺寸网格的 grid_sample 只算我们真正保留的像素（约 6.5 万），
        这正是 2026-05-21 nsys trace 的 swap_core breakdown 里我们原本
        白付的那部分。

        align_corners=False 与 torchvision 的 affine 语义一致，所以输出与
        它所替换的路径在位级别上相近（仅差双线性舍入的最后一位小数）。"""
        src_h = src.shape[-2]
        src_w = src.shape[-1]
        out_h, out_w = int(output_size[0]), int(output_size[1])

        # 像素空间的输出坐标网格。
        y_coords = torch.arange(out_h, dtype=torch.float32, device='cuda')
        x_coords = torch.arange(out_w, dtype=torch.float32, device='cuda')
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 应用像素空间矩阵: in_xy = M[:, :2] @ (x, y) + M[:, 2]
        if isinstance(sample_matrix, torch.Tensor):
            M = sample_matrix.to('cuda').to(torch.float32)
        else:
            M = torch.from_numpy(
                np.ascontiguousarray(sample_matrix, dtype=np.float32)
            ).to('cuda')
        flat = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        in_xy = flat @ M[:, :2].T + M[:, 2]
        in_xy = in_xy.reshape(out_h, out_w, 2)

        # 用 align_corners=False 的语义归一化到 [-1, 1]
        # (pixel center 0 → -1 + 1/W, pixel center W-1 → 1 - 1/W).
        in_xy[..., 0] = ((in_xy[..., 0] + 0.5) / src_w) * 2.0 - 1.0
        in_xy[..., 1] = ((in_xy[..., 1] + 0.5) / src_h) * 2.0 - 1.0

        grid = in_xy.unsqueeze(0)  # (1, out_h, out_w, 2)

        # grid_sample 需要 float 输入；src 可能是 uint8。
        src_4d = src.unsqueeze(0) if src.ndim == 3 else src
        if src_4d.dtype != torch.float32:
            src_4d = src_4d.float()

        sampled = torch.nn.functional.grid_sample(
            src_4d, grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False,
        )
        return sampled.squeeze(0)  # (C, out_h, out_w)

    def _get_worker_stream(self):
        """返回调用线程专用的 CUDA stream，首次调用时懒创建。用于包住每帧
        换脸管线，让每个 worker 线程把自己的 torch 操作 + ORT 调用放在
        自己的 stream 上跑，而不是全局默认那条——消除了跨 worker 的队列
        串行化，那是在 Per-Thread inswapper session 把真正的 ORT 调用
        并行化之后依然存在的瓶颈。stream 存在 thread-local 存储里，随
        worker 一同消亡（TLS 会自动清理已死线程的 slot）。"""
        s = getattr(self._worker_streams_tls, 'stream', None)
        if s is None:
            s = torch.cuda.Stream()
            self._worker_streams_tls.stream = s
        return s

    def _get_gaussian_blur(self, kernel_size, sigma):
        """带记忆的 GaussianBlur module。相同的 (kernel_size, sigma) 对应同一个
        module 对象，于是 kernel tensor 对每种参数组合只构建一次，而不是每帧
        都建。"""
        key = (int(kernel_size), float(sigma))
        blur = self._blur_cache.get(key)
        if blur is None:
            blur = transforms.GaussianBlur(key[0], key[1])
            self._blur_cache[key] = blur
        return blur

    def _get_latent_for(self, s_e, *, scale=1.0, extrap_amount=0.0, latent_mode='emap'):
        """把 `s_e`（source embedding）对应的 swapper conditioning latent 作为
        CUDA tensor 返回。首次调用会真算（CPU matmul + host->device 上传）；
        之后相同 key 的调用直接命中缓存。key 里包含 scale + extrap_amount +
        latent_mode，因为这三者都会改变输出。

        latent_mode：inswapper 用 'emap'（默认），或 'raw'（跳过 emap 投影
        ——给那种直接消费原始归一化 embedding 的模型用）。
        """
        s = np.ascontiguousarray(s_e, dtype=np.float32)
        key = (s.tobytes(), round(float(scale), 4), round(float(extrap_amount), 4), latent_mode)
        cached = self._latent_cache.get(key)
        if cached is not None:
            return cached
        latent_np = self.models.calc_swapper_latent(
            s, scale=scale, extrap_amount=extrap_amount, latent_mode=latent_mode,
        )
        latent = torch.from_numpy(latent_np).float().to('cuda')
        self._latent_cache[key] = latent
        return latent

    def _get_scratch(self, name, shape, dtype=torch.float32):
        """返回一个 per-thread 可复用的 cuda tensor，形状和 dtype 按请求给。只在
        形状或 dtype 变化时才重新分配（例如 pipeline_dim 在帧与帧之间变了）。
        调用方需要已知起始值时，要自己把内容重置（.zero_() / .fill_()）。"""
        cache = getattr(self._scratch, 'tensors', None)
        if cache is None:
            cache = {}
            self._scratch.tensors = cache
        t = cache.get(name)
        shape = tuple(shape)
        if t is None or tuple(t.shape) != shape or t.dtype != dtype:
            t = torch.empty(shape, dtype=dtype, device='cuda')
            cache[name] = t
        return t

    def _rotate_detect_copy(self, img, angle):
        """生成一份旋转检测副本，返回 ``(probe_img, inv_A)``。

        ``img`` 是 (3,H,W) uint8 CUDA tensor，全程不被修改。旋转用
        ``orientation.rotation_affine`` 自构造的 A 配 affine_grid +
        grid_sample，所以 A 的逆就是 kps 的回程，不需要去猜（或在 GPU 上
        现场标定）torchvision rotate 的矩阵——那是原方案唯一没法离线
        验证的环节。副本由调用方逐个生成、检测完立刻释放，避免同时留
        多张全尺寸 CHW tensor 在显存里（workers > 1 时这份开销会乘以
        worker 数）。
        """
        src_hw = (int(img.shape[1]), int(img.shape[2]))
        A, dst_hw = rotation_affine(float(angle), src_hw)
        inv_A = invert_affine(A)
        theta = normalized_grid_matrix(inv_A, src_hw, dst_hw)
        theta_t = torch.tensor(
            [list(theta[0]), list(theta[1])],
            dtype=torch.float32, device=img.device,
        ).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(
            theta_t, [1, 3, dst_hw[0], dst_hw[1]], align_corners=False,
        )
        probe = torch.nn.functional.grid_sample(
            img.unsqueeze(0).to(torch.float32), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )
        return probe.squeeze(0).to(torch.uint8), inv_A

    def _collect_angle_candidates(self, img, angles, detect_mode,
                                  detect_score, detect_input_size):
        """逐个生成旋转检测副本，返回 ``[(angle, kpss_frame, scores)]``。

        ``kpss_frame`` 全部是原帧坐标（``map_kps_back`` 之后）；scores 供
        跨角度合并去重排序。副本逐个生成、检测完立刻释放，不把多张全尺寸
        CHW tensor 同时留在显存里（workers > 1 时这份开销会乘以 worker 数）。
        """
        candidates = []
        for angle in angles:
            probe_img, inv_A = self._rotate_detect_copy(img, angle)
            kpss_rot, scores = self.models.run_detect(
                probe_img, detect_mode, max_num=10,
                score=detect_score, input_size=detect_input_size,
                with_scores=True,
            )
            candidates.append((angle, map_kps_back(kpss_rot, inv_A), scores))
            del probe_img
        return candidates

    def _probe_on_target_miss(self, img, ret, detect_mode, detect_score,
                              detect_input_size, already_rotated):
        """目标人物本帧未匹配上时补做多向探测，只把新增脸追加到 ``ret``。

        匹配判定与下面 swap 循环里的 ``any(>=)`` 语义保持一致（含姿态
        补偿），改动其一需同步。0 度第一遍检出永远保留，补回的脸只追加
        不替换；新增脸补齐 embedding 后照常走原有的目标脸筛选。
        """
        targets = self.target_face_embeddings
        if not targets:
            return ret
        ref_kps_list = self.target_face_ref_kps
        if ref_kps_list is None or len(ref_kps_list) != len(targets):
            ref_kps_list = [None] * len(targets)

        def matched(face_kps, face_emb):
            if face_emb is None:
                return False
            return any(
                self.findCosineDistance(face_emb, target)
                >= effective_threshold_pair(
                    face_kps, kps_ref,
                    self.target_face_threshold,
                    self.target_face_pose_compensation,
                )
                for target, kps_ref in zip(targets, ref_kps_list)
            )

        matched_any = any(matched(face_kps, face_emb)
                          for face_kps, face_emb in ret)
        if not should_probe_on_miss(
            enabled=self.probe_on_target_miss,
            has_targets=bool(targets),
            detected=len(ret),
            matched_any=matched_any,
            already_rotated=already_rotated,
        ):
            return ret

        angles = self.probe_angles
        if angles is None:
            # 没给 --probe-angles 时补探测按 8 向；0 检出帧仍走旧整帧路径，
            # 两者互不干扰。
            angles = parse_probe_angles("45")
        candidates = self._collect_angle_candidates(
            img, angles, detect_mode, detect_score, detect_input_size,
        )
        added = merge_probe_landmarks(
            [face_kps for face_kps, _ in ret], candidates, max_faces=10,
        )
        if not added:
            return ret
        dim = 2 if self.recognize_dim2 else 1
        for face_kps in added:
            kps_array = np.asarray(face_kps, dtype=np.float32)
            face_emb, _ = self.models.run_recognize(img, kps_array, dim=dim)
            ret.append([kps_array, face_emb])
        return ret

    def _kps_for_swap(self, face_kps, align_kps_map):
        """--align-for-swap：查旁路映射后交给 stdlib 纯决策函数。

        ``ret`` 刻意保持 2 元素结构，68 点结果走 ``id(face_kps)`` 映射表
        旁路传递；决策本身（侧度门 + 68 点质量门）在
        ``detect_retry.align_kps_for_swap`` 里，可离线做行为测试。
        """
        if self.align_for_swap is None:
            return face_kps
        return align_kps_for_swap(
            face_kps, align_kps_map.get(id(face_kps)), self.align_for_swap,
        )

    def probe_orientation(self, img, detect_mode, detect_score, input_size=640):
        """依次试 [0, 90, 180, 270] 四个旋转角，选 detector 找到脸最多的那个朝向。
        平手时取 0（角度最小的候选），所以证据不足时我们倾向于不旋转。

        返回胜出角度（单位度）。调用方负责在真正执行检测前把图像转过这个
        角度。"""
        best_angle = 0
        best_count = -1
        for angle in (0, 90, 180, 270):
            if angle == 0:
                probe_img = img
            else:
                probe_img = v2.functional.rotate(
                    img, angle=angle,
                    interpolation=v2.InterpolationMode.BILINEAR, expand=True,
                )
            kpss = self.models.run_detect(probe_img, detect_mode, max_num=20, score=detect_score, input_size=input_size)
            n = len(kpss)
            if n > best_count:
                best_count = n
                best_angle = angle
        return best_angle


    ## Queues for the Coordinator
    # @profile
    def swap_video(self, target_image, frame_number=0):
        """把 source 身份换进一帧的 target 脸上。

        `target_image` 是 HxWx3 uint8 RGB（numpy），或者同形状的 CUDA
        tensor。返回成品帧，HxWx3 uint8 的 CUDA tensor。
        """
        with nvtx_range(f"swap_video[f={frame_number}]"):
            worker_stream = self._get_worker_stream()
            with torch.cuda.stream(worker_stream):
                result = self._swap_video_inner(
                    target_image, frame_number,
                )
            # 跨 stream 等待，不是 host 同步。任何在全局默认 stream 上读
            # `result` 的人（把数据拷回 CPU 写文件的那个调用方）从此能看到
            # 对 worker_stream 尚未完成的写入的依赖。默认 stream 只入队一个
            # 等待令牌，不会阻塞 host。
            torch.cuda.default_stream().wait_stream(worker_stream)
            return result

    def _swap_video_inner(self, target_image, frame_number):
        # 取一份 parameters 的本地拷贝，避免线程安全问题
        parameters = self.parameters.copy()

        # 既接受 CPU 上的 numpy 数组（旧路径 / PyAV-CPU 解码），也接受 CUDA
        # tensor（torchcodec 的 GPU 常驻解码）。数据本来就在 GPU 上时跳过
        # CPU 上传，这正是 GPU 常驻路径的全部意义所在。
        if isinstance(target_image, torch.Tensor):
            img = target_image
            if img.device.type != 'cuda':
                img = img.to('cuda')
            if img.dtype != torch.uint8:
                img = img.to(torch.uint8)
        else:
            # 阶段 7：闸门关闭时逐字保留原来的 astype；开启后仅当帧已是
            # uint8 C contiguous 时复用原数组，省掉一整帧 memcpy。PyAV 每帧
            # to_ndarray 都产出新数组，跳过复制只会让 torch.from_numpy 借用
            # 它，同步 .to('cuda') 完成前不会有别人改写这块内存。
            if self.zero_copy_frames:
                plan = normalize_plan(
                    target_image.dtype.name,
                    bool(target_image.flags.c_contiguous),
                    target_image.shape[0],
                    target_image.shape[1],
                    zero_copy=True,
                )
                frame = (
                    target_image if not plan.needs_copy
                    else np.ascontiguousarray(target_image, dtype=np.uint8)
                )
            else:
                frame = target_image.astype('uint8')
            img = torch.from_numpy(frame).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW
        
        # 画面小于 512 时放大
        img_x = img.size()[2]
        img_y = img.size()[1]
        
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=False)
            else:
                tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=False)

            img = tscale(img)
            
        elif img_x<512:
            tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=False)
            img = tscale(img)
        
        elif img_y<512:
            tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=False)
            img = tscale(img)    

        # 决定检测前把画面旋转多少。
        # 自动模式（OrientAutoSwitch）在第一帧（或连续检测失败之后）试探 4 个
        # 朝向，缓存胜出的角度，并叠加一个眼轴倾角的 EMA 做 90 度以内的微调。
        # 自动模式优先于手动的 OrientSwitch/Slider 组合。
        auto_orient = parameters.get('OrientAutoSwitch', False)
        detect_mode = parameters['DetectTypeTextSel']
        detect_score = parameters['DetectScoreSlider'] / 100.0
        detect_input_size = int(parameters.get('DetectInputSizeTextSel', 640))

        applied_angle = 0.0
        if auto_orient:
            if self.detected_orient_angle is None:
                self.detected_orient_angle = self.probe_orientation(img, detect_mode, detect_score, detect_input_size)
                self.orient_miss_streak = 0
                self.orient_fine_angle = 0.0
            applied_angle = float(self.detected_orient_angle) + float(self.orient_fine_angle)
        elif parameters['OrientSwitch']:
            applied_angle = float(parameters['OrientSlider'])

        if applied_angle != 0.0:
            img = v2.functional.rotate(img, angle=applied_angle, interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        # 找出画面里所有的脸，返回 5 点 kpss 列表
        # 只有打开质量门控重检时才多要一份 confidence；不传开关时
        # run_detect 的返回类型与旧版逐字一致。
        need_scores = self.detect_retry_quality is not None
        detect_result = self.models.run_detect(
            img, detect_mode, max_num=10, score=detect_score,
            input_size=detect_input_size, with_scores=need_scores,
        )
        if need_scores:
            kpss, det_scores = detect_result
        else:
            kpss, det_scores = detect_result, None

        # 整帧朝向兜底（默认开启，--no-auto-orient 关闭）：只在第一遍 0
        # 检出的帧上试 90/180/270，命中后把整帧转正重新检测。正常视频
        # 第一遍就检出，零额外推理、零旋转，输出逐像素不变。
        # GUI 时代的 OrientAutoSwitch / OrientSwitch 在检测前已经转过一次，
        # 这里让位，避免两次旋转叠加后逆旋不完整。
        if should_probe(
            detected=len(kpss),
            enabled=self.auto_orient_fallback,
            already_rotated=(
                auto_orient
                or bool(parameters.get('OrientSwitch', False))
                or applied_angle != 0.0
            ),
        ):
            if self.probe_angles is None:
                # 旧路径：整帧转正、在旋转域里换脸、末尾逆旋回来。
                # 不传 --probe-angles 时逐字保留，workers=1 下逐像素回归。
                counts = {}
                for angle in PROBE_ANGLES:
                    probe_img = v2.functional.rotate(
                        img, angle=angle,
                        interpolation=v2.InterpolationMode.BILINEAR, expand=True,
                    )
                    counts[angle] = len(self.models.run_detect(
                        probe_img, detect_mode, max_num=10,
                        score=detect_score, input_size=detect_input_size,
                    ))
                fallback_angle = choose_angle(counts)
                if fallback_angle:
                    # 只用 90 倍数：正旋 H×W → W×H，末尾的逆旋 W×H → H×W，
                    # 输出尺寸与原帧严格一致，编码器不会看到变尺寸的帧。
                    # 命中角度的图重新转一次，避免把 3 张全尺寸 CHW tensor
                    # 同时留在显存里（workers>1 时这份开销会乘以 worker 数）。
                    applied_angle = float(fallback_angle)
                    img = v2.functional.rotate(
                        img, angle=applied_angle,
                        interpolation=v2.InterpolationMode.BILINEAR, expand=True,
                    )
                    kpss = self.models.run_detect(
                        img, detect_mode, max_num=10,
                        score=detect_score, input_size=detect_input_size,
                    )
            else:
                # 新路径（--probe-angles）：img 全程不动，只转检测副本，
                # 把 5 点 kps 逆映射回原帧坐标后直接交给 swap_core。
                # 输出尺寸天然等于输入尺寸，也不需要末尾的逆旋。
                candidates = self._collect_angle_candidates(
                    img, self.probe_angles, detect_mode, detect_score,
                    detect_input_size,
                )
                counts = {
                    angle: len(kpss_frame)
                    for angle, kpss_frame, _scores in candidates
                }
                fallback_angle = choose_angle(counts, self.probe_angles)
                if fallback_angle:
                    # applied_angle 保持 0：末尾三个逆旋分支都不触发，
                    # paste_back 的 clamp 直接落在原帧画布上。
                    chosen = next(
                        kpss_frame
                        for angle, kpss_frame, _ in candidates
                        if angle == fallback_angle
                    )
                    kpss = np.asarray(
                        chosen, dtype=np.float32,
                    ).reshape(-1, 5, 2)

        # 漏检重试（DetectRetrySwitch）：0 检出帧升一档输入尺寸、抬高分
        # 数阈值再检一次（detect_retry.retry_plan 的两道防误检闸）。只在
        # 本来不会换脸的帧上追加推理；auto-orient 开启时不重试（它自己的
        # 4 向重探已覆盖横竖颠倒类漏检，双开会在同一帧跑两套试探）。
        quality_limits = self.detect_retry_quality
        # 整帧已被旋转（GUI OrientSwitch / 旧整帧兜底命中）时 kps 不在原帧
        # 坐标域，第一遍的 scores 也已过期，不做质量门控重检。
        quality_ok = quality_limits is not None and applied_angle == 0.0
        max_conf = None
        if (quality_ok and det_scores is not None and len(kpss) > 0
                and len(det_scores) == len(kpss)):
            max_conf = float(max(det_scores))
        frame_quality = (
            min((kps_quality(face) for face in kpss), default=None)
            if quality_ok and len(kpss) > 0 else None
        )
        if len(kpss) == 0 or quality_ok:
            retry = retry_plan(
                detected=len(kpss),
                input_size=detect_input_size,
                detect_score=detect_score,
                auto_orient=auto_orient,
                enabled=bool(parameters.get('DetectRetrySwitch')),
                quality_enabled=quality_ok,
                max_conf=max_conf,
                frame_kps_quality=frame_quality,
                min_conf=quality_limits[0] if quality_limits else 0.0,
                min_kps_quality=quality_limits[1] if quality_limits else 0.0,
            )
            if retry is not None:
                retry_size, retry_score = retry
                retry_result = self.models.run_detect(
                    img, detect_mode, max_num=10,
                    score=retry_score, input_size=retry_size,
                    with_scores=quality_ok,
                )
                if quality_ok:
                    retry_kpss, retry_scores = retry_result
                else:
                    retry_kpss, retry_scores = retry_result, None
                if len(kpss) == 0:
                    # 0 检出帧：沿用旧行为，直接用重检结果。
                    kpss = retry_kpss
                else:
                    # detected>0 的质量重检：只追加、不替换。直接覆盖会
                    # 为了救一张侧脸把已检出的正脸丢掉（S5 §3.1 同类坑）。
                    added = merge_probe_landmarks(
                        kpss, [(0.0, retry_kpss, retry_scores)], max_faces=10,
                    )
                    if added:
                        kpss = np.concatenate(
                            [np.asarray(kpss, dtype=np.float32),
                             np.asarray(added, dtype=np.float32)],
                            axis=0,
                        )

        # 自动朝向反馈：统计检测成功次数用于重新试探，并按第一张检测到的脸的
        # 眼轴更新微调角度的 EMA。
        if auto_orient:
            if len(kpss) == 0:
                self.orient_miss_streak += 1
                if self.orient_miss_streak >= self.orient_miss_threshold:
                    self.detected_orient_angle = None
                    self.orient_fine_angle = 0.0
            else:
                self.orient_miss_streak = 0
                try:
                    e_l = kpss[0][0]
                    e_r = kpss[0][1]
                    tilt = float(np.degrees(np.arctan2(e_r[1] - e_l[1], e_r[0] - e_l[0])))
                    # 挡掉虚假的大倾角（姿势古怪的脸）；90 度这种量级的朝向
                    # 本该由粗试探处理掉。
                    if abs(tilt) < 30.0:
                        self.orient_fine_angle = 0.9 * self.orient_fine_angle + 0.1 * tilt
                except (IndexError, TypeError, ValueError):
                    pass
        # 给画面里所有检测到的脸算 embedding。
        # headless video 且未指定 --target-face 时，下面的分支只把同一个
        # force_source_embedding 贴到所有检测到的脸上，face_emb 没有任何
        # 读取方——跳过每帧 N 次 run_recognize 的 ORT 调用，像素输出不变。
        # 图片 / GUI 的 found_faces 匹配路径、以及 --target-face 的相似度
        # 筛选都要真的读 embedding，照常计算。判定抽到 recognize.py 以便
        # 脱离 GPU 栈做行为测试。
        skip_recognize = should_skip_recognize(
            force_source=self.force_source_embedding,
            targets=self.target_face_embeddings,
            opt_out=self.force_recognize,
        )
        ret = []
        # --align-for-swap 的旁路映射：刻意不把 ret 扩成 3 元素，因为
        # 后面多处 ``for face_kps, face_emb in ret`` 解包依赖 2 元素结构。
        # key 用同一个 numpy 对象的 id（ret 持有引用，不会被回收后复用）。
        align_kps_map = {}
        want_align_kps = self.align_for_swap is not None and not skip_recognize
        for face_kps in kpss:
            face_emb = None
            if not skip_recognize:
                rec_result = self.models.run_recognize(
                    img, face_kps, dim=2 if self.recognize_dim2 else 1,
                    with_align_kps=want_align_kps,
                )
                if want_align_kps:
                    face_emb, _, align_kps = rec_result
                    align_kps_map[id(face_kps)] = align_kps
                else:
                    face_emb, _ = rec_result
            ret.append([face_kps, face_emb])

        # --probe-on-target-miss：指定 --target-face 且本帧已有脸却没匹配上
        # 目标人物时，补做多向探测并合并去重，只把新增脸追加到 ret。
        # 默认关闭时这一层只是一次布尔判断，正常帧逐像素行为不变。
        if self.probe_on_target_miss and ret:
            ret = self._probe_on_target_miss(
                img, ret, detect_mode, detect_score, detect_input_size,
                already_rotated=(
                    auto_orient
                    or bool(parameters.get('OrientSwitch', False))
                    or applied_angle != 0.0
                ),
            )
        
        if ret and self.force_source_embedding is not None:
            # Headless 策略：一个 source 身份，作用到调用方选中的脸上。设置了
            # target_face_embeddings 时，只有匹配那些身份的脸会被换掉，其他人
            # 一概不动。
            targets = self.target_face_embeddings
            if EXPERIMENTAL_FACE_BATCH and not targets:
                # 阶段 4 的 7.1 接入点：闸门关闭时逐字走下面的原逐脸循环。
                # 打开后 swap_core_batched 自己按 batch_eligibility 判定并
                # 在不适用时回退逐脸；--target-face 的相似度筛选不走 batch。
                img, _ = self.swap_core_batched(
                    img, [face_kps for face_kps, _ in ret],
                    self.force_source_embedding, parameters,
                )
            else:
                # 姿态补偿（PoseCompensationSlider）：按（检测脸, 参考图）
                # 这对姿态算各自的判定线——侧面参考图匹配正脸帧时相似度
                # 同样系统性偏低。ref_kps 缺失时退化为只看检测脸（B1a）；
                # 补偿 0 时全部恒等于 target_face_threshold，与旧判定
                # 逐字等价。any(>=) 与 best-match 语义一致：只要有参考图
                # 过线就换，取哪个参考图不影响"换不换"。
                ref_kps_list = self.target_face_ref_kps
                if ref_kps_list is None or len(ref_kps_list) != len(targets):
                    ref_kps_list = [None] * len(targets)
                for face_kps, face_emb in ret:
                    if targets and not any(
                        self.findCosineDistance(face_emb, target)
                        >= effective_threshold_pair(
                            face_kps, kps_ref,
                            self.target_face_threshold,
                            self.target_face_pose_compensation,
                        )
                        for target, kps_ref in zip(targets, ref_kps_list)
                    ):
                        continue
                    with nvtx_range("swap_core"):
                        img = self.swap_core(
                            img, self._kps_for_swap(face_kps, align_kps_map),
                            self.force_source_embedding,
                            parameters,
                        )

            img = img.permute(1,2,0)
            if applied_angle != 0.0:
                img = img.permute(2,0,1)
                img = transforms.functional.rotate(img, angle=-applied_angle, expand=True)
                img = img.permute(1,2,0)

        elif ret:
            # 对画面里每一张检测到的脸，在 self.found_faces 里找出最匹配的
            # 条目（若它已分配 source）并换脸。各次检测互不相干：found_faces
            # 里排在前头、这一帧没匹配上的 slot，不能挡住后排匹配上的 slot。
            # 取最佳匹配（超过阈值中相似度最高者）保证一次检测只换一次脸，即使
            # 有多个 found_face 的 embedding 都越过了阈值。
            threshold = float(parameters["ThresholdSlider"])
            for fface in ret:
                best_sim = -float('inf')
                best_slot = None
                for found_face in self.found_faces:
                    if not found_face.get("SourceFaceAssignments"):
                        continue  # 只有 target 的 slot，还没分配 source
                    emb = found_face.get("Embedding")
                    if emb is None:
                        continue
                    try:
                        sim = self.findCosineDistance(fface[1], emb)
                    except Exception:
                        # 某个 slot 的 embedding 格式不对不该让整帧中止——
                        # 跳过它，继续匹配其余。
                        continue
                    if sim >= threshold and sim > best_sim:
                        best_sim = sim
                        best_slot = found_face
                if best_slot is not None:
                    s_e = best_slot["AssignedEmbedding"]
                    with nvtx_range("swap_core"):
                        img = self.swap_core(
                            img, self._kps_for_swap(fface[0], align_kps_map),
                            s_e, parameters, slot=best_slot,
                        )

            img = img.permute(1,2,0)
            if applied_angle != 0.0:
                img = img.permute(2,0,1)
                img = transforms.functional.rotate(img, angle=-applied_angle, expand=True)
                img = img.permute(1,2,0)

        else:
            img = img.permute(1,2,0)
            if applied_angle != 0.0:
                img = img.permute(2,0,1)
                img = v2.functional.rotate(img, angle=-applied_angle, interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                img = img.permute(1,2,0)
        
        # 把小画面缩回原尺寸
        if img_x <512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=False)
            img = img.permute(2,0,1)
            img = tscale(img)
            img = img.permute(1,2,0)
            

        # 结果以 HxWx3 uint8 留在 CUDA 上，这样 Qt 预览的 CUDA-GL 桥能直接
        # 上传，不必 GPU->CPU->GPU 往返一趟。需要 numpy 的消费方（录像文件写
        # 入、Save Image）自己拷——把这一趟躲不掉的往返挪出每帧显示的热路径。
        return img.to(torch.uint8).contiguous()

    def findCosineDistance(self, vector1, vector2):
        cos_dist = 1.0 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0

        return 100.0-cos_dist*50.0



    # @profile
    def swap_core(self, img, kps, s_e, parameters, precomputed_latent=None,
                  slot=None, batch_slot=None):  # img = RGB
        """换一张脸。``batch_slot`` 是阶段 4 的 7.1 实验钩子，默认 None——
        现有逐脸路径只多一次 bool 判断，行为逐字不变。"""
        # "inswapper-512" 选中 inswapper_512 模型：单次前向吃下完整 512x512
        # 脸部裁剪，不做 polyphase 分解。
        # "inswapper-256" 用专门的单次 pass inswapper_256_phase1；
        # "hyperswap-256"/"hyperswap-1b-256"/"hyperswap-1c-256" 跑原生 256
        # 生成器（三个变体只有权重不同）。其他任何值——包括来自 marker 的
        # parameter snapshot 里的陈旧值——都退回到 128 基线，而不是抛异常。
        swapper_type = str(parameters['SwapperTypeTextSel'])
        use_native_512 = (swapper_type == 'inswapper-512')
        use_native_256 = (swapper_type == 'inswapper-256')
        use_hyperswap = swapper_type.startswith('hyperswap')
        if use_native_512:
            swap_size = 512
            dim = 1
        elif use_native_256 or use_hyperswap:
            swap_size = 256
            dim = 1
        else:
            swap_size = 128
            dim = 1
        # inswapper 各变体（128 / 256 原生 / 512 原生）都吃 emap 投影后的
        # latent；HyperSwap 吃的是 raw L2-normalized embedding。
        latent_mode = 'raw' if use_hyperswap else 'emap'

        # 计算 Pipeline size
        # 找出最大的 arcface  pairwise 距离
        pdist = np.square(kps - kps[:, None])
        pdist2 = (pdist[:,:,0] + pdist[:,:,1])

        lmk1 = int(pdist2.argmax()/4)
        lmk2 = pdist2.argmax() % 4

        scale = np.linalg.norm(kps[lmk1] - kps[lmk2])/np.linalg.norm(self.arcface_dst[lmk1] - self.arcface_dst[lmk2])
        pipeline_size = int(scale*128)

        # 下限 128（swapper 的最小输入），上限 swap_size。上限才是承重墙：当
        # 一张脸占满源画面的很大一块时，scale*128 可能涨到 800-1500。swapper
        # 本身最多也只看到 swap_size（128/256/512），所以超出 swap_size 的部分
        # 只是对模型实际细节带宽的双线性/双三次拉伸——更大的 pipeline buffer 每帧
        # 要在输入仿射、swap 上采样（约 1083 行）、mask 缩放（约 1193 行）和
        # restorer 裁剪（约 1740 行）上付出 O(pipeline_size^2) 的代价，却带不来
        # 任何真实保真度。设了上限之后，躲不掉的那次上采样被推到反向仿射贴回
        # （约 1221 行）里的一次双线性插值。
        if pipeline_size < 128:
            pipeline_size = 128
        elif pipeline_size > swap_size:
            pipeline_size = swap_size
        pipeline_dim = pipeline_size/128.0

        # 变换
        dst = self.arcface_dst * pipeline_dim
        dst[:,0] += 8.0*pipeline_dim

        # 修改参考点
        if parameters['FaceAdjSwitch']:
            dst[:,0] += parameters['KPSXSlider']
            dst[:,1] += parameters['KPSYSlider']
            dst[:,0] -= (swap_size-1)
            dst[:,0] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,0] += (swap_size+1)
            dst[:,1] -= (swap_size-1)
            dst[:,1] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,1] += (swap_size+1)

        tform = trans.SimilarityTransform.from_estimate(kps, dst)

        # 一次 grid_sample 调用取到 pipeline_size 大小的对齐脸部裁剪，而不是
        # 先 warp 整帧再裁剪。旧的 v2.functional.affine 路径会算遍输入画面的
        # 每一个像素（1080p 下约 600 万），随后 v2.functional.crop 又把角落
        # pipeline_size x pipeline_size 之外的部分全丢掉。用 pipeline_size 尺寸
        # 网格的 grid_sample 只采样我们真正留下的像素。tform.inverse.params 是
        # 把输出（pipeline 坐标）映射到输入的像素空间矩阵
        # (img coords), which is exactly what _warp_grid_sample needs.
        with nvtx_range("sc_input_warp"):
            pipeline_input_face = self._warp_grid_sample(
                img,
                np.asarray(tform.inverse.params)[:2, :],
                (pipeline_size, pipeline_size),
            )
            # 转回 uint8，对齐下游消费者（apply_occlusion 等）期望的 dtype；
            # grid_sample 内部按 float32 计算，所以返回的是 float32。
            pipeline_input_face = pipeline_input_face.to(torch.uint8)

        # Likeness / Distinctiveness 微调——在原始 inswapper 输入之上做
        # latent 整形。两者的中性值分别是 1.0 / 0.0。
        latent_scale = float(parameters.get('LikenessSlider', 100)) / 100.0
        extrap_amount = float(parameters.get('EmbExtrapSlider', 0)) / 100.0
        # High Fidelity Cached 模式：如果这个 slot 已经有测好的身份差距，
        # 在这里把它折进 s_e，让第一道（也是唯一一道）pass 就用上修正后的
        # embedding。没有缓存差距的启动帧会落到下面的 2-pass 代码块，由它从
        # 本帧的换脸输出测出差距并存下来。
        hf_switch_on = bool(parameters.get('HighFidelitySwitch'))
        hf_mode = str(parameters.get('HighFidelityModeTextSel', 'Cached'))
        hf_alpha = float(parameters.get('HighFidelityAlphaSlider', 50)) / 100.0
        cached_gap = slot.get('HFCorrectionGap') if slot is not None else None
        # HFRefinePending 即使存在缓存差距，也强制本次换脸重新测一次；下面
        # 的 2-pass 代码块随后按滑动平均把新测得的样本并进缓存。合并之后该
        # 标志被清除，下一帧又回到单 pass 的缓存路径。
        hf_refine_pending = bool(slot.get('HFRefinePending')) if slot is not None else False
        hf_cached_hit = (
            hf_switch_on and hf_mode == 'Cached'
            and cached_gap is not None and hf_alpha > 0.0
            and not hf_refine_pending
        )
        with nvtx_range("sc_latent"):
            s_e_for_latent = s_e
            if hf_cached_hit:
                s_e_arr = np.asarray(s_e, dtype=np.float32).reshape(-1)
                s_e_for_latent = s_e_arr + hf_alpha * cached_gap
            if precomputed_latent is not None:
                latent = precomputed_latent
            else:
                latent = self._get_latent_for(
                    s_e_for_latent, scale=latent_scale, extrap_amount=extrap_amount,
                    latent_mode=latent_mode,
                )
        # 只绑定，不克隆。下面两个分支（FaceAdjSwitch 仿射和无条件的
        # Resize）都返回新 tensor，所以 pipeline_input_face 原对象永远不会
        # 被就地修改。
        swap_face_input = pipeline_input_face

        # 可选缩放 # 修改变换矩阵
        if parameters['FaceAdjSwitch']:
            swap_face_input = v2.functional.affine(swap_face_input, 0, (0, 0), 1 + parameters['FaceScaleSlider'] / 100, 0, center=(pipeline_dim*128-1, pipeline_dim*128-1), interpolation=v2.InterpolationMode.BILINEAR)

        itex = 1
        if parameters['StrengthSwitch']:
            itex = ceil(parameters['StrengthSlider'] / 100.)

        # output_size = int(128 * dim)  -- 用不上了；swap_face_output 在被读
        # 取之前总是先被下面的 poly_pass(...) 重新赋值。
        # 用 functional 形式——每次调用都构造 v2.Resize() 会分配一个变换
        # 对象，在逐脸循环里累积起来不是小数目。v2.functional.resize 是这个
        # 类封装的底层调用，参数语义和输出都一样。
        swap_face_input = v2.functional.resize(
            swap_face_input,
            [swap_size, swap_size],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=False,
        )
        swap_face_input = swap_face_input.permute(1, 2, 0)
        swap_face_input = torch.div(swap_face_input, 255.0)

        # 每次强度迭代分发一个 polyphase pass。具体实现由
        # _polyphase_pass_v1（生产路径）、_native_512_pass（单次的
        # inswapper_512，经 "inswapper-512" 选中）、_native_256_pass（单次的
        # inswapper_256_phase1，经 "inswapper-256" 选中）或 _hyperswap_pass
        # （单次的 hyperswap_1a/1b/1c_256，经各 "hyperswap-*-256" 选中）提供。
        # "inswapper-128"（默认值）和任何不认识的旧值都落到
        # _polyphase_pass_v1。
        if use_native_512:
            poly_pass = self._native_512_pass
        elif use_native_256:
            poly_pass = self._native_256_pass
        elif use_hyperswap:
            poly_pass = self._hyperswap_pass
        else:
            poly_pass = self._polyphase_pass_v1

        # ===== 实验性钩子：阶段 4 的 7.1 单帧多脸 batch ==================
        # 只有 swap_core_batched 会传 batch_slot；闸门未打开时无人调用。
        # prepare 阶段在写回 img 之前把对齐裁剪填进 batch buffer 就返回；
        # finish 阶段把 poly_pass 换成只读批输出的闭包，让下面 *255/clamp
        # 与 735 行之后的全部 postprocess 原样复用，不复制第二套实现。
        if batch_slot is not None:
            index = batch_slot["index"]
            if batch_slot["mode"] == "prepare":
                # (H, W, 3) 0-1 -> (3, H, W)；此时 img 尚未被任何脸写回。
                batch_slot["input"][index].copy_(
                    swap_face_input.permute(2, 0, 1)
                )
                batch_slot["bbox"] = self._face_bbox(
                    img.shape, tform, pipeline_size,
                )
                batch_slot["pipeline_size"] = pipeline_size
                return None
            # finish：itex 必须为 1、HF 2-pass 必须关闭，两者由
            # face_batch.batch_eligibility 保证。
            def poly_pass(_inp, _latent, _dim, _size, _params):
                return batch_slot["output"][index].permute(1, 2, 0).clone()

        for k in range(itex):
            with nvtx_range(f"sc_polypass[k={k}]"):
                swap_face_output = poly_pass(swap_face_input, latent, dim, swap_size, parameters,)
                # 按引用把当前输入记到 prev_face——不克隆。poly_pass 写的是
                # 一个 per-thread 缓存 buffer；下游没有任何东西会就地修改
                # swap_face_input，所以光有引用就够了。（旧代码防御性地克隆过，
                # 多余。）
                prev_face = swap_face_input
                # 只有要作为下一次迭代的输入回喂时才克隆输出；否则下一次
                # poly_pass 调用会 .zero_() 掉缓存 buffer，把我们正要拿去换的
                # 数据擦掉。最后一次迭代没有下一道 pass，所以跳过克隆。
                if k < itex - 1:
                    swap_face_input = swap_face_output.clone()
                # mul/clamp 都是 out-of-place：swap_face_output 重新绑定到新
                # tensor，缓存 buffer 原样留着复用。
                swap_face_output = torch.mul(swap_face_output, 255)
                swap_face_output = torch.clamp(swap_face_output, 0, 255)

        # ====== High Fidelity 2-pass 精化 ==================================
        # 用 ArcFace 测量换脸输出的身份，算出一个把结果推回 s_e 的修正
        # source embedding，然后把换脸重跑一遍。
        #
        # 两种模式：
        #   Per-frame：每帧都测 + 重跑（每张脸的代价翻倍）。
        #   Cached：    每个 slot 只测一次，把差距存到 slot 上，后续帧直接
        #              跳过这个代码块——修正已经折进上面第一道 pass 的
        #              latent 里了。只有启动帧是 2-pass。
        # 命中 Cached 的情况在这里就被挡掉，不必付第二次换脸的代价。
        if hf_switch_on and not hf_cached_hit and hf_alpha > 0.0:
          with nvtx_range("sc_hf_refine"):
            # swap_face_output 是 HWC、0-255 的 float；arcface 要的是 CHW uint8。
            chip = swap_face_output.permute(2, 0, 1).to(torch.uint8)
            try:
                output_emb = self.models.embed_face_chip(chip)
            except Exception as e:
                output_emb = None
                print(f'[swap_core] HighFidelity arcface failed: {e}')
            if output_emb is not None:
                # 在与原 s_e 相同的空间里做修正（这里不做归一化——
                # calc_swapper_latent 会处理）。alpha=0 表示不修正；
                # alpha=1 表示完整的一步修正。
                s_e_arr = np.asarray(s_e, dtype=np.float32).reshape(-1)
                out_arr = np.asarray(output_emb, dtype=np.float32).reshape(-1)
                gap = s_e_arr - out_arr
                # Cached 模式：把差距存到 slot 上，让后续帧走上文的单 pass
                # 缓存路径。差距已存在时（refine 路径），按滑动平均把新样本并
                # 进缓存——不同帧上多次 refine 点击会让差距收敛到 source 身份
                # 偏差的真实均值，从而压低当初逼出这套机制的逐帧测量噪声。
                if hf_mode == 'Cached' and slot is not None:
                    prev = slot.get('HFCorrectionGap')
                    n_prev = int(slot.get('HFCorrectionGapSamples', 0) or 0)
                    if prev is None or n_prev <= 0:
                        merged_gap = gap
                        merged_n = 1
                    else:
                        merged_gap = (n_prev * prev + gap) / (n_prev + 1)
                        merged_gap = merged_gap.astype(np.float32, copy=False)
                        merged_n = n_prev + 1
                    slot['HFCorrectionGap'] = merged_gap
                    slot['HFCorrectionGapSamples'] = merged_n
                    slot['HFRefinePending'] = False
                    gap = merged_gap
                s_e_corrected = s_e_arr + hf_alpha * gap
                latent2_np = self.models.calc_swapper_latent(
                    s_e_corrected,
                    scale=latent_scale,
                    extrap_amount=extrap_amount,
                    latent_mode=latent_mode,
                )
                latent2 = torch.from_numpy(latent2_np).float().to('cuda')
                # 用修正后的 latent 把同一套迭代流水线重跑一遍。不要重跑
                # StrengthSwitch 的迭代——那些累积的是强度，不是身份。
                swap_face_output = poly_pass(
                    swap_face_input, latent2, dim, swap_size, parameters,
                )
                swap_face_output = torch.mul(swap_face_output, 255)
                swap_face_output = torch.clamp(swap_face_output, 0, 255)

        swap = swap_face_output.permute(2, 0, 1)

        # 每帧换脸输出的调试转储。pipeline 里的 tensor 是 RGB；cv2 要 BGR，
        # 所以写入前翻转通道顺序。
        # cv2.imwrite('out.jpg', cv2.cvtColor(swap.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))

        if parameters['StrengthSwitch']:
            if itex == 0:
                swap = pipeline_input_face.clone()
            else:
                alpha = np.mod(parameters['StrengthSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # 混合 the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)

                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)
        # 在 float32 下 resize，不用 antialias 低通，然后再量化。此前转成
        # uint8 + antialias=True 会把换脸结果柔化（float 先变成取整整数再插值，
        # 且降采样路径里还内置了一个类似高斯的低通滤波）。
        with nvtx_range("sc_resize_to_pipeline"):
            swap = v2.Resize(
                (pipeline_size, pipeline_size),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=False,
            )(swap.type(torch.float32))
            swap = swap.clamp(0, 255).type(torch.uint8)

        # 在 GPU 上做 LAB 色彩迁移：把 swap 各通道的 LAB 均值/标准差搬到与
        # 换脸前 target 裁剪一致。没有 device->host 往返，不用 cv2——还是原来
        # 那个 Reinhard 变换，但这个开关不再引发一次性的 cv2/torch 预热卡顿。
        if parameters['ColorMatchSwitch']:
          with nvtx_range("sc_color_match"):
            swap_lab = _rgb_chw_to_lab(swap)
            tgt_lab = _rgb_chw_to_lab(pipeline_input_face)
            s_mean = swap_lab.mean(dim=(1, 2), keepdim=True)
            s_std = swap_lab.std(dim=(1, 2), keepdim=True) + 1e-6
            t_mean = tgt_lab.mean(dim=(1, 2), keepdim=True)
            t_std = tgt_lab.std(dim=(1, 2), keepdim=True)
            swap = _lab_to_rgb_chw_uint8(
                (swap_lab - s_mean) / s_std * t_std + t_mean
            )

        # 应用色彩校正
        if parameters['ColorSwitch']:
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastSlider'])
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaSlider'], 1.0)
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationSlider'])

            swap = swap.permute(1, 2, 0).type(torch.float32)
            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=device)
            swap += del_color
            swap = torch.clamp(swap, min=0, max=255)
            swap = swap.permute(2, 0, 1)

        with nvtx_range("sc_mask_compose"):
            # 建 border mask。用 per-thread 可复用的 (1, 128, 128) buffer；每帧
            # 重置为全 1，再把四条边带打成 0。
            border_mask = self._get_scratch('border_mask', (1, 128, 128)).fill_(1.0)

            top = parameters['BorderTopSlider']
            left = parameters['BorderSidesSlider']
            right = 128-parameters['BorderSidesSlider']
            bottom = 128-parameters['BorderBottomSlider']

            border_mask[:, :top, :] = 0
            border_mask[:, bottom:, :] = 0
            border_mask[:, :, :left] = 0
            border_mask[:, :, right:] = 0

            border_mask = self._get_gaussian_blur( parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2,)(border_mask)

            # 建 image mask。用 per-thread 可复用的 (1, 128, 128) buffer，每帧
            # 重置为全 1；下游的乘法（Occluder/FaceParser/Diff）都重新绑定到新
            # tensor，所以缓存的那块存储只被当作起始值使用。
            swap_mask = self._get_scratch('swap_mask', (1, 128, 128)).fill_(1.0)

            # 脸部差异处理
            if parameters["DiffSwitch"]:
                mask = self.apply_fake_diff(swap, pipeline_input_face, parameters["DiffSlider"])
                mask = self._get_gaussian_blur( parameters['BlendSlider']*2+1, (parameters['BlendSlider']+1)*0.2, )(mask.type(torch.float32))
                swap = swap*mask + pipeline_input_face*(1-mask)

            # restorer
            if parameters["RestorerSwitch"]:
                swap = self.apply_restorer(swap, parameters)

            # occluder
            if parameters["OccluderSwitch"]:
                mask = self.apply_occlusion(pipeline_input_face, parameters["OccluderSlider"])
                mask = v2.Resize((128, 128))(mask)
                swap_mask = torch.mul(swap_mask, mask)

            # DeepFaceLab XSeg（用户自备的遮挡模型）
            if parameters["DFLXSegSwitch"]:
                mask = self.apply_dfl_xseg(pipeline_input_face, parameters["DFLXSegSizeSlider"])
                mask = v2.Resize((128, 128))(mask)
                if parameters["DFLXSegBlurSlider"] > 0:
                    mask = self._get_gaussian_blur(
                        parameters['DFLXSegBlurSlider']*2+1,
                        (parameters['DFLXSegBlurSlider']+1)*0.2,
                    )(mask)
                swap_mask = torch.mul(swap_mask, mask)


            if parameters["FaceParserSwitch"]:
                mask = self.apply_face_parser(swap, parameters["FaceParserSlider"], parameters['MouthParserSlider'])
                mask = v2.Resize((128, 128))(mask)
                swap_mask = torch.mul(swap_mask, mask)

            # 给 swap_mask 结果加模糊
            swap_mask = self._get_gaussian_blur(
                parameters['BlendSlider']*2+1,
                (parameters['BlendSlider']+1)*0.2,
            )(swap_mask)


            # 合并 border mask 与 swap mask，缩放后作用到 swap 上
            swap_mask = torch.mul(swap_mask, border_mask)
            # swap_mask = t512(swap_mask)
            swap_mask = v2.Resize((pipeline_size, pipeline_size))(swap_mask)
            swap = torch.mul(swap, swap_mask)

        with nvtx_range("paste_back"):
            # 计算要合并回原始画面的区域
            tform = trans.SimilarityTransform.from_estimate(kps, dst)
            IM512 = tform.inverse.params[0:2, :]
            corners = np.array([[0,0], [0,pipeline_size-1], [pipeline_size-1, 0], [pipeline_size-1, pipeline_size-1]])

            x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
            y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
          
            left = floor(np.min(x))
            if left<0:
                left=0
            top = floor(np.min(y))
            if top<0: 
                top=0
            right = ceil(np.max(x))
            if right>img.shape[2]:
                right=img.shape[2]            
            bottom = ceil(np.max(y))
            if bottom>img.shape[1]:
                bottom=img.shape[1]   

            # 一次 grid_sample 调用，把 swap + mask 反向 warp 回 img 画布，
            # 输出尺寸只按目标 bbox 给。旧路径先把 swap 补齐到整帧、warp 整帧、
            # 再裁到 bbox——仿射输出的大部分白算了。grid_sample 直接在 bbox 尺寸
            # 的输出上采样。把 swap（3 通道）和 mask（1 通道）堆起来，多花一个
            # 通道，让一次 warp 同时处理两者。
            # tform.params 是像素空间里 img → swap 的矩阵，正好就是 grid_sample
            # 需要的"每个输出点该从输入哪里取"的矩阵。把 (left, top) 偏移烘进
            # 矩阵的平移列，helper 就能用普通的输出像素坐标 [0, bbox_w/h)。
            stacked = torch.cat((swap, swap_mask), dim=0)
            bbox_h = bottom - top
            bbox_w = right - left
            if bbox_h > 0 and bbox_w > 0:
                T = np.asarray(tform.params)[:2, :].astype(np.float32, copy=True)
                T[:, 2] = T[:, :2] @ np.array([left, top], dtype=np.float32) + T[:, 2]
                sampled = self._warp_grid_sample(
                    stacked, T, (bbox_h, bbox_w),
                )
                swap = sampled[0:3].permute(1, 2, 0)
                swap_mask = sampled[3:4].permute(1, 2, 0)
            else:
                # 退化的 bbox（clamp 之后脸整个移出画布）。跳过 warp；下游的
                # 合成本来也是空操作，因为拿空 tensor 做 torch.add 会报错。把
                # swap/swap_mask 设为空，让下面那句就地 img 赋值也变成空操作。
                swap = torch.empty(
                    (0, 0, 3), dtype=torch.float32, device='cuda',
                )
                swap_mask = torch.empty(
                    (0, 0, 1), dtype=torch.float32, device='cuda',
                )
            swap_mask = torch.sub(1, swap_mask)

            # 把 mask 作用到原图的对应区域
            img_crop = img[0:3, top:bottom, left:right]
            img_crop = img_crop.permute(1,2,0)            
            img_crop = torch.mul(swap_mask,img_crop)
          
            # 把裁出的区域加回去，放回原图
            swap = torch.add(swap, img_crop)
            swap = swap.type(torch.uint8)
            swap = swap.permute(2,0,1)
            img[0:3, top:bottom, left:right] = swap  

        return img

    # ===== 阶段 4 的 7.1：单帧多脸 batch（实验性，未启用） ===============

    @staticmethod
    def _face_bbox(img_shape, tform, pipeline_size):
        """算 paste_back 会写回的矩形 ``(left, top, right, bottom)``。

        与 ``swap_core`` 的 paste_back 段是同一套算法。实验性 batch 的
        prepare 阶段必须在写回之前就知道 bbox，用来判定人脸是否重叠。
        **改动 paste_back 的 bbox 计算时必须同步这里。**
        """
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([
            [0, 0], [0, pipeline_size - 1],
            [pipeline_size - 1, 0], [pipeline_size - 1, pipeline_size - 1],
        ])
        x = (IM512[0][0] * corners[:, 0]
             + IM512[0][1] * corners[:, 1] + IM512[0][2])
        y = (IM512[1][0] * corners[:, 0]
             + IM512[1][1] * corners[:, 1] + IM512[1][2])
        left = max(0, floor(np.min(x)))
        top = max(0, floor(np.min(y)))
        right = min(img_shape[2], ceil(np.max(x)))
        bottom = min(img_shape[1], ceil(np.max(y)))
        return left, top, right, bottom

    def swap_core_batched(self, img, faces, s_e, parameters):
        """把一帧里的 N 张脸合成一次 swapper 前向，返回 ``(img, status)``。

        **当前不启用**：``EXPERIMENTAL_FACE_BATCH=False``，且
        ``_swap_video_inner`` 不会调用本方法。解冻前必须在 GPU 环境用多人脸
        视频做与逐脸路径的 PSNR 比对；六条阻断条件见
        ``docs/gpu-utilization-plan.md`` 7.1。当前
        ``inswapper_128.fp16.onnx`` 为 ``static-batch=1``，判定会直接回退。

        思路：第一遍 ``prepare`` 只填 batch 输入、收集 bbox，完全不写回 img；
        中间一次 ``run_swapper_batched``；第二遍 ``finish`` 按原顺序把各脸的
        输出喂回 ``swap_core`` 的 postprocess。这样 mask / restorer /
        paste_back 全部复用原有 190 行，不必复制第二套实现。

        bbox 两两不重叠是硬条件：finish 阶段重算 ``pipeline_input_face`` 时
        读到的 img 已被前面的脸写回，重叠时与逐脸路径不再等价。
        """
        if not EXPERIMENTAL_FACE_BATCH:
            raise RuntimeError(
                "single-frame multi-face batch is experimental and disabled; "
                "see docs/gpu-utilization-plan.md section 7.1"
            )

        n = len(faces)
        swapper_type = str(parameters['SwapperTypeTextSel'])
        latent_scale = float(parameters.get('LikenessSlider', 100)) / 100.0
        extrap_amount = float(parameters.get('EmbExtrapSlider', 0)) / 100.0
        latent = self._get_latent_for(
            s_e, scale=latent_scale, extrap_amount=extrap_amount,
            latent_mode='emap',
        )

        # 128 inswapper 的固定输入形状。判据会拒绝 pipeline_size != 128 的脸，
        # 所以这里的 copy_ 形状一定吻合。
        batch_in = torch.empty(
            (n, 3, 128, 128), dtype=torch.float32, device='cuda',
        )
        batch_out = torch.empty_like(batch_in)
        slots = []
        for index, kps in enumerate(faces):
            slot = {
                "mode": "prepare", "index": index,
                "input": batch_in, "output": batch_out,
            }
            self.swap_core(
                img, kps, s_e, parameters,
                precomputed_latent=latent, batch_slot=slot,
            )
            slots.append(slot)

        ok, reason = batch_eligibility(
            [FaceCandidate(s["bbox"], s["pipeline_size"]) for s in slots],
            swapper_type=swapper_type,
            parameters=parameters,
            batch_backend_available=self.models.swapper_supports_dynamic_batch(),
            target_face_filtering=bool(self.target_face_embeddings),
        )
        if not ok:
            # prepare 阶段没有写回过 img，这里可以干净地退回逐脸路径。
            for kps in faces:
                img = self.swap_core(img, kps, s_e, parameters)
            return img, f"fallback: {reason}"

        # source 在所有脸之间共享，run_swapper_batched 只绑定 latent 的第 0 行。
        self.models.run_swapper_batched(batch_in, latent, batch_out)

        for index, kps in enumerate(faces):
            slot = slots[index]
            slot["mode"] = "finish"
            img = self.swap_core(
                img, kps, s_e, parameters,
                precomputed_latent=latent, batch_slot=slot,
            )
        return img, "batched"

    def cross_frame_batch_swap(
        self,
        frame_items,
        s_e,
        parameters,
        *,
        candidates,
        workers: int,
        every_nth: int,
        batch_backend_available: bool,
    ):
        """实验性 7.2：跨帧 batch 前向 + 逐帧回填，返回 ``(imgs, result)``。

        **当前不启用**：闸门 ``EXPERIMENTAL_CROSS_FRAME_BATCH=False``，且
        ``_swap_video_inner`` 不调用本方法。闸门关闭时 ``run_cross_frame_batch``
        返回 ``gated``，这里补一次逐脸 ``fallback``，因此即便将来误接进主路径，
        输出也与基线一致。

        ``frame_items`` 是按解码顺序排列的 ``[(img_chw, kps), ...]``；
        ``candidates`` 是对应的 ``FrameCandidate`` 元数据。prepare / finish
        复用 7.1 的 ``swap_core(batch_slot=...)`` 钩子，所以各帧的 mask、
        restorer、paste_back 仍走原有实现；帧与帧之间没有 bbox 重叠问题。
        """
        n = len(frame_items)
        # 闸门关闭时这些 buffer 会被分配但不使用；主路径不调用本方法，
        # 解冻后才会真正走到这里。
        batch_in = torch.empty(
            (n, 3, 128, 128), dtype=torch.float32, device='cuda',
        )
        batch_out = torch.empty_like(batch_in)
        latent_scale = float(parameters.get('LikenessSlider', 100)) / 100.0
        extrap_amount = float(parameters.get('EmbExtrapSlider', 0)) / 100.0
        latent = self._get_latent_for(
            s_e, scale=latent_scale, extrap_amount=extrap_amount,
            latent_mode='emap',
        )
        slots = [
            {
                "mode": "prepare", "index": i,
                "input": batch_in, "output": batch_out,
            }
            for i in range(n)
        ]
        out_imgs = [item[0] for item in frame_items]

        def prepare(index, item):
            img, kps = item
            self.swap_core(
                img, kps, s_e, parameters,
                precomputed_latent=latent, batch_slot=slots[index],
            )

        def infer(count):
            self.models.run_swapper_batched(batch_in, latent, batch_out)

        def finish(index, item):
            img, kps = item
            slots[index]["mode"] = "finish"
            out_imgs[index] = self.swap_core(
                img, kps, s_e, parameters,
                precomputed_latent=latent, batch_slot=slots[index],
            )

        def fallback():
            for index, (img, kps) in enumerate(frame_items):
                out_imgs[index] = self.swap_core(img, kps, s_e, parameters)

        eligible = cross_frame_batch_eligibility(
            candidates,
            workers=workers,
            every_nth=every_nth,
            swapper_type=str(parameters['SwapperTypeTextSel']),
            parameters=parameters,
            batch_backend_available=batch_backend_available,
        )
        result = run_cross_frame_batch(
            gate_open=EXPERIMENTAL_CROSS_FRAME_BATCH,
            eligible=eligible,
            prepare=prepare,
            infer=infer,
            finish=finish,
            fallback=fallback,
            frame_items=frame_items,
        )
        if result.status == "gated":
            # 编排层在闸门关闭时不碰任何帧；由这里补逐脸路径，保证输出等价。
            fallback()
        return out_imgs, result

    # ------------------------------------------------------------------
    # Polyphase passes
    #
    # 每个函数实现一轮"抽稀 → 逐 phase 换脸 → 重组"的循环。输入是对齐后的脸，
    # (swap_size, swap_size, 3) 布局，cuda 上 [0, 1] 的 float32。输出是同一
    # 布局/范围内重组后的图像；调用方负责 *255 / clamp / 强度混合。
    # ------------------------------------------------------------------

    def _native_512_pass(self, swap_face_input, latent, dim, swap_size, parameters):
        """用原生 512x512 的 inswapper 直接做单块换脸。不做 polyphase 分解：
        把完整的 (512, 512, 3) 脸部裁剪喂给模型，原样取它的
        (512, 512, 3) 输出。这条路径上 dim 始终为 1。"""
        del dim, parameters  # 与 polyphase pass 保持签名一致；这里用不到。

        # Per-thread scratch——和 v1 用同一个 key，因为输出形状/用途完全一致，
        # 两条路径可以共用这个 buffer。
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        ).zero_()

        # inswapper_512 模型期望 (1, 3, 512, 512)。swap_face_input 是 swap_size
        # 下的 (H, W, 3)；permute + unsqueeze 成连续的
        # scratch slot。embedding scratch 与 run_swapper_512 的 `source`
        # 绑定的 1x512 约定对齐。
        batch_in = self._get_scratch(
            f'native512_in_{swap_size}', (1, 3, swap_size, swap_size),
        )
        batch_out = self._get_scratch(
            f'native512_out_{swap_size}', (1, 3, swap_size, swap_size),
        )
        emb_in = self._get_scratch('native512_emb', (1, 512))

        batch_in[0].copy_(swap_face_input.permute(2, 0, 1))
        # latent 是 cuda 上的 (1, 512)，来自 calc_swapper_latent / _get_latent_for。
        emb_in.copy_(latent)

        self.models.run_swapper_512(batch_in, emb_in, batch_out)

        # (1, 3, H, W) -> (H, W, 3)，写进调用方期望的输出 buffer。
        swap_face_output[:] = batch_out[0].permute(1, 2, 0)
        return swap_face_output

    def _native_256_pass(self, swap_face_input, latent, dim, swap_size, parameters):
        """用原生 inswapper_256_phase1 模型直接做单块换脸。不做 polyphase 分解：
        喂入完整的 (256, 256, 3) 脸部裁剪，原样取它的 (256, 256, 3) 输出。
        dim 始终为 1。"""
        del dim, parameters  # 与 polyphase pass 保持签名一致；这里用不到。

        # Per-thread scratch——与 v1 共用 poly_out_{swap_size} 这个 key，
        # 因为输出形状/用途完全一致。
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        ).zero_()

        # inswapper_256_phase1 期望 (1, 3, 256, 256)；swap_face_input 是
        # swap_size 下的 (H, W, 3)。embedding scratch 与 run_swapper_256 的
        # 1x512 `source` 绑定对齐。
        batch_in = self._get_scratch(
            f'native256_in_{swap_size}', (1, 3, swap_size, swap_size),
        )
        batch_out = self._get_scratch(
            f'native256_out_{swap_size}', (1, 3, swap_size, swap_size),
        )
        emb_in = self._get_scratch('native256_emb', (1, 512))

        batch_in[0].copy_(swap_face_input.permute(2, 0, 1))
        # latent 是 cuda 上的 (1, 512)，来自 calc_swapper_latent / _get_latent_for。
        emb_in.copy_(latent)

        self.models.run_swapper_256(batch_in, emb_in, batch_out)

        # (1, 3, H, W) -> (H, W, 3) into the output buffer the caller expects.
        swap_face_output[:] = batch_out[0].permute(1, 2, 0)
        return swap_face_output

    # SwapperTypeTextSel 值 → Models 上对应的 run 方法。三个 hyperswap
    # 变体共享 _hyperswap_pass，pass 依此挑选权重；未知值（陈旧 snapshot
    # 等）回到 1a。
    _HYPERSWAP_RUNS = {
        'hyperswap-256': 'run_hyperswap',
        'hyperswap-1b-256': 'run_hyperswap_1b',
        'hyperswap-1c-256': 'run_hyperswap_1c',
    }

    def _hyperswap_pass(self, swap_face_input, latent, dim, swap_size, parameters):
        """用 HyperSwap 生成器做单块换脸（dim 始终为 1）。

        与其他 pass 契约相同——输入输出都是 0-1 区间的 (H, W, 3)——因此
        调用方的强度循环和每一个下游消费者都不用改动。HyperSwap 本身工作在
        [-1, 1]（均值 0.5 / 标准差 0.5），所以归一化放在这里做，而不是放在
        调用处；这样也保证第二次强度迭代仍然正确，因为它会把 0-1 的输出回喂
        进来。
        """
        del dim  # 与 polyphase pass 保持签名一致；这里用不到。
        run_hyperswap = getattr(self.models, self._HYPERSWAP_RUNS.get(
            str(parameters['SwapperTypeTextSel']), 'run_hyperswap',
        ))

        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        )

        batch_in = self._get_scratch(
            f'hyperswap_in_{swap_size}', (1, 3, swap_size, swap_size),
        )
        batch_out = self._get_scratch(
            f'hyperswap_out_{swap_size}', (1, 3, swap_size, swap_size),
        )
        emb_in = self._get_scratch('hyperswap_emb', (1, 512))

        # (H, W, 3) 0-1 -> (1, 3, H, W) in [-1, 1].
        normalized = swap_face_input.permute(2, 0, 1).unsqueeze(0)
        normalized = (normalized - 0.5) / 0.5
        batch_in.copy_(normalized)
        emb_in.copy_(latent)

        run_hyperswap(batch_in, emb_in, batch_out)

        # (1, 3, H, W) in [-1, 1] -> (H, W, 3) in 0-1.
        swap_face_output[:] = (
            batch_out[0].permute(1, 2, 0).mul(0.5).add(0.5)
        )
        return swap_face_output

    # @profile
    def _polyphase_pass_v1(self, swap_face_input, latent, dim, swap_size, parameters):
        """生产用 polyphase pass。对 dim×dim 里的每个 phase 偏移 (j, i)，输入按
        stride 抽稀后送进 swapper 跑 128×128，再交错拼回。"""
        # Per-thread 可复用 scratch，以 swap_size 为 key，让多个 pipeline_dim
        # 值共存而不必重新分配。不做 .zero_()——下面的向量化交错会无条件写遍
        # 每一个输出元素，预先清零是白费的防御性动作。
        swap_face_output = self._get_scratch(
            f'poly_out_{swap_size}', (swap_size, swap_size, 3),
        )

        # 批量 polyphase：把全部 dim^2 个按 stride 抽稀的 phase 叠成一个
        # (N, 3, 128, 128) tensor，一次 ONNX 调用跑完，而不是调 N 次。每帧省下
        # N-1 次 Python/ONNX session 往返；dim=4（N=16）时收益最大。
        # Per-thread scratch buffer 以 dim 为 key。
        n_phases = dim * dim
        batch_in = self._get_scratch(f'batch_in_d{dim}', (n_phases, 3, 128, 128))
        batch_out = self._get_scratch(f'batch_out_d{dim}', (n_phases, 3, 128, 128))

        # 向量化 stride 抽稀。等价于：
        #   for j, i in product(range(dim), range(dim)):
        #     batch_in[j*dim+i].copy_(swap_face_input[j::dim, i::dim].permute(2,0,1))
        # 但只需一次 strided→contiguous 拷贝，而不是 dim*dim 个 kernel。
        # view 成 (128, dim, 128, dim, 3) 把每条 H/W 轴拆成（内=128，外=dim），
        # 于是 view5d[:, j, :, i, :] 正好是 phase tile [j::dim, i::dim]。
        # permute (1, 3, 4, 0, 2) 重排成 (j, i, c, row, col)，与把 batch_in
        # 看成 (dim, dim, 3, 128, 128) 对齐。swap_face_input.contiguous() 不可
        # 省略：上游的 permute+div 可能让 stride 变成非规范形式，而 .view() 拒绝
        # 非连续输入；本来就连续的 tensor 会短路成 self。
        with nvtx_range(f"poly_assemble[d={dim}]"):
            swap_face_input = swap_face_input.contiguous()
            batch_in.view(dim, dim, 3, 128, 128).copy_(
                swap_face_input.view(128, dim, 128, dim, 3).permute(1, 3, 4, 0, 2)
            )

        with nvtx_range(f"poly_swap[d={dim}]"):
            # source embedding 在所有 polyphase phase 之间是共享的——inswapper
            # 的 source 绑定始终是 (1, 512)，模型内部会广播。latent 原样传即可，
            # 不必展开成 (n_phases, 512)；当年那次 expand-and-copy 只是绑定期
            # 的填充，模型从来没读过。
            self.models.run_swapper_batched(batch_in, latent, batch_out)

        with nvtx_range(f"poly_interleave[d={dim}]"):
            # 向量化交错回填：组装 permute 的逆操作。把 batch_out 看成
            # (dim, dim, 3, 128, 128)，permute (3, 0, 4, 1, 2) 后落在
            # (row, j, col, i, c)，正好就是 swap_face_output 的
            # view(128, dim, 128, dim, 3)。一次拷贝代替 dim*dim 次逐 phase 写入。
            swap_face_output.view(128, dim, 128, dim, 3).copy_(
                batch_out.view(dim, dim, 3, 128, 128).permute(3, 0, 4, 1, 2)
            )

        return swap_face_output

    # @profile
    def apply_occlusion(self, img, amount):
        with nvtx_range("apply_occlusion"):
            return self._apply_occlusion_inner(img, amount)

    def _apply_occlusion_inner(self, img, amount):
        img = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)(img)
        # img = torch.mul(img, 255.0)
        # img = img.permute(2, 0, 1)

        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)

        outpred = torch.ones((256,256), dtype=torch.float32, device=device).contiguous()
        
        self.models.run_occluder(img, outpred)        
                
        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:                   
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            
        if amount <0:      
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred


    def apply_dfl_xseg(self, img, amount):
        with nvtx_range("apply_dfl_xseg"):
            return self._apply_dfl_xseg_inner(img, amount)

    def _apply_dfl_xseg_inner(self, img, amount):
        # Models.run_dfl_xseg 负责 resize / BGR / layout，返回 [0, 1] 区间内
        # 的 (res, res) 软 mask，其中 1 = 脸部区域（保留换脸结果）。
        outpred = self.models.run_dfl_xseg(img)
        res = outpred.shape[-1]

        # 把 XSeg 软 mask 阈值化成二值区域，好让下面的膨胀/腐蚀有干净的边可推
        # （与 Occluder 一致）。
        outpred = (outpred > 0.5)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount > 0:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, res, res))
        return outpred


    # # @profile
    def apply_face_parser(self, img, FaceAmount, MouthAmount):
        with nvtx_range("apply_face_parser"):
            return self._apply_face_parser_inner(img, FaceAmount, MouthAmount)

    def _apply_face_parser_inner(self, img, FaceAmount, MouthAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        img = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)(img)
        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512)).contiguous()
        outpred = torch.empty((1,19,512,512), dtype=torch.float32, device='cuda').contiguous()

        self.models.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # 嘴部解析
        if MouthAmount <0:
            mouth_idxs = torch.tensor([11], device='cuda')
            iters = int(-MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1, 1, 512, 512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel, padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        elif MouthAmount >0:
            mouth_idxs = torch.tensor([11,12,13], device='cuda')
            iters = int(MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1,1,512,512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel, padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        else:
            mouth_parse = torch.ones((1, 512, 512), dtype=torch.float32, device='cuda')

        # 背景解析
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=device)
        bg_parse = torch.isin(outpred, bg_idxs)
        bg_parse = torch.clamp(~bg_parse, 0, 1).type(torch.float32)
        bg_parse = torch.reshape(bg_parse, (1, 512, 512))



        if FaceAmount > 0:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)

        elif FaceAmount < 0:
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)
            bg_parse = torch.reshape(bg_parse, (1, 512, 512))
        else:
            bg_parse = torch.ones((1,512,512), dtype=torch.float32, device='cuda')


        # cv2.imwrite('test.jpg', bg_parse.permute(1, 2, 0).cpu().numpy()*255)

        out_parse = torch.mul(bg_parse, mouth_parse)

        return out_parse
        
    

    def apply_restorer(self, swapped_face_upscaled, parameters):
        with nvtx_range("apply_restorer"):
            return self._apply_restorer_inner(swapped_face_upscaled, parameters)

    def _apply_restorer_inner(self, swapped_face_upscaled, parameters):
        face_size = swapped_face_upscaled.shape[1]
        face_dim = face_size / 128.0

        input_face = swapped_face_upscaled.clone()

        # 如果使用单独的检测模式
        transformed = False
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            transformed = True
            if parameters['RestorerDetTypeTextSel'] == 'Blend':
                dst_init = self.arcface_dst

            elif parameters['RestorerDetTypeTextSel'] == 'Reference':
                try:
                    dst_init = self.models.resnet50(input_face, score=parameters['DetectScoreSlider']/100.0)/4.0
                except:
                    return

            dst = dst_init * face_dim
            dst[:, 0] += 8.0 * face_dim

            f_kps = self.FFHQ_kps * face_dim
            f_kps[:, 0] += 8.0 * face_dim

            tform = trans.SimilarityTransform.from_estimate(dst, f_kps)

            # 变换、缩放并归一化
            input_face = v2.functional.affine(input_face, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            input_face = v2.functional.crop(input_face, 0,0, face_size, face_size)

        input_face = torch.div(input_face, 255)
        input_face = v2.functional.normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        input_face = torch.unsqueeze(input_face, 0).contiguous()

        # 跑选中的 restorer。这里不把模型输出 resize 回 face_size。旧代码先付
        # 一次 O(face_size^2) 的 Resize(out_size -> face_size)，再付一次
        # O(face_size^2) 的就地反向仿射。新路径把两步合并成下面一次
        # grid_sample，直接从模型原生输出 buffer（256 或 512）采样到 face_size
        # 尺寸的输出。为了跟踪，每个分支都记下 `out_size`。
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            input_face = v2.Resize((256, 256), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_256(input_face, outpred)
            out_size = 256

        elif parameters['RestorerTypeTextSel'] == 'GFPGAN':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_GFPGAN(input_face, outpred)
            out_size = 512

        elif parameters['RestorerTypeTextSel'] == 'CF':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_codeformer(input_face, outpred)
            out_size = 512

        elif parameters['RestorerTypeTextSel'] == 'GPEN512':
            input_face = v2.Resize((512, 512), antialias=False)(input_face)
            outpred = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_512(input_face, outpred)
            out_size = 512


        # 还原成 cxHxW @ 255——此时 outpred 还停在 out_size 尺寸。
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if transformed:
            # 一次 grid_sample 同时完成反向仿射和从 out_size 到 face_size 的
            # 缩小。原先的两步路径是：
            #   Resize(out_size -> face_size)，在 face_size_aligned 上做
            #   v2.functional.affine(tform.inverse)，在 face_size_unaligned 上做
            # v2.functional.affine 把参数当作 FORWARD 变换，并为反向采样取逆，
            # 所以传 tform.inverse 进去，等价于在
            # tform.forward(output_pixel) 处对输入采样。再复合上先前那次
            # Resize 引入的均匀 (out_size/face_size) 缩放，就得到我们的
            # grid_sample helper 消费的反向映射：face_size_unaligned 的输出 ->
            # 经 scale * tform.forward 落到 out_size_aligned 的输入上。
            scale_io = out_size / float(face_size)
            T = np.eye(3, dtype=np.float32)
            T[:2, :2] = tform.params[:2, :2] * scale_io
            T[:2, 2] = tform.params[:2, 2] * scale_io
            outpred = self._affine_resize_grid_sample(
                outpred.unsqueeze(0), T,
                in_h=out_size, in_w=out_size,
                out_h=face_size, out_w=face_size,
            ).squeeze(0)
        elif out_size != face_size:
            # 没有请求对齐 warp——但仍需落回 face_size 的 pipeline buffer。
            # 一次 Resize，不仿射。
            outpred = v2.Resize((face_size, face_size), antialias=False)(outpred)

        # 混合
        alpha = float(parameters["RestorerSlider"])/100.0
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    def _affine_resize_grid_sample(self, src, tform_pix, in_h, in_w, out_h, out_w):
        """One-shot affine warp + size change via affine_grid + grid_sample.

        `tform_pix` 是一个 3x3 像素空间矩阵，把输出像素坐标映射到输入像素
        坐标（仿射 warper 消费的标准反向采样映射）。`src` 是 (N, C, in_h,
        in_w)，结果是 (N, C, out_h, out_w)。这里用 align_corners=True，与本文件
        其他网格计算所用的像素中心约定一致（约 1201 行的角点 landmark 计算用
        的是闭区间端点，同一套约定）。

        apply_restorer 用它把 [Resize(out_size -> face_size)] 和
        [InverseAffine(face_size)] 融合成 out_h x out_w 尺寸上的一次 pass。
        """
        n, c, _, _ = src.shape
        dev = src.device
        # 把像素空间的 tform 转成归一化空间的 theta：
        #   theta_norm = M_in_inv @ T_pix @ M_out
        # 其中 M(W,H) 把 norm[-1,1] 映射到 pixel[0, W-1]。
        M_out = torch.tensor([
            [(out_w - 1) * 0.5, 0.0, (out_w - 1) * 0.5],
            [0.0, (out_h - 1) * 0.5, (out_h - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ], device=dev, dtype=torch.float32)
        M_in_inv = torch.tensor([
            [2.0 / (in_w - 1), 0.0, -1.0],
            [0.0, 2.0 / (in_h - 1), -1.0],
            [0.0, 0.0, 1.0],
        ], device=dev, dtype=torch.float32)
        T = torch.as_tensor(np.asarray(tform_pix, dtype=np.float32), device=dev)
        theta = (M_in_inv @ T @ M_out)[:2, :].unsqueeze(0).expand(n, -1, -1)
        grid = torch.nn.functional.affine_grid(
            theta, [n, c, out_h, out_w], align_corners=True,
        )
        return torch.nn.functional.grid_sample(
            src, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
        )

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)
        
        # 逐通道找出换脸结果与原图的差异
        fthresh = DiffAmount*2.55
        
        # 二值化
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1 
        
        # 只要有一个通道越过阈值，就把它并进 mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1
        
        diff = diff.permute(2,0,1)

        return diff    
    

        # cv2.imwrite('ab.jpg', mask.permute(1, 2, 0).cpu().numpy())
