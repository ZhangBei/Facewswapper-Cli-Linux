
import os
import cv2
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import onnxruntime
import onnx
from itertools import product as product
import subprocess as sp
import threading

from faceswapper._nvtx import nvtx_range
from faceswapper.detect_buffers import OutputBufferCache, resolve_reuse_buffers
onnxruntime.set_default_logger_severity(4)


DEFAULT_MODELS_FOLDER = './models'


class Models():
    # 这些属性名被（重新）赋值即代表 VRAM 发生了变化。__setattr__ 在写入
    # 其中任何一个时把 vram_dirty 置位，Qt coordinator 于是可以在下一个
    # tick 发出 vram_updated，无需在约 15 处懒加载点逐个加埋点。
    _VRAM_TRACKED_ATTRS = frozenset([
        'retinaface_model', 'scrdf_model', 'resnet50_model',
        'recognition_model', 'swapper_model',
        'swapper_512_model', 'swapper_256_model', 'hyperswap_model',
        'GFPGAN_model', 'GPEN_256_model', 'GPEN_512_model', 'codeformer_model',
        'occluder_model', 'faceparser_model', 'dfl_xseg_model',
    ])

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in self._VRAM_TRACKED_ATTRS:
            object.__setattr__(self, 'vram_dirty', True)

    def __init__(self):
        # models folder —— 所有 *.onnx / *.engine / *.pth 都相对它解析。
        # 通过 set_models_folder() 覆盖（由 Settings tab 的 "Models
        # Folder" 选择器驱动）。
        self.models_folder = DEFAULT_MODELS_FOLDER

        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.providers = [('CUDAExecutionProvider')]
        
        self.retinaface_model = []
        self.scrdf_model = []
        self.resnet50_model, self.anchors  = [], []

        self.recognition_model = []
        self.swapper_model = []
        # inswapper_512: 与之并列的 512x512 进 / 512x512 出变体。首次
        # run_swapper_512 调用时懒加载；source conditioning 格式与 128
        # 相同，所以 calc_swapper_latent 的 `latent` 对它同样适用。
        self.swapper_512_model = []
        # inswapper_256_phase1: 原生 256x256 单次 pass 变体。与 128/512
        # 一样吃 emap 投影后的 latent；只在选中 "256-Native" 模式后，首次
        # run_swapper_256 调用时懒加载。
        self.swapper_256_model = []
        # hyperswap_1a_256: 原生 256x256 单次 pass 的 HyperSwap 生成器。
        # 吃的是 raw L2-normalized 512-d embedding（不经 emap），且要求
        # 裁剪图按 (x/255 - 0.5) / 0.5 归一化到 [-1, 1]。首次
        # run_hyperswap 调用时懒加载。
        self.hyperswap_model = []

        self.emap = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        
        self.occluder_model = []
        self.faceparser_model = []
        # DeepFaceLab XSeg 遮挡 mask（用户自备 ONNX）。市面上的导出变体在
        # tensor layout（TF 原版是 NHWC，某些转换器给出 NCHW）、输入/输出
        # 节点名、分辨率、FP16/FP32 I/O 上都可能不同。
        # _load_dfl_xseg_model 会对图做一次 introspection，把这份契约存进
        # 下面几个 attr，这样 run_dfl_xseg 无论遇到哪种导出都能正确绑定。
        self.dfl_xseg_model = []
        self._dfl_xseg_in_name = None
        self._dfl_xseg_out_name = None
        self._dfl_xseg_res = 256
        self._dfl_xseg_nhwc = True
        self._dfl_xseg_in_dtype = np.float32
        self._dfl_xseg_out_dtype = np.float32

        self.swapper_model_kps = []
        self.swapper_model_swap = []
        # TRT vs ONNX 状态。每个 "_uses_trt" 标志由对应的懒加载器
        # (_load_*_model) 置位，由对应的调用点读取。delete_models /
        # delete_models 会把它重置为 False，这样下次调用会重新探测
        # engine 是否存在。
        self._retinaface_uses_trt = False
        self._recognition_uses_trt = False
        # retinaface 后处理用的 anchor center 只依赖 (height, width,
        # stride)，而它们全部由 input_size 推导——input_size 只在
        # DetectInputSize 滑杆移动时变化。因此把它缓存到 self 上，
        # meshgrid + reshape + stack 的开销按 (input_size, stride)
        # 组合只付一次，而不是每个 worker 每帧都付。tensor 常驻 GPU；
        # 缓存未命中时构建设备并同步一次，让其他 worker 的 stream 看到
        # 写入。dict 变更受 GIL 保护——首次并发计算只会产出相同结果，
        # 某一次写入胜出。
        self._retinaface_center_cache: dict = {}
        # 每个 worker 一个的 retinaface (1, 3, H, W) float32 输入缓冲，
        # 以 input_size 为键。预分配省掉了每帧的 torch.zeros 调用，并让
        # (1, 3, H, W) layout 一路保持下去，预处理路径不再需要最后补一次
        # permute+contiguous 往返。按 worker 分开是因为各 worker 会并发
        # 写入各自的副本。
        self._retinaface_buffers_tls = threading.local()
        # 阶段 6：9 个 io_binding 输出缓冲是否按 worker 复用。默认 False，
        # 关掉时 _detect_retinaface_inner 逐字走原来的每帧 torch.empty 路径。
        # 判定与 shape 计算在 stdlib-only 的 detect_buffers.py。
        self._reuse_detect_buffers: bool = False
        # 主管线 session 加载消息里 [N/M] 前缀的进度计数。计数按 phase
        # 划分：模型加载来自几个互不重叠的用户驱动阶段（find =
        # retinaface + arcface，swap = inswapper 变体），phase 一变计数
        # 就归零。M =（本 phase 的模型数）× worker 数（Per-Thread）或
        # × 1（Shared）。计数溢出无妨——例如同一会话里同时加载
        # inswapper_128 和 _512 会显示 [2/1]。
        # 按属性区分的后端偏好。键是同时有 ONNX 与 TRT 两条路径的
        # Models 属性名（'swapper_model'、'swapper_512_model'、
        # 'retinaface_model'、'recognition_model'）。值: 'trt'（强制
        # TRT）、'onnx'（强制 ONNX），缺省 = auto（当前行为——有
        # engine 就优先 TRT）。由 Settings tab 的 Backend 开关驱动，
        # 在下次懒加载时生效。
        self._backend_pref: dict = {}
        # CUDA EP 的 VRAM arena 扩展策略。默认 kSameAsRequested；改成
        # kNextPowerOfTwo 只是减少重复 cudaMalloc、让"显存占用"数字变大，
        # 不是提速手段（见 docs/gpu-utilization-plan.md 6.2）。必须在任何
        # session 构建前设置，运行时切换对已加载的 session 无效。
        self._arena_strategy: str = 'kSameAsRequested'
        self.syncvec = torch.empty((1,1), dtype=torch.float32, device='cuda:0')

        # model session mode——控制主管线用到的每一个 ORT-TRT-EP session
        # （inswapper、inswapper_512、retinaface、recognition）。
        # 'Shared' 是每个模型在所有 worker 线程间共用一份 session
        # （VRAM 省，但 worker 会在该 session 的内部状态上串行化）；
        # 'Per-Thread' 给每个 worker 每种模型各一份 session，可以并行
        # 跑，代价是每个 worker 每个模型多占约 150-300MB VRAM。磁盘上的
        # TRT engine 缓存在各 session 间共享，engine 构建只付一次。
        self._model_session_mode: str = 'Shared'
        self._sessions_tls = threading.local()
        # name → 我们交出的每个 Per-Thread session 的强引用列表。
        # 切换模式 / 卸载的路径要丢掉这些引用，垂死 worker 的 session
        # 才能被正常 GC（只靠 TLS 不行，因为 Python 的 thread-local
        # 伸不进其他线程的 slot）。
        self._sessions_lock = threading.Lock()
        self._all_sessions: dict[str, list] = {}

        # 会话级的源人脸平均 embedding，由 main_window 在识别源脸时
        # 写入。供 Distinctiveness 滑杆使用（calc_swapper_latent）——
        # None 表示"还没有均值"，此时 extrap 分支直接空转。
        self._mean_emb = None

    def _mp(self, name):
        """把模型文件名解析到当前配置的 folder 下。"""
        return os.path.join(self.models_folder, name)

    def set_models_folder(self, path):
        """让所有模型查找指向新 folder。传假值 = 恢复默认。会丢掉所有
        已加载模型，让下次推理调用从新路径重新加载；同时把
        TensorRTEngine 里 inswapper engine 的常量重新指向，让 TRT 也用上
        新位置。"""
        new_folder = path if path else DEFAULT_MODELS_FOLDER
        if new_folder == self.models_folder:
            return
        self.models_folder = new_folder
        try:
            from faceswapper import TensorRTEngine as TRT
            TRT.set_models_folder(new_folder)
        except Exception:
            # TRT 是可选的——更新失败就静默跳过。
            pass
        # 强制重载，让下次调用拿到新路径。
        self.delete_models()

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, input_size=640):
        """跑一次人脸检测。`input_size` 是模型输入的正方形边长
        （32 的倍数——320 / 416 / 480 / 640）。Retinaface 与 SCRDF
        都接受动态输入 shape，所以这个值对它们有效。"""
        kpss = []

        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self._load_retinaface_model()

            kpss = self.detect_retinaface(img, max_num=max_num, score=score, input_size=input_size)

        elif detect_mode=='SCRDF':
            self._ensure_scrdf_model()

            kpss = self.detect_scrdf(img, max_num=max_num, score=score, input_size=input_size)

        return kpss
        
    # Models 属性名 → _get_model_session 用的短 TLS/强引用键。
    # Per-Thread 模式从不写共享属性，所以 is_model_loaded 必须改去查
    # 强引用列表。
    _ATTR_TO_TLS_NAME = {
        'swapper_model':       'swapper',
        'swapper_512_model':   'swapper_512',
        'swapper_256_model':   'swapper_256',
        'hyperswap_model':     'hyperswap',
        'retinaface_model':    'retinaface',
        'recognition_model':   'recognition',
    }

    def is_model_loaded(self, attr_name):
        """attr_name 指向的 Models 属性当前是否持有活的 session/wrapper。
        空 list / None / False 都算未加载。Per-Thread 模式下共享属性
        永远不会被写入——session 住在 `_sessions_tls` 和
        `_all_sessions[name]` 里——所以还要查强引用列表。Settings tab
        的清单用它。"""
        val = getattr(self, attr_name, None)
        if val is not None and val is not False:
            if not (isinstance(val, (list, tuple)) and len(val) == 0):
                return True
        # Per-Thread 兜底: 哪怕共享属性还是 []，某个 worker 可能已经有
        # session 了。
        tls_name = self._ATTR_TO_TLS_NAME.get(attr_name)
        if tls_name is not None:
            all_sessions = getattr(self, '_all_sessions', None)
            if all_sessions:
                lock = getattr(self, '_sessions_lock', None)
                if lock is not None:
                    with lock:
                        sessions = all_sessions.get(tls_name)
                else:
                    sessions = all_sessions.get(tls_name)
                if sessions:
                    return True
        return False

    # 懒加载器会遵守 _backend_pref 的属性。其他模型只有 ONNX 一条路径，
    # 这个开关对它们没有意义。
    _BACKEND_PREF_ATTRS = frozenset([
        'swapper_model', 'swapper_512_model', 'swapper_256_model',
        'hyperswap_model',
        'retinaface_model', 'recognition_model',
    ])

    def set_backend_preference(self, attr_name, backend, *, unload=True):
        """设置单个模型的后端偏好。

            backend == 'trt':  强制走 TRT 路径（只有运行时 engine 加载
                               失败才会回退到 ONNX）。
            backend == 'onnx': 即使磁盘上有 engine 也强制走 ONNX 路径。
            backend in (None, 'auto'): 清除覆盖——下次加载时自动探测
                               （有 engine 就优先 TRT）。

        unload=True（默认）且该模型当前已加载时，顺便把它卸掉，这样下次
        推理调用会按新偏好重新加载。未知属性名直接忽略。"""
        if attr_name not in self._BACKEND_PREF_ATTRS:
            return
        if backend in (None, 'auto'):
            self._backend_pref.pop(attr_name, None)
        elif backend in ('trt', 'onnx'):
            self._backend_pref[attr_name] = backend
        else:
            return

    def set_backend_preference_for_all(self, backend, *, unload=True):
        """把后端偏好应用到所有支持切换的模型上。

        CLI 的 --backend 是全局开关；逐个枚举属性容易在新增模型时漏项
        （hyperswap 就曾因此仍在 --backend cuda 下去初始化 TRT EP）。
        """
        for attr_name in self._BACKEND_PREF_ATTRS:
            self.set_backend_preference(attr_name, backend, unload=unload)

    def set_model_session_mode(self, mode: str) -> None:
        """切换 ORT session 的共享策略，必须在任何模型加载前调用。

        'Per-Thread' 让每个 worker 线程各持一份 session 与 CUDA stream，
        是 --workers > 1 真正并行占用 GPU 的前提。
        """
        if mode not in ("Shared", "Per-Thread"):
            raise ValueError(f"unknown model session mode: {mode}")
        self._model_session_mode = mode

    def set_arena_strategy(self, strategy: str) -> None:
        """设置 CUDA EP 的 arena 扩展策略，必须在任何 session 加载前调用。

        只接受 ORT 认识的两个值，未知值直接报错而不是静默传给 ORT——
        拼错的策略名会被 ORT 当成默认行为，排查起来很费劲。这是阶段 3
        的 6.2：收益小，默认仍是 kSameAsRequested。
        """
        valid = ('kSameAsRequested', 'kNextPowerOfTwo')
        if strategy not in valid:
            raise ValueError(
                f"unknown arena strategy {strategy!r}; expected one of {valid}"
            )
        self._arena_strategy = strategy

    def set_reuse_detect_buffers(self, enabled: bool, backend: str = "cuda") -> None:
        """阶段 6：是否复用 retinaface 的 9 个 io_binding 输出缓冲。

        ``resolve_reuse_buffers`` 会在 TRT-EP / 未知 backend 下关掉它：TRT 的
        io_binding 输出行为未经真机核对，不猜。默认 False 时
        ``_detect_retinaface_inner`` 一帧一分配的旧路径逐字保留。
        """
        self._reuse_detect_buffers = resolve_reuse_buffers(enabled, backend)

    def get_backend_preference(self, attr_name):
        """返回 'trt' / 'onnx' / None（auto）。"""
        return self._backend_pref.get(attr_name)

    def delete_models(self):

        self.retinaface_model = []
        self.scrdf_model = []
        self.resnet50_model = []
        self.recognition_model = []
        self.swapper_model = []
        self.swapper_512_model = []
        self.swapper_256_model = []
        self.hyperswap_model = []
        self.GFPGAN_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.codeformer_model = []
        self.occluder_model = []
        self.faceparser_model = []
        self.dfl_xseg_model = []
        # TRT/batched-unsupported 标志也一并丢掉——对应的懒加载器下次
        # 使用时会重新探测（这样在 Clear VRAM 之后做的 Build TensorRT
        # 会被自动识别到）。
        self._swapper_uses_trt = False
        self._swapper_batch_unsupported = False
        self._swapper_512_uses_trt = False
        self._swapper_512_batch_unsupported = False
        self._swapper_256_uses_trt = False
        self._hyperswap_uses_trt = False
        self._retinaface_uses_trt = False
        self._recognition_uses_trt = False

    def run_recognize(self, img, kps, dim=1):
        with nvtx_range("run_recognize"):
            if not self.recognition_model:
                self._load_recognition_model()

            embedding, cropped_image = self.recognize(img, kps, dim)
            return embedding, cropped_image
    

    def calc_swapper_latent(self, source_embedding, *, scale=1.0, extrap_amount=0.0, latent_mode='emap'):
        """计算 swapper 的 conditioning latent。相比原始 insightface 公式
        多出来的几个旋钮：
            extrap_amount: 在 normalize -> emap -> normalize 这条管线之前，
                把 source_embedding 推离会话均值（通过
                目前没有任何调用方设置它，所以 extrapolation 恒为空转）。
                0 = 不改变。
            scale: 乘在最终 latent 上。>1 会增强换脸结果里的身份强度。
            latent_mode: 'emap'（默认——insightface inswapper:
                normalize -> emap 投影 -> normalize）或 'raw'
                （只做 L2 归一化；不做 emap 投影——给直接消费 raw
                embedding 的模型用）。
        """
        s = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
        mean = getattr(self, '_mean_emb', None)
        if extrap_amount > 0.0 and mean is not None and mean.shape == s.shape:
            # 相减前先把 mean 重新缩放到与 s_e 相同的量级，这样无论用户的
            # embedding 当初是怎么归一化的，extrapolation 都在一个一致的
            # 空间里进行。
            s_norm = float(np.linalg.norm(s))
            mean_scaled = mean * s_norm
            s = s + float(extrap_amount) * (s - mean_scaled)
        n_e = s / l2norm(s)
        latent = n_e.reshape((1, -1))

        if latent_mode == 'raw':
            # Hyperswap 路径: 模型直接消费 raw L2-normalized embedding。
            # 没有 emap，也没有第二次 normalize。
            if scale != 1.0:
                latent = latent * float(scale)
            return latent

        # 首次 emap 模式调用时懒加载 emap 投影矩阵。历史上这里用的是
        # graph.initializer[-1]，但 FP16 ONNX 在 emap 之后还有一个 32x32 的
        # _ln_scale_32x32 tensor，那个启发式就失效了。改为按名字查找
        # （'buff2fs' 在 FP32 与 FP16 导出中都是稳定的），并回退到
        # "那个 (512, 512) 的 initializer"，对将来的改名留一点余量。
        if getattr(self, 'emap', None) is None or len(self.emap) == 0:
            graph = onnx.load(self._mp("inswapper_128.fp16.onnx")).graph
            emap_arr = None
            for init in graph.initializer:
                if init.name == 'buff2fs':
                    emap_arr = onnx.numpy_helper.to_array(init)
                    break
            if emap_arr is None:
                for init in graph.initializer:
                    arr = onnx.numpy_helper.to_array(init)
                    if arr.shape == (512, 512):
                        emap_arr = arr
                        break
            if emap_arr is None:
                raise RuntimeError(
                    "emap (512x512) not found in inswapper_128.fp16.onnx"
                )
            # FP16 ONNX 把 emap 存成 float16；提升到 float32，让下游的
            # np.dot 保持 FP32，数值上更安全。
            self.emap = emap_arr.astype(np.float32, copy=False)

        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        if scale != 1.0:
            latent = latent * float(scale)
        return latent

    def embed_face_chip(self, img_chw_uint8):
        """对一张已对齐的人脸裁剪图求 ArcFace embedding。跳过了
        `recognize()` 里那步由 kps 驱动的相似变换——传入的必须已经是
        裁紧且对齐的人脸。High Fidelity 用它衡量换脸结果的身份一致性，
        不需要重新检测 kps。

        img_chw_uint8: torch (3, H, W) uint8 RGB CUDA tensor；函数内部
            会缩放到 112x112。
        返回: numpy (512,) float32 embedding（未做 L2 归一化，与
            `recognize()` 的返回一致）。"""
        # 缩放到识别器期望的 112x112 输入。
        img = img_chw_uint8.to(torch.float32)
        if img.shape[-1] != 112 or img.shape[-2] != 112:
            img = v2.functional.resize(img, [112, 112], antialias=True)
        # ArcFace 要求 BGR，(x - 127.5) / 127.5。
        img = img[[2, 1, 0]]
        img = (img - 127.5) / 127.5
        img = img.unsqueeze(0).contiguous()

        recognition_session = self._get_recognition_session()
        output = torch.empty((1, 512), dtype=torch.float32, device='cuda').contiguous()
        io_binding = recognition_session.io_binding()
        io_binding.bind_input(
            name=recognition_session.get_inputs()[0].name, device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 112, 112),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(
            name=recognition_session.get_outputs()[0].name, device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=output.data_ptr(),
        )
        if self._should_drain_syncvec():
            self.syncvec.cpu()
        recognition_session.run_with_iobinding(io_binding)
        return output.squeeze().cpu().numpy()
        
    # @profile
    def run_swapper(self, image, embedding, output):
        # 取与调用线程相称的 session。Shared 模式下就是唯一的
        # self.swapper_model；Per-Thread 模式下每个 worker 各有一份
        # session，inswapper 才能真正并行跑。
        swapper_session = self._get_swapper_session()

        # 当前加载的 inswapper ONNX 可能声明 FP16 或 FP32 的 I/O
        # （加载时记入 _swapper_io_dtype）。调用方一律传入并期待 FP32
        # 的 torch tensor，所以模型要求 FP16 时在边界处转换——输入用
        # .half()，输出用一个临时 FP16 缓冲，再拷回调用方的缓冲。
        io_dtype = self._swapper_io_dtype
        if io_dtype == np.float16:
            image_in = image.half() if image.dtype != torch.float16 else image
            embedding_in = embedding.half() if embedding.dtype != torch.float16 else embedding
            output_buf = torch.empty_like(output, dtype=torch.float16).contiguous()
        else:
            image_in = image
            embedding_in = embedding
            output_buf = output

        # 两个后端（TRT-EP 与 CUDA-EP）返回的都是 ORT
        # InferenceSession，所以一条 io_binding 路径同时服务两者——
        # ORT 内部会把调用路由给当前生效的 provider。
        io_binding = swapper_session.io_binding()
        io_binding.bind_input(name='target', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,3,128,128), buffer_ptr=image_in.data_ptr())
        io_binding.bind_input(name='source', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,512), buffer_ptr=embedding_in.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=io_dtype, shape=(1,3,128,128), buffer_ptr=output_buf.data_ptr())

        # ORT 读取绑定的输入缓冲前，先 drain 掉 torch 在当前 CUDA stream
        # 上未完成的写入——但只在 torch 与 ORT 位于不同 stream 时才需要。
        # Per-Thread 模式下 worker stream 处于活动状态时，该 ORT session
        # 是用指向 torch 正使用的同一条 stream 的
        # `user_compute_stream` 构建的，于是 ORT 把自己的 kernel 严格
        # 排在那一条 stream 上 torch 的 kernel 之后——不存在竞争，也不需要
        # 同步。在这里省掉 syncvec drain 是 per-thread stream 重构的结构性
        # 回报；它消除了此前 nsys trace 显示为下一个瓶颈的、发生在全局默认
        # stream 上的跨 worker 主机端停顿。
        if self._swapper_should_drain_syncvec():
            with nvtx_range("syncvec_drain"):
                self.syncvec.cpu()
        with nvtx_range("ort_run_swapper"):
            swapper_session.run_with_iobinding(io_binding)

        # 如果刚才跑的是 FP16 模式，把结果搬回调用方的 FP32 缓冲。
        # copy_ 会处理 dtype 转换。
        if io_dtype == np.float16:
            output.copy_(output_buf)

    def _create_retinaface_session(self):
        """构建一个新的 retinaface ORT-TRT-EP session。遵守后端偏好：
        'onnx' → 只用 CUDA-EP；默认 → TRT-EP，输入 profile 动态覆盖
        320-640，TRT-EP 初始化失败则回退 CUDA-EP。
        `user_compute_stream` 会在调用线程有 torch stream 活动时
        （Per-Thread + worker stream）自动接上它。"""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('retinaface_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('det_10g.onnx')
        # 按加载的模型实际声明的名字来绑定——ONNX 导出的命名并不统一
        # （input.1 / input、448 / scores8 / …）。
        def _io_names(path):
            m = onnx.load(path, load_external_data=False)
            return ([i.name for i in m.graph.input],
                    [o.name for o in m.graph.output])
        det_in, det_out = _io_names(onnx_path)
        det_img_name = det_in[0]
        if det_out:
            print('[Models] retinaface IO: input=%r outputs=%r' % (det_in, det_out))
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 2 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # 注意: 我们有意不设 `trt_dump_ep_context_model=True`。
                # 该标志会在 cache 目录里 engine 旁边多写一个 EP-context-model
                # ONNX 包装文件；而且 ORT 还要求这个标志打开时 cache 路径
                # 必须是相对路径（出于安全考虑，它会在每次 session 加载时
                # 打一条 [E] 警告）。我们靠 `trt_engine_cache_enable=True`
                # 换后续加载的速度——那条路径与绝对 cache 目录相处良好，
                # 也正是它让 Per-Thread 模式在首次构建之后变得便宜。
                # 动态输入 profile: 一个 engine 覆盖所有 DetectInputSize
                # 取值（320 / 416 / 480 / 640）。opt 取 640——TRT 会针对
                # 这个点做特化。
                'trt_profile_min_shapes': '%s:1x3x320x320' % det_img_name,
                'trt_profile_opt_shapes': '%s:1x3x640x640' % det_img_name,
                'trt_profile_max_shapes': '%s:1x3x640x640' % det_img_name,
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='HEURISTIC')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] retinaface TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # retinaface 的输入是动态的 320→640 profile，所以 EXHAUSTIVE
            # 会在每次 DetectInputSize 变化时重新 benchmark（每个 session
            # 都要来一次）。HEURISTIC 不做 benchmark 就能挑到一个接近最优
            # 的算法 → 不会出现交互卡顿。
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='HEURISTIC')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._retinaface_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # profile 上下界就是我们上面配置的那些；不需要 introspection
        # engine，因为 ORT-TRT-EP 会照我们提的做。
        self._retinaface_trt_input_min = 320
        self._retinaface_trt_input_max = 640
        self._retinaface_engine_size_warned = False
        self._retinaface_io_names = (det_img_name, det_out)
        backend = 'TensorrtExecutionProvider' if self._retinaface_uses_trt else 'CUDAExecutionProvider'
        print(f'[Models] retinaface: {backend} (accepts input 320-640)')
        return sess

    def _load_retinaface_model(self):
        """Shared 模式加载器: 构建 session 并存到 self.retinaface_model。
        Per-Thread 模式直接走 _get_retinaface_session。"""
        self.retinaface_model = self._create_retinaface_session()

    def _get_retinaface_session(self):
        return self._get_model_session(
            'retinaface', 'retinaface_model', self._create_retinaface_session,
        )

    def _get_retinaface_input_buffer(self, input_size):
        """retinaface ORT 调用用的、按 worker 缓存的 (1, 3, H, W) float32
        输入缓冲。首次使用时按 (worker 线程, input_size) 懒构建。调用方
        往里清零再写入；ORT 的 io_binding 直接读 .data_ptr()。每种
        input_size 只分配一次，所以 DetectInputSize 滑杆不动时，detect
        的热路径上完全没有分配器压力。"""
        bufs = getattr(self._retinaface_buffers_tls, 'cache', None)
        if bufs is None:
            bufs = {}
            self._retinaface_buffers_tls.cache = bufs
        key = int(input_size)
        buf = bufs.get(key)
        if buf is None:
            buf = torch.zeros(
                (1, 3, key, key), dtype=torch.float32, device='cuda:0',
            )
            bufs[key] = buf
        return buf

    def _get_retinaface_output_buffers(self, input_height, input_width, strides):
        """阶段 6：按 worker 复用的 9 个 io_binding 输出缓冲。

        shape 只由 ``(input_height, input_width, strides)`` 决定，一次 video
        里恒定。缓存挂在同一个 ``_retinaface_buffers_tls`` 上，所以每个
        worker 各有一组——不同 worker 的 stream 并发，共享会让 ORT 的写入
        互相踩。同 worker 内是安全的：``rf_postprocess`` 在 worker 自己的
        stream 上读完这一帧，下一帧的 ORT 写入才排到它后面。

        ``OutputBufferCache`` 是 stdlib-only，``torch.empty`` 由这里的 lambda
        注入，缓存命中/失效的行为因此有离线行为测试覆盖。
        """
        tls = self._retinaface_buffers_tls
        cache = getattr(tls, 'output_cache', None)
        if cache is None:
            cache = OutputBufferCache(
                lambda shape, spec: torch.empty(
                    shape, dtype=torch.float32, device='cuda',
                )
            )
            tls.output_cache = cache
        return cache.get(input_height, input_width, strides)

    def _get_retinaface_gpu_anchors(self, height, width, stride):
        """给定 det_10g feature map，返回 CUDA 上 (height*width*2, 2) 的
        anchor center torch tensor。检测器每格 2 个 anchor；翻倍的行按
        交错排列，与模型输出的顺序一致。

        缓存在 self._retinaface_center_cache 上。每个 (h, w, stride)
        组合首次调用会构建并 torch.cuda.synchronize()，让写入对其他
        worker 的 stream 可见；后续调用直接返回缓存的 tensor，不产生
        同步。"""
        key = (int(height), int(width), int(stride))
        cached = self._retinaface_center_cache.get(key)
        if cached is not None:
            return cached
        y_coords = torch.arange(height, dtype=torch.float32, device='cuda:0')
        x_coords = torch.arange(width, dtype=torch.float32, device='cuda:0')
        # indexing='ij' → yy[h, w] = h, xx[h, w] = w。沿最后一轴 stack
        # [xx, yy] 得到 anchors[h, w] = (x, y)，x 是列、y 是行，与
        # numpy 原版的 np.mgrid[:H, :W][::-1] 一致。
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        anchors = torch.stack([xx, yy], dim=-1) * float(stride)  # (H, W, 2)
        anchors = anchors.reshape(-1, 2)                          # (H*W, 2)
        # 每行复制一份，对应每格 2 个 anchor。用 expand+reshape 做交错
        # （与 np.stack([a]*2, axis=1).reshape 等价）。
        anchors = anchors.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 2).contiguous()
        # 强制构建 kernel 落地，避免另一个 stream 上的 worker 读到尚未
        # 写完的缓存项。只在缓存未命中时付这笔钱（应用启动时每种
        # input_size 约 3 次）。
        torch.cuda.synchronize()
        self._retinaface_center_cache[key] = anchors
        return anchors

    def _create_recognition_session(self):
        """构建一个新的 arcface (w600k_r50) ORT-TRT-EP session。与
        retinaface 同一套模式: 默认 TRT-EP，失败回退 CUDA-EP。输入是
        固定的 (1, 3, 112, 112) 人脸 chip——不需要动态 profile。"""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('recognition_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('w600k_r50.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 1 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # 注意: 我们有意不设 `trt_dump_ep_context_model=True`。
                # 该标志会在 cache 目录里 engine 旁边多写一个 EP-context-model
                # ONNX 包装文件；而且 ORT 还要求这个标志打开时 cache 路径
                # 必须是相对路径（出于安全考虑，它会在每次 session 加载时
                # 打一条 [E] 警告）。我们靠 `trt_engine_cache_enable=True`
                # 换后续加载的速度——那条路径与绝对 cache 目录相处良好，
                # 也正是它让 Per-Thread 模式在首次构建之后变得便宜。
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] arcface TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # arcface 输入固定是 112×112 → EXHAUSTIVE，一次性 warmup。
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._recognition_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        backend = 'TensorrtExecutionProvider' if self._recognition_uses_trt else 'CUDAExecutionProvider'
        print(f'[Models] arcface (w600k_r50): {backend}')
        return sess

    def _load_recognition_model(self):
        self.recognition_model = self._create_recognition_session()

    def _get_recognition_session(self):
        return self._get_model_session(
            'recognition', 'recognition_model', self._create_recognition_session,
        )

    def _create_swapper_session(self):
        """构建并返回一个新的 inswapper ORT InferenceSession，按用户的
        后端偏好挑选 provider。Shared 与 Per-Thread 模式都会调到这里；
        由调用方决定把返回的 session 存到哪。

        Provider 语义:

            'onnx'（或没有可用 TRT）→ CUDAExecutionProvider
                纯 cuDNN/cuBLAS 路径。总能跑。TRT 初始化失败时作为
                兜底。
            'trt'（缺省时默认）    → TensorrtExecutionProvider
                ORT 首次运行时会内部构建并缓存一个 TensorRT engine。
                最关键的选项是 trt_layer_norm_fp32_fallback=True——
                少了它，FP16 的 LayerNorm 累加器会在 inswapper 图上
                溢出，产出肉眼可见的坏输出（与我们手工构建 FP16 engine
                时撞上的是同一个故障）。这个选项是 ORT-TRT-EP 特有的
                旋钮，原生 TRT API 里没有对应物，这也是我们像
                retinaface/arcface 那样选择走 ORT、而不是自己构建
                engine 的原因。

        思路来自 VisoMaster 的 models_processor.trt_ep_options。ORT 会把
        构建好的 engine 持久化到 {models_folder}/ort_trt_cache，后续
        session 能快速重载。

        构建完成后我们读 session.get_providers() 来得知实际生效的是哪个
        provider——ORT 在 TRT-EP 初始化失败时会静默回退到 CUDA EP。
        _swapper_uses_trt 标志反映的是现实，而不是请求。"""
        # 防御性: 就在创建 session 之前把 env 层的 Default logger 严重级别
        # 夹紧。ORT_LOG_SEVERITY_LEVEL 本已在 faceswapper/_native_dlls.py
        # 里、ORT import 之前设好，但如果 ORT 因故被更早导入（site-packages
        # 的传递性 import、热重载），Env logger 可能已按 INFO 创建。通过
        # 进程内 API 重新声明一次，可以消掉 BFCArena 的啰嗦输出——否则它会
        # 在第一个 CUDA-EP session 上灌满 stdout。
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('swapper_model')
        want_trt = pref != 'onnx'

        onnx_path = self._mp("inswapper_128.fp16.onnx")
        sess = None

        if want_trt:
            # ORT-TRT-EP 会把配置的 cache 路径拼到自己的工作目录基址后面
            # ——传相对路径会拼出 "./models\ort_trt_cache\./models\ort_trt_cache"
            # 这种废话，然后 provider 静默回退到 CUDA EP。所以必须传绝对
            # 且已规范化的路径。
            cache_dir = os.path.abspath(self._mp("ort_trt_cache"))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                # 关键标志: 即使整体跑 FP16，也让 LayerNorm 的归约留在
                # FP32。少了它，inswapper 的 FP16 路径会产出视觉上完全
                # 坏掉的输出。
                'trt_layer_norm_fp32_fallback': True,
                'trt_builder_optimization_level': 5,
                # 注意: 我们有意不设 `trt_dump_ep_context_model=True`。
                # 该标志会在 cache 目录里 engine 旁边多写一个 EP-context-model
                # ONNX 包装文件；而且 ORT 还要求这个标志打开时 cache 路径
                # 必须是相对路径（出于安全考虑，它会在每次 session 加载时
                # 打一条 [E] 警告）。我们靠 `trt_engine_cache_enable=True`
                # 换后续加载的速度——那条路径与绝对 cache 目录相处良好，
                # 也正是它让 Per-Thread 模式在首次构建之后变得便宜。
            }
            # 如果调用线程自己设了 current CUDA stream（Per-Thread 模式下
            # VideoManager 的 worker 模式正是如此），通过
            # user_compute_stream 把它交给 ORT-TRT-EP，ORT 就会把自己的
            # kernel 派发到 torch 正在用的同一条 stream 上。torch 与 ORT
            # 于是共享执行顺序——不存在跨 stream 竞争，syncvec 的 drain
            # 不再必要，worker 也不会在全局默认 stream 上串行化。
            try:
                cur_stream = torch.cuda.current_stream()
                # worker 跑在默认 stream 上时跳过（Shared 模式，或没有
                # 启用 per-worker stream）: device→host 的 syncvec 已足以
                # 保证跨 stream 顺序，而把 ORT 绑到 torch 本来就会串行化
                # 的那条 stream 上并不会带来额外收益。
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                # CUDA-EP 兜底路径带着同一组调好的选项（包括
                # user_compute_stream），这样一次静默的 TRT→CUDA 回退仍然
                # 绑在 worker stream 上——只写一个裸字符串条目会丢掉 stream，
                # 让推理跑到默认 stream 上去。
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] inswapper TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # inswapper_128 输入固定是 128×128，所以 EXHAUSTIVE 那一次性的
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        # TRT-EP 真的被选上了吗？get_providers() 按优先级返回生效的
        # provider 列表；第一项是 TRT 就说明它热着。静默回退（请求了
        # TRT-EP 但 ORT 仍然选了 CUDA）在这里会得到 False。只探测一次并
        # 缓存——同一后端偏好下同一模型的所有 session 都会落到同一个
        # provider，所以没必要为每个 Per-Thread session 重新探测。
        active = sess.get_providers()
        self._swapper_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # 探测模型实际的 I/O dtype。真正全 FP16 的 inswapper 导出会声明
        # FP16 输入/输出并要求绑定 FP16 缓冲；较早的"FP16 权重、FP32 IO"
        # 导出声明 FP32，可直接接受 FP32 缓冲。我们据此在 io_binding
        # 边界做转换。ORT 的 input.type 是形如 "tensor(float16)" 或
        # "tensor(float)" 的字符串。
        inp = sess.get_inputs()[0]
        self._swapper_io_dtype = (
            np.float16 if 'float16' in inp.type else np.float32
        )
        # 探测静态 batch 的 ONNX 导出（shape[0] 是 int，而不是符号维的
        # 字符串/None）。若是，就预置 batch-unsupported 标志，让
        # run_swapper_batched 直接走逐次调用的兜底路径，不必付那一次性
        # 异常的代价。
        try:
            shape0 = inp.shape[0]
            self._swapper_batch_unsupported = isinstance(shape0, int) and shape0 == 1
        except Exception:
            self._swapper_batch_unsupported = False
        backend = 'TensorrtExecutionProvider' if self._swapper_uses_trt else 'CUDAExecutionProvider'
        batch_note = ' static-batch=1' if self._swapper_batch_unsupported else ''
        print(f'[Models] inswapper: {backend} (io={self._swapper_io_dtype.__name__}{batch_note})')
        return sess

    # ===== 通用 per-thread session 分发器 =================================
    # 主管线里的每个 ORT-TRT-EP 模型（inswapper、inswapper_512、
    # retinaface、recognition）都走这几个 helper。模式是进程级的；每个
    # 模型自带自己的 (name, shared-attr, factory)，所以同一个分发器能
    # 服务所有这些模型。

    def _get_model_session(self, name, shared_attr, create_fn):
        """按当前模式返回 `name` 对应的 ORT session。

        Shared 模式: 懒加载进 self.<shared_attr>，所有线程共用同一份
        session。Per-Thread 模式: 懒加载进 thread-local 存储，每个
        worker 各得一份 session。磁盘上的 TRT engine 缓存是共享的，所以
        第 2 到第 N 个 Per-Thread session 会反序列化已有的 engine，
        而不是重新构建。

        create_fn() 是产出新 session 的工厂——它运行在调用线程上，这样
        它的 `user_compute_stream`（如果有）会接上该线程当前的 torch
        stream。"""
        if self._model_session_mode == 'Per-Thread':
            sess = getattr(self._sessions_tls, name, None)
            if sess is None:
                sess = create_fn()
                setattr(self._sessions_tls, name, sess)
                with self._sessions_lock:
                    self._all_sessions.setdefault(name, []).append(sess)
            return sess
        # Shared 模式: 只懒加载一次，所有线程复用。
        sess = getattr(self, shared_attr, None)
        if not sess:
            sess = create_fn()
            setattr(self, shared_attr, sess)
        return sess

    def _should_drain_syncvec(self):
        """为 True 时，ORT 调用前需要显式 drain 一次 torch stream，ORT
        才会读到输入缓冲的最新写入。只有当 torch 当前 stream 与构建该
        ORT session 时用的 stream 是同一条时才为 False——这发生在
        Per-Thread 模式下 worker 正跑在自己的
        `with torch.cuda.stream(worker_stream)` 上下文里。同一条
        stream = 隐式顺序，不存在竞争。"""
        if self._model_session_mode != 'Per-Thread':
            return True
        try:
            cur = torch.cuda.current_stream()
            return cur is None or cur == torch.cuda.default_stream()
        except Exception:
            return True

    # ===== 共享的 ORT session 配置 =======================================
    # 每个 _create_*_session 工厂里，显式 ONNX 分支和 TRT 分支的 CUDA
    # 兜底都走这两个 helper，这样 CUDA-EP 路径在各处被调成完全一致
    # （此前每个工厂各自内联一份——彼此还不一样）。

    def _cuda_ep_provider_options(self, cudnn_algo='EXHAUSTIVE'):
        """CUDAExecutionProvider 选项 dict。每次调用返回一个新 dict
        （ORT 可能会改它，而且 stream 的值是线程相关的）。必须在 worker
        线程上调用，这样 `torch.cuda.current_stream()` 取到的是该 worker
        的 stream——与 TRT 分支早已使用的构建期契约相同。

        cudnn_algo:
            'EXHAUSTIVE'（默认）——benchmark 出最快的 conv 算法。稳态
                最好；首次推理时按 shape 做一次 benchmark。适合输入
                shape 固定的模型（swapper、arcface、inswapper_512），

            'HEURISTIC'——用解析方式挑一个接近最优的算法，不做
                benchmark。适合 retinaface，它的输入是动态的 320→640
                profile（EXHAUSTIVE 会在每次 DetectInputSize 变化时
                重新 benchmark，每个 session 都要卡一次）。
        """
        opts = {
            'arena_extend_strategy': self._arena_strategy,
            'cudnn_conv_algo_search': cudnn_algo,
        }
        try:
            cur = torch.cuda.current_stream()
            if cur is not None and cur != torch.cuda.default_stream():
                opts['user_compute_stream'] = str(int(cur.cuda_stream))
        except Exception:
            pass
        return opts

    def _make_session_options(self):
        """调过的 SessionOptions，用来避免 CPU 线程超订。ORT 默认按 CPU
        核心数给每个 session 配 intra-op 线程池；Per-Thread 模式下约有
        3·N 个 session 同时活着（N = worker 线程数），那样就是
        N·3·core_count 个 ORT 线程互相抢，还要和 N 个 torch worker 抢
        ——这对 CUDA-EP（逐节点派发）的伤害远大于 TRT-EP（整个融合成
        一个 engine 节点）。我们让每个 session 保持精简，并发度交给那
        N 个 worker 线程去提供。

        另外设 log_severity_level=3（此前这是唯一被设置的一项），因为
        ORT 1.26 通过 -1 继承默认严重级别的行为是坏的，否则每个 session
        初始化时都会喷出约 100 行 INFO。"""
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if self._model_session_mode == 'Per-Thread':
            # 每个 session 一个 intra-op 线程；并发度来自 N 个各自驱动
            # 自己 session 的 worker 线程。
            so.intra_op_num_threads = 1
        else:
            # Shared 模式: 总共约 3 个 session，被 N 个并发 Run() 调用
            # 访问——给它们一个小池子，避免 CPU 侧工作被串行化，但仍要
            # 明显低于 core_count，免得又把超订引回来。
            cores = os.cpu_count() or 4
            so.intra_op_num_threads = max(2, min(8, cores // 2))
        return so

    def clear_per_thread_sessions(self, name=None):
        """释放对 Per-Thread session 的强引用，让垂死 worker 的 session
        得以 GC。传 `name` 只清某个模型的 per-thread session；传 None
        全部清掉。每个活着的 worker 的 TLS slot 保持不动，正在进行的
        调用不受影响——Python 的 threading.local 会在 worker 退出时
        自动清理该线程的 slot，而一旦强引用列表不再持有它，session
        就会被 GC。"""
        with self._sessions_lock:
            if name is None:
                self._all_sessions = {}
            else:
                self._all_sessions.pop(name, None)
        self.vram_dirty = True

    # ===== Inswapper session 粘合层 =======================================
    # Inswapper 是通用分发器上的第一个模型。保留 `_get_swapper_session`
    # 这个 helper 名，这样 run_swapper / run_swapper_batched 里已有的
    # 调用点无需改动就能继续工作。

    def _get_swapper_session(self):
        return self._get_model_session(
            'swapper', 'swapper_model', self._create_swapper_session,
        )

    def swapper_supports_dynamic_batch(self) -> bool:
        """当前 swapper ONNX 图是否接受动态 batch 维。

        ``_create_swapper_session`` 在懒加载时探测输入第 0 维；若锁死为 1
        （日志里的 ``static-batch=1``），N>1 的 ``io_binding`` 会被 ONNX
        拒绝。这里显式触发一次懒加载，避免调用方在 session 尚未创建时把
        ``_swapper_batch_unsupported`` 的默认值 ``False`` 误读成「支持」。
        """
        self._get_swapper_session()
        return not bool(getattr(self, '_swapper_batch_unsupported', False))

    def _swapper_should_drain_syncvec(self):
        return self._should_drain_syncvec()
    # @profile
    def run_swapper_batched(self, image_batch, embedding_batch, output_batch):
        """对共享同一个 source 的 N 个 target tile 批量跑 swapper。
        shape:
            image_batch    (N, 3, 128, 128) float32 cuda——连续
            embedding_batch(1, 512) 或 (N, 512) float32 cuda——只读第 0 行；
                inswapper 的 source 绑定固定是 (1, 512)，模型内部会做
                广播。(N, 512) 是为了兼容既有代码路径仍在产出这种形状的
                调用方，但那是浪费内存。
            output_batch   (N, 3, 128, 128) float32 cuda——已预分配
        模型第一次批量尝试失败时回退到逐次调用 run_swapper（某些
        inswapper 导出把 batch 维锁死为 1）。首次失败后 unsupported 标志
        会粘住，后续调用跳过探测。

        两个后端（ORT TRT-EP 与 CUDA-EP）都通过单次 io_binding 调用接受
        动态 batch——ONNX 本身把 batch 维声明为符号，ORT 会把这次调用
        路由给当前生效的 provider。"""
        # 用一次 N=1 的探测调用来懒初始化合适的 session（Shared 或
        # Per-Thread）。这之后 run_swapper 已填好 _swapper_io_dtype 与
        # _swapper_batch_unsupported，调用线程的 session 也就绪了。
        n = image_batch.shape[0]
        # source 在所有 target 间共享，所以无论调用方传的是 (1, 512) 还是
        # (N, 512)，一律只绑定第 0 行。
        emb_one_caller = embedding_batch[0:1]
        if n == 1:
            self.run_swapper(image_batch, emb_one_caller, output_batch)
            return

        # 先触发 session 懒加载：_create_swapper_session 会探测 ONNX 的
        # batch 维并设置 _swapper_batch_unsupported。若把检查放在加载之前，
        # 标志可能还是 __init__ 里的默认 False，首次 N>1 调用会直闯
        # io_binding 而被 ONNX 拒绝（11-06 受控实验的 CUDA 700）。
        swapper_session = self._get_swapper_session()
        if getattr(self, '_swapper_batch_unsupported', False):
            for k in range(n):
                self.run_swapper(
                    image_batch[k:k+1], emb_one_caller, output_batch[k:k+1],
                )
            return

        io_dtype = self._swapper_io_dtype
        if io_dtype == np.float16:
            image_in = image_batch.half() if image_batch.dtype != torch.float16 else image_batch
            # source 只有第一行会被读到；所以只把这一片转成 FP16，而不是
            # 把全部 N 行都转。
            embedding_in = emb_one_caller.half() if emb_one_caller.dtype != torch.float16 else emb_one_caller
            output_buf = torch.empty_like(output_batch, dtype=torch.float16).contiguous()
        else:
            image_in = image_batch
            embedding_in = emb_one_caller
            output_buf = output_batch

        # inswapper 的 source 绑定固定是 (1, 512)；模型在内部把 source
        # 身份广播到整个 target batch。embedding_in 在上面已经通过
        # emb_one_caller 变成 (1, 512)；对连续的 row-0 分片而言
        # .contiguous() 是空操作。
        emb_one = embedding_in.contiguous()

        io_binding = swapper_session.io_binding()
        io_binding.bind_input(
            name='target', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(n, 3, 128, 128),
            buffer_ptr=image_in.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(1, 512),
            buffer_ptr=emb_one.data_ptr(),
        )
        io_binding.bind_output(
            name='output', device_type='cuda', device_id=0,
            element_type=io_dtype, shape=(n, 3, 128, 128),
            buffer_ptr=output_buf.data_ptr(),
        )
        # 与 run_swapper 相同的 torch-stream drain——取决于 ORT 与 torch
        # 是否在不同 stream 上。理由见 _swapper_should_drain_syncvec /
        # run_swapper。
        if self._swapper_should_drain_syncvec():
            with nvtx_range("syncvec_drain"):
                self.syncvec.cpu()
        try:
            with nvtx_range(f"ort_run_swapper_batched[n={n}]"):
                swapper_session.run_with_iobinding(io_binding)
        except Exception as e:
            print('[Models] swapper batched call rejected (%s: %s); '
                  'falling back to per-call for the rest of the session'
                  % (type(e).__name__, e))
            self._swapper_batch_unsupported = True
            for k in range(n):
                self.run_swapper(
                    image_batch[k:k+1], embedding_batch[k:k+1], output_batch[k:k+1],
                )
            return

        if io_dtype == np.float16:
            output_batch.copy_(output_buf)

    # ===== inswapper_512（512x512 进 / 512x512 出）=========================
    # 与 run_swapper / run_swapper_batched 并列的一套 API。conditioning
    # 向量相同（(1,512) float32），source-embedding → latent 的管线也相同
    # （都走 calc_swapper_latent）。只有 target/output 的空间 shape 不同。

    def run_swapper_512(self, image, embedding, output):
        swapper_512_session = self._get_swapper_512_session()

        io_binding = swapper_512_session.io_binding()
        io_binding.bind_input(
            name='target', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name='output', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self._should_drain_syncvec():
            self.syncvec.cpu()
        swapper_512_session.run_with_iobinding(io_binding)

    def _create_swapper_512_session(self):
        """构建一个新的 inswapper_512 ORT-TRT-EP session。输入是静态的
        (1, 3, 512, 512)——不需要动态 profile。engine 层面 batch 恒为 1；
        调用方每张脸调用一次 run_swapper_512。"""
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        pref = self._backend_pref.get('swapper_512_model')
        want_trt = pref != 'onnx'
        onnx_path = self._mp('inswapper_512_level2.onnx')
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': True,
                'trt_builder_optimization_level': 5,
                # 注意: 我们有意不设 `trt_dump_ep_context_model=True`。
                # 该标志会在 cache 目录里 engine 旁边多写一个 EP-context-model
                # ONNX 包装文件；而且 ORT 还要求这个标志打开时 cache 路径
                # 必须是相对路径（出于安全考虑，它会在每次 session 加载时
                # 打一条 [E] 警告）。我们靠 `trt_engine_cache_enable=True`
                # 换后续加载的速度——那条路径与绝对 cache 目录相处良好，
                # 也正是它让 Per-Thread 模式在首次构建之后变得便宜。
            }
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] inswapper_512 TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (type(e).__name__, e))
                sess = None

        if sess is None:
            # inswapper_512 输入固定是 512×512 → EXHAUSTIVE。
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        self._swapper_512_uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        # 探测静态 batch 的 ONNX 导出。inswapper_512_level2.onnx 通常带的是
        # batch=1（level-2 变体是逐次调用的）。若 batch 是符号维，则支持
        # 动态 batch。
        try:
            inp_shape = sess.get_inputs()[0].shape
            shape0 = inp_shape[0]
            self._swapper_512_batch_unsupported = (
                isinstance(shape0, int) and shape0 == 1
            )
        except Exception:
            self._swapper_512_batch_unsupported = True
        backend = 'TensorrtExecutionProvider' if self._swapper_512_uses_trt else 'CUDAExecutionProvider'
        batch_note = ' static-batch=1' if self._swapper_512_batch_unsupported else ''
        print(f'[Models] inswapper_512: {backend}{batch_note}')
        return sess

    def _get_swapper_512_session(self):
        return self._get_model_session(
            'swapper_512', 'swapper_512_model', self._create_swapper_512_session,
        )

    # ===== inswapper_256_phase1（256x256 原生，emap latent）==============
    # 单 tile 的原生 256 swapper。source conditioning 与 128 / 512 两个
    # inswapper 相同（喂给 `source` 的是 emap 投影后的 latent）。I/O
    # tensor 名是 `target256`（输入，1×3×256×256）、`source`（输入，
    # 1×512）和 `p2_refined`（输出，1×3×256×256）。该 ONNX 声明了动态的
    # 空间输入，所以 TRT-EP 分支把 profile 固定在 256（我们永远只喂
    # 256×256）。

    def run_swapper_256(self, image, embedding, output):
        swapper_256_session = self._get_swapper_256_session()

        io_binding = swapper_256_session.io_binding()
        io_binding.bind_input(
            name='target256', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name='p2_refined', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self._should_drain_syncvec():
            self.syncvec.cpu()
        swapper_256_session.run_with_iobinding(io_binding)

    def _create_single_pass_swapper_session(
        self, onnx_name, *, fp16, profiles, label, backend_pref, uses_trt_attr,
    ):
        """为单次 pass 的 swapper 生成器构建一个 ORT session。

        `profiles` 固定住动态的空间输入（{input: shape 字符串} 映射）；
        ONNX 完全静态的生成器传 None。`fp16` 只为 engine 构建切换 TRT
        FP16——I/O 缓冲仍是 FP32，这两个导出模型声明的都是 FP32。

        两个 native-256 swapper 共用这一份：彼此只有 FP16 设置、profile
        固定值和文件名不同。
        """
        try:
            onnxruntime.set_default_logger_severity(3)
        except (AttributeError, OSError):
            pass

        want_trt = self._backend_pref.get(backend_pref) != 'onnx'
        onnx_path = self._mp(onnx_name)
        sess = None

        if want_trt:
            cache_dir = os.path.abspath(self._mp('ort_trt_cache'))
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            trt_ep_options = {
                'trt_max_workspace_size': 4 << 30,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache_dir,
                'trt_fp16_enable': fp16,
                'trt_builder_optimization_level': 5,
            }
            if profiles:
                trt_ep_options.update(profiles)
            try:
                cur_stream = torch.cuda.current_stream()
                if cur_stream is not None and cur_stream != torch.cuda.default_stream():
                    trt_ep_options['user_compute_stream'] = str(int(cur_stream.cuda_stream))
            except Exception:
                pass
            try:
                sess_options = self._make_session_options()
                cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
                sess = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=[
                        ('TensorrtExecutionProvider', trt_ep_options),
                        ('CUDAExecutionProvider', cuda_options),
                    ],
                )
            except Exception as e:
                print('[Models] %s TRT-EP init failed (%s: %s); '
                      'falling back to CUDA EP' % (label, type(e).__name__, e))
                sess = None

        if sess is None:
            cuda_options = self._cuda_ep_provider_options(cudnn_algo='EXHAUSTIVE')
            sess_options = self._make_session_options()
            sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', cuda_options)],
            )

        active = sess.get_providers()
        uses_trt = bool(active) and active[0] == 'TensorrtExecutionProvider'
        setattr(self, uses_trt_attr, uses_trt)
        backend = 'TensorrtExecutionProvider' if uses_trt else 'CUDAExecutionProvider'
        print(f'[Models] {label}: {backend}')
        return sess

    def _create_swapper_256_session(self):
        """构建一个新的 inswapper_256_phase1 ORT-TRT-EP session。该 ONNX
        有一个动态 (1, 3, H, W) 输入；我们永远只喂 256×256，所以 TRT
        profile 的 min=opt=max 全固定在 256。FP32 I/O；batch 恒为 1
        （run_swapper_256 每张脸调用一次）。"""
        # FP16 对这一系 native-256 inswapper 不安全。模型用分解后的
        # ReduceMean→Sub→Sqrt→Div 方差组做归一化，而 FP16 方差会出现
        # 灾难性抵消 → 一张模糊/带噪点的脸。整个 engine 按 FP32 构建:
        # 模型本身带的就是 FP32 权重（没有 FP16 导出），FP32 就是它的
        # 原生精度，而 TRT 靠 kernel 融合仍能胜过 CUDA-EP。代价: engine
        # VRAM 约为 FP16 构建的 2 倍（Per-Thread 模式下有 N 个 per-worker
        # engine 时值得注意）。（隔壁的 phase2 变体还额外带有
        # InstanceNormalization 算子，会让 trt_layer_norm_fp32_fallback
        # 失效——FP32 是这些 native-256 swapper 已验证的安全路径。）
        return self._create_single_pass_swapper_session(
            'inswapper_256_phase1.onnx',
            fp16=False,
            # 动态的空间输入固定到我们唯一喂的那一个尺寸。只有
            # `target256` 是动态的；`source` (1, 512) 是静态的，不需要
            # profile。
            profiles={
                'trt_profile_min_shapes': 'target256:1x3x256x256',
                'trt_profile_opt_shapes': 'target256:1x3x256x256',
                'trt_profile_max_shapes': 'target256:1x3x256x256',
            },
            label='inswapper_256_phase1',
            backend_pref='swapper_256_model',
            uses_trt_attr='_swapper_256_uses_trt',
        )

    def _get_swapper_256_session(self):
        return self._get_model_session(
            'swapper_256', 'swapper_256_model', self._create_swapper_256_session,
        )

    def run_hyperswap(self, image, embedding, output):
        """单次 pass 的 HyperSwap 生成器。

        image    (1, 3, 256, 256) float32 cuda，取值 [-1, 1]——裁剪图
                 已按 (x/255 - 0.5) / 0.5 归一化。这步由调用方
                 （_hyperswap_pass）做，因为同一个缓冲还要喂给要求 0-1
                 的 inswapper 路径。
        embedding (1, 512) float32 cuda，raw L2-normalized 身份向量
                 （latent_mode='raw'——不做 emap 投影）。
        output   (1, 3, 256, 256) float32 cuda，接收仍是 [-1, 1] 的
                 混合结果；由调用方做反归一化。

        ONNX 还输出第二个张量（'mask'），而图内部已经把它应用过了
        （output*mask + target*(1-mask)），所以它保持不绑定，不为它
        付代价。
        """
        session = self._get_hyperswap_session()

        io_binding = session.io_binding()
        io_binding.bind_input(
            name='target', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name='source', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name='output', device_type='cuda', device_id=0,
            element_type=np.float32, shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self._should_drain_syncvec():
            self.syncvec.cpu()
        session.run_with_iobinding(io_binding)

    def _create_hyperswap_session(self):
        """构建一个新的 hyperswap_1a_256 ORT session。

        该导出是 FP32（FaceFusion 有 fp16 IO 变体，但我们下载的那个声明
        的是 FP32），所以 TRT 与 CUDA 两条路径都绑定 FP32 缓冲，
        run_hyperswap 不做任何 dtype 转换。TRT FP16 只为 engine 构建启用
        ——它把 engine VRAM 减半，而这个生成器以卷积为主，AAD 堆栈里没有
        inswapper 需要 fp32-fallback 保护的那种 LayerNorm。所有输入都是
        静态的 (1,3,256,256) 和 (1,512)，所以不需要 TRT profile。
        """
        return self._create_single_pass_swapper_session(
            'hyperswap_1a_256.onnx',
            fp16=True,
            profiles=None,
            label='hyperswap_1a_256',
            backend_pref='hyperswap_model',
            uses_trt_attr='_hyperswap_uses_trt',
        )

    def _get_hyperswap_session(self):
        return self._get_model_session(
            'hyperswap', 'hyperswap_model', self._create_hyperswap_session,
        )

    def run_GFPGAN(self, image, output):
        if not self.GFPGAN_model:
            cuda_options = {"arena_extend_strategy": self._arena_strategy, 'cudnn_conv_algo_search': 'DEFAULT'}
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 3
            sess_options.enable_cpu_mem_arena = False

            # Fast load
            self.GFPGAN_model = onnxruntime.InferenceSession( self._mp("GFPGANv1.4.onnx"), sess_options, providers=[("CUDAExecutionProvider", cuda_options), 'CPUExecutionProvider'])

            # Normal load
            # self.GFPGAN_model = onnxruntime.InferenceSession( "./models/GFPGANv1.4.onnx", providers=self.providers)

        io_binding = self.GFPGAN_model.io_binding()
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())  
                
        self.syncvec.cpu()          
        self.GFPGAN_model.run_with_iobinding(io_binding)                

    
    def run_GPEN_512(self, image, output):
        if not self.GPEN_512_model:
            self.GPEN_512_model = onnxruntime.InferenceSession( self._mp("GPEN-BFR-512.onnx"), providers=self.providers)
 
        io_binding = self.GPEN_512_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())            
        
        self.syncvec.cpu()          
        self.GPEN_512_model.run_with_iobinding(io_binding)   

    def run_GPEN_256(self, image, output):
        if not self.GPEN_256_model:
            self.GPEN_256_model = onnxruntime.InferenceSession( self._mp("GPEN-BFR-256.onnx"), providers=self.providers)

        io_binding = self.GPEN_256_model.io_binding()         
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.GPEN_256_model.run_with_iobinding(io_binding) 

    def run_codeformer(self, image, output):   
        if not self.codeformer_model:    
            self.codeformer_model = onnxruntime.InferenceSession( self._mp("codeformer_fp16.onnx"), providers=self.providers)

        io_binding = self.codeformer_model.io_binding() 
        io_binding.bind_input(name='x', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([0.9], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())
        
        self.syncvec.cpu()          
        self.codeformer_model.run_with_iobinding(io_binding)        

    def run_occluder(self, image, output):    
        if not self.occluder_model:
            self.occluder_model = onnxruntime.InferenceSession(self._mp("occluder.onnx"), providers=self.providers)

        io_binding = self.occluder_model.io_binding()            
        io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')  
        self.syncvec.cpu()         
        self.occluder_model.run_with_iobinding(io_binding)  


    def run_faceparser(self, image, output):    
        if not self.faceparser_model:
            self.faceparser_model = onnxruntime.InferenceSession(self._mp("faceparser_resnet34.onnx"), providers=self.providers)

        io_binding = self.faceparser_model.io_binding()            
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())   

        # torch.cuda.synchronize('cuda')
        self.syncvec.cpu()
        self.faceparser_model.run_with_iobinding(io_binding)

    @staticmethod
    def _ort_type_to_np(type_str):
        """把 ORT NodeArg 的类型字符串（'tensor(float16)'）映射为 numpy
        dtype。无法识别的一律按 float32 处理。"""
        return np.float16 if 'float16' in type_str else np.float32

    def _load_dfl_xseg_model(self):
        """懒加载用户自备的 DeepFaceLab XSeg ONNX 并 introspection 它的
        I/O 契约。DFL 的原生导出是 NHWC (1, res, res, 3)，mask 输出也是
        NHWC (1, res, res, 1)；某些社区转换器给出的是 NCHW。节点名和
        分辨率同样会变，所以这些值都从图上读，不硬编码。"""
        self.dfl_xseg_model = onnxruntime.InferenceSession(
            self._mp("dfl_xseg.onnx"), providers=self.providers,
        )

        inp = self.dfl_xseg_model.get_inputs()[0]
        out = self.dfl_xseg_model.get_outputs()[0]
        self._dfl_xseg_in_name = inp.name
        self._dfl_xseg_out_name = out.name
        self._dfl_xseg_in_dtype = self._ort_type_to_np(inp.type)
        self._dfl_xseg_out_dtype = self._ort_type_to_np(out.type)

        # 从输入 shape 判定 layout 与分辨率。符号维（例如 'batch'）过来是
        # 字符串；通道/空间维通常是具体数值。dim 1 == 3 即 NCHW，否则按
        # NHWC 处理。
        shape = list(inp.shape)
        def _as_int(v, fallback):
            return v if isinstance(v, int) and v > 0 else fallback
        if len(shape) == 4 and shape[1] == 3:
            self._dfl_xseg_nhwc = False
            self._dfl_xseg_res = _as_int(shape[2], 256)
        else:
            self._dfl_xseg_nhwc = True
            self._dfl_xseg_res = _as_int(shape[1], 256)

        print(
            f"[Models] dfl_xseg.onnx loaded: res={self._dfl_xseg_res} "
            f"layout={'NHWC' if self._dfl_xseg_nhwc else 'NCHW'} "
            f"in='{self._dfl_xseg_in_name}'({inp.type}) "
            f"out='{self._dfl_xseg_out_name}'({out.type})"
        )

    def run_dfl_xseg(self, img):
        """对一张对齐的人脸跑 DeepFaceLab XSeg 遮挡模型。

        `img` 是 cuda 上 (3, H, W) 的 RGB float tensor，取值 [0, 255]
        （经管线对齐后的输入人脸）。返回 cuda 上 (res, res) 的 float32
        mask，取值 [0, 1]，其中 1 = 保留换脸结果，0 = 让原图透出来
        （DFL 约定: XSeg 把人脸区域涂成前景）。

        预处理与 DFL 一致: BGR 通道顺序，取值 [0, 1]。

        模型由用户自备，可能不存在；此时只警告一次，并返回全 1 的 mask
        （等于不做合成），而不是让换脸管线每帧都崩。"""
        if not self.dfl_xseg_model:
            path = self._mp("dfl_xseg.onnx")
            if not os.path.exists(path):
                if not getattr(self, '_dfl_xseg_missing_warned', False):
                    self._dfl_xseg_missing_warned = True
                    print(
                        f"[Models] DFL XSeg mask enabled but '{path}' is "
                        "missing — skipping (drop your exported XSeg .onnx "
                        "there to use it)."
                    )
                return torch.ones(
                    (self._dfl_xseg_res, self._dfl_xseg_res),
                    dtype=torch.float32, device='cuda',
                )
            self._load_dfl_xseg_model()

        res = self._dfl_xseg_res

        # (3, H, W) RGB [0,255] -> (1, 3, res, res) BGR [0,1]。对齐后的人脸
        # 是 uint8；先做类型转换，因为 F.interpolate 的 bilinear 路径没有
        # 实现 Byte dtype。
        x = img.unsqueeze(0).to(torch.float32)
        if x.shape[-1] != res or x.shape[-2] != res:
            x = torch.nn.functional.interpolate(
                x, size=(res, res), mode='bilinear', align_corners=False,
            )
        x = x.flip(1)            # RGB -> BGR (DFL trains on cv2/BGR faces)
        x = torch.div(x, 255.0)
        if self._dfl_xseg_nhwc:
            x = x.permute(0, 2, 3, 1)            # -> (1, res, res, 3)
        x = x.contiguous()
        if self._dfl_xseg_in_dtype == np.float16:
            x = x.half()
        else:
            x = x.float()

        if self._dfl_xseg_nhwc:
            in_shape = (1, res, res, 3)
            out_shape = (1, res, res, 1)
        else:
            in_shape = (1, 3, res, res)
            out_shape = (1, 1, res, res)

        out_torch_dtype = (
            torch.float16 if self._dfl_xseg_out_dtype == np.float16
            else torch.float32
        )
        outpred = torch.empty(
            out_shape, dtype=out_torch_dtype, device='cuda',
        ).contiguous()

        io_binding = self.dfl_xseg_model.io_binding()
        io_binding.bind_input(
            name=self._dfl_xseg_in_name, device_type='cuda', device_id=0,
            element_type=self._dfl_xseg_in_dtype, shape=in_shape,
            buffer_ptr=x.data_ptr(),
        )
        io_binding.bind_output(
            name=self._dfl_xseg_out_name, device_type='cuda', device_id=0,
            element_type=self._dfl_xseg_out_dtype, shape=out_shape,
            buffer_ptr=outpred.data_ptr(),
        )

        self.syncvec.cpu()
        self.dfl_xseg_model.run_with_iobinding(io_binding)

        return outpred.float().reshape(res, res)


    def detect_retinaface(self, img, max_num, score, input_size=640):
        with nvtx_range(f"detect_retinaface[s={input_size}]"):
            return self._detect_retinaface_inner(img, max_num, score, input_size)

    def _detect_retinaface_inner(self, img, max_num, score, input_size=640):
        # 把图缩放到 input_size 之内。模型是全卷积的，所以任何 32 的
        # 倍数边长都接受；输入越小 → 计算量线性越少 → 检测越快，代价是
        # 小脸的召回率下降。
        #
        # TRT 激活时把请求夹到配置好的 TRT-EP profile 范围（320-640）内。
        # session 在 _create_retinaface_session 里配的就是那个显式
        # profile，超出范围的请求会在 ORT 内部直接报错。
        with nvtx_range("rf_preprocess"):
            if self._retinaface_uses_trt and hasattr(self, '_retinaface_trt_input_max'):
                lo = self._retinaface_trt_input_min
                hi = self._retinaface_trt_input_max
                req = int(input_size)
                if req < lo or req > hi:
                    if not self._retinaface_engine_size_warned:
                        print(
                            '[Models] retinaface TRT-EP profile accepts '
                            f'sizes {lo}-{hi}; requested {req}. Clamping to '
                            f'{hi}.'
                        )
                        self._retinaface_engine_size_warned = True
                    req = max(lo, min(hi, req))
                input_size = req
            input_size = (int(input_size), int(input_size))
            # 纯 Python 浮点运算——此前在 int 参数上用 torch.div 会为了
            # 一次标量除法把两边都包成 CUDA tensor，付出 tensor 分配的
            # 开销，还通过 `.numpy()` 强制了一次下游同步。这些都不必要；
            # 它们只是标量。
            im_ratio = float(img.size()[1]) / float(img.size()[2])

            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio>model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / float(img.size()[1])

            # 在 GPU 上把输入缩放到 (3, new_h, new_w) uint8。这个 kernel 是
            # 预处理的主要开销；下面链条的其余部分都只是对缩放后 tile 的
            # 廉价算术。
            img_resized = v2.functional.resize(
                img, [new_height, new_width], antialias=True,
            )

            # 预分配的 per-worker letterbox 缓冲 (1, 3, H, W) float32。
            # 跨调用复用它省掉了每帧 torch.zeros 的分配器开销；端到端保持
            # 规范的 (1, 3, H, W) shape，也免掉了旧路径在 io_binding
            # bind_input 调用前所需的 HWC↔CHW permute + .contiguous() 往返。
            det_img = self._get_retinaface_input_buffer(input_size[0])
            # 清零上一帧的 letterbox 填充。在缓存缓冲上 zero_() 只是一个
            # 小 kernel；只清那几条填充带反而需要更多次 launch，不如整体
            # 清一遍。
            det_img.zero_()
            # BGR 交换 + uint8→float + 归一化 + 分片写入融合进一条链式
            # 表达式。img_resized[[2, 1, 0]] 是 BGR 重索引（高级索引会返回
            # 连续副本）；.float() 落地一个 float32 缓冲；.sub_()/.div_()
            # 在那个新缓冲上原地做（无额外分配）；.copy_() 把结果写进
            # det_img 的 letterbox 槽位。合计: 一次 gather + 一次类型转换
            # + 两次原地操作 + 一次 copy_，而旧路径是六步链（zero、slice、
            # BGR、sub、div、contiguous）。
            det_img[0, :, :new_height, :new_width].copy_(
                img_resized[[2, 1, 0]].float().sub_(127.5).div_(128.0)
            )

        with nvtx_range("rf_ort_run"):
            # det_10g 有 9 个输出——3 个 score、3 个 bbox delta、3 个 kps
            # delta，分别在 stride 8/16/32 上，按这个顺序声明（各导出的
            # 名字不同）。下游循环按索引消费它们。

            # 在 GPU 上预分配输出缓冲。shape 由 input_size + stride 完全
            # 确定（det_10g 每格 2 个 anchor）。把它们作为 buffer_ptr 绑定，
            # ORT 就能直接写进我们的 torch tensor，于是整个后处理都留在 GPU
            # ——没有 copy_outputs_to_cpu，NMS 前也没有主机往返。
            input_height = det_img.shape[2]
            input_width = det_img.shape[3]
            strides = (8, 16, 32)
            if self._reuse_detect_buffers:
                # 阶段 6：命中 per-worker 缓存，热路径上不再有每帧 9 次
                # torch.empty。缓冲顺序是 score×N → bbox×N → kps×N，与下面
                # det_out_names 的三个切片一致；N 取 len(strides) 而非写死 3。
                buffers = self._get_retinaface_output_buffers(
                    input_height, input_width, strides,
                )
                n = len(strides)
                score_bufs = list(buffers[0:n])
                bbox_bufs = list(buffers[n:2 * n])
                kps_bufs = list(buffers[2 * n:3 * n])
            else:
                # 默认路径：每帧各自分配 9 个输出缓冲，逻辑与改造前逐字相同
                # （只多了外层 if 的缩进）。
                score_bufs = []
                bbox_bufs = []
                kps_bufs = []
                for s in strides:
                    n_anchors = (input_height // s) * (input_width // s) * 2
                    score_bufs.append(torch.empty((n_anchors, 1), dtype=torch.float32, device='cuda'))
                    bbox_bufs.append(torch.empty((n_anchors, 4), dtype=torch.float32, device='cuda'))
                    kps_bufs.append(torch.empty((n_anchors, 10), dtype=torch.float32, device='cuda'))

            retinaface_session = self._get_retinaface_session()
            det_img_name, det_out_names = self._retinaface_io_names
            io_binding = retinaface_session.io_binding()
            io_binding.bind_input(
                name=det_img_name, device_type='cuda', device_id=0,
                element_type=np.float32, shape=det_img.size(),
                buffer_ptr=det_img.data_ptr(),
            )
            for i, name in enumerate(det_out_names[:3]):
                buf = score_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
            for i, name in enumerate(det_out_names[3:6]):
                buf = bbox_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )
            for i, name in enumerate(det_out_names[6:9]):
                buf = kps_bufs[i]
                io_binding.bind_output(
                    name=name, device_type='cuda', device_id=0,
                    element_type=np.float32, shape=tuple(buf.shape),
                    buffer_ptr=buf.data_ptr(),
                )

            # 有条件的 torch-stream drain。当 session 是用指向 worker 当前
            # torch stream 的 user_compute_stream 构建时（Per-Thread +
            # worker stream）跳过: 同一条 stream 意味着隐式顺序。
            if self._should_drain_syncvec():
                self.syncvec.cpu()
            retinaface_session.run_with_iobinding(io_binding)

        with nvtx_range("rf_postprocess"):
            # ===== GPU 侧后处理 =====
            # 从这里开始全部在 torch 里、worker 的 CUDA stream 上运行——
            # bbox 解码、kps 解码、score 阈值、跨 stride 拼接、排序、
            # 尺度还原、NMS。CUDA kernel 执行期间 GIL 是释放的，所以其他
            # worker 不会在 detect_retinaface 尾段的大头上排队等它
            # （那是 2026-05-21 nsys trace 里 "User Request" GIL 等待的
            # 主要来源）。
            scores_list = []
            bboxes_list = []
            kpss_list = []
            for idx, stride in enumerate(strides):
                scores_t = score_bufs[idx].view(-1)              # (N,)
                bbox_preds = bbox_bufs[idx] * stride             # (N, 4)
                kps_preds = kps_bufs[idx] * stride               # (N, 10)
                height = input_height // stride
                width = input_width // stride
                anchor_centers = self._get_retinaface_gpu_anchors(
                    height, width, stride,
                )                                                # (N, 2)

                pos_mask = scores_t >= score

                # Bbox 解码: anchor ± delta。符号模式 [-x1, -y1, +x2, +y2]
                # 对应 det_10g 的 FCOS 式人脸回归头。
                ax = anchor_centers[:, 0]
                ay = anchor_centers[:, 1]
                bboxes = torch.stack(
                    [
                        ax - bbox_preds[:, 0],
                        ay - bbox_preds[:, 1],
                        ax + bbox_preds[:, 2],
                        ay + bbox_preds[:, 3],
                    ],
                    dim=-1,
                )                                                # (N, 4)

                # KPS 解码: 每个 anchor 5 个 (x, y) 对，共用同一个 anchor
                # 偏移。numpy 原版是逐对循环；向量化后就是
                # `view + broadcast-add`。
                kpss = kps_preds.view(-1, 5, 2) + anchor_centers.unsqueeze(1)

                scores_list.append(scores_t[pos_mask])
                bboxes_list.append(bboxes[pos_mask])
                kpss_list.append(kpss[pos_mask])

            all_scores = torch.cat(scores_list)
            all_bboxes = torch.cat(bboxes_list)
            all_kpss = torch.cat(kpss_list)

            if all_scores.numel() == 0:
                return np.zeros((0, 5, 2), dtype=np.float32)

            # 按 score 降序排（与旧代码里的 np.argsort()[::-1] 一致），然后
            # 在 IoU 空间做 NMS。除以 det_scale 把坐标从 letterbox 后的
            # 模型空间还原回原图坐标系。
            order = torch.argsort(all_scores, descending=True)
            all_scores = all_scores[order]
            all_bboxes = all_bboxes[order] / det_scale
            all_kpss = all_kpss[order] / det_scale

            keep = torchvision.ops.nms(all_bboxes, all_scores, iou_threshold=0.4)

            # max_num 截断，权重是"面积 - 中心距"，与旧启发式一致。保留了
            # 原实现的一个怪癖: 把 det_img.shape[0]/[1]（即 (1, 3, H, W)
            # tensor 的 batch 与 channel 维——值为 0 和 1）当作"图像中心"；
            # 中心距项相对面积很小，实际差别可以忽略。
            if max_num > 0 and keep.shape[0] > max_num:
                kept_bboxes = all_bboxes[keep]
                area = (kept_bboxes[:, 2] - kept_bboxes[:, 0]) * (
                    kept_bboxes[:, 3] - kept_bboxes[:, 1]
                )
                cy = float(det_img.shape[0] // 2)
                cx = float(det_img.shape[1] // 2)
                dx = (kept_bboxes[:, 0] + kept_bboxes[:, 2]) / 2 - cy
                dy = (kept_bboxes[:, 1] + kept_bboxes[:, 3]) / 2 - cx
                offset_dist_squared = dx * dx + dy * dy
                values = area - offset_dist_squared * 2.0
                bindex = torch.argsort(values, descending=True)[:max_num]
                keep = keep[bindex]

            # 最末尾唯一一次设备边界。在此之前的一切都只是排在 worker 的
            # CUDA stream 上。
            return all_kpss[keep].cpu().numpy()

    def detect_scrdf(self, img, max_num, score, input_size=640):
        # 把图缩放到 input_size 之内。模型是全卷积的，所以任何 32 的
        # 倍数边长都接受；输入越小 → 计算量线性越少 → 检测越快，代价是
        # 小脸的召回率下降。
        input_size = (int(input_size), int(input_size))
        # 纯 Python 浮点运算——此前在 int 参数上用 torch.div 会为了
        # 一次标量除法把两边都包成 CUDA tensor，付出 tensor 分配的
        # 开销，还通过 `.numpy()` 强制了一次下游同步。这些都不必要；
        # 它们只是标量。
        im_ratio = float(img.size()[1]) / float(img.size()[2])

        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / float(img.size()[1])

        # 用函数式而不是 transform 对象，避免每次调用都新建一个 v2.Resize。
        img = v2.functional.resize(
            img, [new_height, new_width], antialias=True,
        )
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # 转成 BGR 并归一化
        det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128
        
        # 准备数据并查找模型参数
        det_img = torch.unsqueeze(det_img, 0).contiguous()
        input_name = self.scrdf_model.get_inputs()[0].name
        
        outputs = self.scrdf_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        
        io_binding = self.scrdf_model.io_binding() 
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=det_img.size(), buffer_ptr=det_img.data_ptr())
        
        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda') 
        
        # 同步并运行模型
        syncvec = self.syncvec.cpu()        
        self.scrdf_model.run_with_iobinding(io_binding)
        
        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]
        
        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                if len(center_cache)<100:
                    center_cache[key] = anchor_centers
            
            pos_inds = np.where(scores>=score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)  
            
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i%2] + kps_preds[:, i]
                py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1) 
            #kpss = kps_preds
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        # det_scale 现在是 Python float（过去是 torch tensor，要在这里
        # 强制过一遍 .numpy()，那是个隐藏的同步点）。
        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        # torchvision.ops.nms（CUDA）: +1 面积约定的位精确性说明见
        # detect_retinaface。
        thresh = 0.4
        if pre_det.shape[0] == 0:
            keep = []
        else:
            boxes_t = torch.from_numpy(pre_det[:, :4]).contiguous().cuda()
            scores_t = torch.from_numpy(pre_det[:, 4]).contiguous().cuda()
            keep_t = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=thresh)
            keep = keep_t.cpu().tolist()

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return kpss

    def recognize(self, img, face_kps, dim):
        with nvtx_range("rec_preprocess"):
            # 求变换
            dst = self.arcface_dst.copy() * dim

            tform = trans.SimilarityTransform.from_estimate(face_kps, dst)

            # 施加变换
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, dim*112, dim*112)

            # 转成 BGR 并归一化
            img = img.permute(1,2,0) #112,112,3
            cropped_image = img
            img = img[:, :, [2,1,0]]
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5) #dim*112, dim*112, 3

        with nvtx_range("rec_ort_run"):
            temp_holder = []
            recognition_session = self._get_recognition_session()
            output = torch.empty((1, 512), dtype=torch.float32, device='cuda').contiguous()
            io_binding = recognition_session.io_binding()
            io_binding.bind_output(
                name=recognition_session.get_outputs()[0].name, device_type='cuda', device_id=0,
                element_type=np.float32, shape=(1, 512),
                buffer_ptr=output.data_ptr(),
            )

            drain = self._should_drain_syncvec()
            for j in range(dim):
                for i in range(dim):
                    input = img[j::dim, i::dim].permute(2, 0, 1)
                    input = torch.unsqueeze(input, 0).contiguous()
                    io_binding.bind_input(
                        name=recognition_session.get_inputs()[0].name, device_type='cuda', device_id=0,
                        element_type=np.float32, shape=(1, 3, 112, 112),
                        buffer_ptr=input.data_ptr(),
                    )
                    if drain:
                        self.syncvec.cpu()
                    recognition_session.run_with_iobinding(io_binding)
                    temp_holder.append(torch.squeeze(output).cpu().numpy())

            # 返回 embedding
            out = np.mean(temp_holder, 0)
            return out, cropped_image
        
    def resnet50(self, image, score=.5):   
        if not self.resnet50_model:
            self.resnet50_model = onnxruntime.InferenceSession(self._mp("res50.onnx"), providers=self.providers)
            
            feature_maps = [[64, 64], [32, 32], [16, 16]]
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            image_size = 512
            
            for k, f in enumerate(feature_maps):
                min_size_array = min_sizes[k]
                for i, j in product(range(f[0]), range(f[1])):
                    for min_size in min_size_array:
                        s_kx = min_size / image_size
                        s_ky = min_size / image_size
                        dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                        dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            self.anchors += [cx, cy, s_kx, s_ky]   

        # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.permute(1,2,0)
        
        # image = image - [104, 117, 123]
        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device='cuda')
        image = torch.sub(image, mean)
        
        # image = image.transpose(2, 0, 1)
        # image = np.float32(image[np.newaxis,:,:,:])
        image = image.permute(2,0,1)
        image = torch.unsqueeze(v2.Resize((512, 512), antialias=False)(image), 0)

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device='cuda')
        
        # ort_inputs = {"input": image}        
        conf = torch.empty((1,10752,2), dtype=torch.float32, device='cuda').contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device='cuda').contiguous()

        io_binding = self.resnet50_model.io_binding() 
        io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type='cuda', device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())
        
        # _, conf, landmarks = self.resnet_model.run(None, ort_inputs)        
        torch.cuda.synchronize('cuda')
        self.resnet50_model.run_with_iobinding(io_binding)        
        

        # conf = torch.from_numpy(conf)
        # scores = conf.squeeze(0).numpy()[:, 1]
        scores = torch.squeeze(conf)[:, 1]
        
        # landmarks = torch.from_numpy(landmarks)
        # landmarks = landmarks.to('cuda')        

        priors = torch.tensor(self.anchors).view(-1, 4)
        priors = priors.to('cuda')

        # pre = landmarks.squeeze(0) 
        pre = torch.squeeze(landmarks, 0)
        
        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        # landmarks = landmarks * scale1
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()  

        # ignore low scores
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()  
        scores = scores.cpu().numpy()  
        
        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        landmarks = landmarks[order][0]

        return np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])