"""RetinaFace 输出缓冲的 shape 计算与 per-worker 复用（stdlib-only）。

``Models._detect_retinaface_inner`` 每帧为 9 个 io_binding 输出各做一次
``torch.empty``。这些 shape 只由 ``(input_height, input_width, stride)`` 决定，
而前两个尺寸在一次 video 里恒定（除非改 ``DetectInputSizeTextSel``）。输入
缓冲（``_get_retinaface_input_buffer``）和 anchor centers 早已按 worker 缓存，
输出缓冲是这条链上唯一还每帧重新分配的一环。

这里把「算 shape → 按 key 命中缓存 → 未命中才 alloc」抽成不依赖 torch 的纯
逻辑：``alloc`` 由调用方注入，于是缓存行为能用假 alloc 离线做行为测试。
torch 侧的绑定仍在 ``Models.py``；本模块不 import 任何 GPU 栈。

术语沿用 CONTEXT.md：中文行文，领域术语保留英文原词。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

# det_10g 在三个 stride 上各输出 score / bbox / kps 三组。
STRIDES: tuple[int, ...] = (8, 16, 32)
KINDS: tuple[str, ...] = ("score", "bbox", "kps")
BUFFER_WIDTHS: dict[str, int] = {"score": 1, "bbox": 4, "kps": 10}


@dataclass(frozen=True)
class BufferSpec:
    """一个 io_binding 输出缓冲的描述。"""

    kind: str
    stride: int
    shape: tuple[int, int]


def output_buffer_specs(
    input_height: int,
    input_width: int,
    strides: Iterable[int] = STRIDES,
) -> tuple[BufferSpec, ...]:
    """按 ``bind_output`` 的消费顺序给出 9 个输出的 shape。

    顺序刻意是 score×3 → bbox×3 → kps×3，与
    ``Models._detect_retinaface_inner`` 里 ``det_out_names`` 的三个切片一致，
    调用方可以直接用 ``[0:3] / [3:6] / [6:9]`` 取回三组缓冲。

    ``n_anchors`` 的算法必须与该函数的逐帧路径逐字相同：每格 2 个 anchor，
    feature map 边长是 ``input_size // stride``（整除，不是向上取整）。
    """
    h, w = int(input_height), int(input_width)
    if h <= 0 or w <= 0:
        raise ValueError(f"detector input size must be positive, got {(h, w)!r}")
    st = tuple(int(s) for s in strides)
    if not st or any(s <= 0 for s in st):
        raise ValueError(f"strides must be positive and non-empty, got {st!r}")

    specs: list[BufferSpec] = []
    for kind in KINDS:
        width = BUFFER_WIDTHS[kind]
        for stride in st:
            n_anchors = (h // stride) * (w // stride) * 2
            specs.append(BufferSpec(kind, stride, (n_anchors, width)))
    return tuple(specs)


def resolve_reuse_buffers(enabled: bool, backend: str = "cuda") -> bool:
    """是否启用输出缓冲复用。

    只接受显式开启、且走 ONNX/CUDA EP 的路径。TRT-EP 的 io_binding 输出行为
    未经真机核对，这里不猜、不静默套用（与 ``decoder.py`` 探测失败不静默
    回退是同一条原则）。未知 backend 一律关闭。
    """
    return bool(enabled) and backend == "cuda"


class OutputBufferCache:
    """按 ``(h, w, strides)`` 缓存一组输出缓冲，alloc 由调用方注入。

    每个 worker 必须各自持有一个实例（挂在 ``threading.local`` 上）：不同
    worker 的 stream 并发推进，共享同一组缓冲会让 ORT 的写入互相踩。同一
    worker 内则是安全的——``rf_postprocess`` 在 worker 自己的 stream 上读完
    这一帧，下一帧的 ORT 写入才排到它后面。

    ``alloc(shape, spec)`` 返回缓冲对象；生产代码传的是
    ``torch.empty(shape, dtype=torch.float32, device='cuda')`` 的包装，测试
    传假对象即可验证「同 key 零分配、key 变才重建」。
    """

    def __init__(self, alloc: Callable[[tuple[int, int], BufferSpec], object]):
        self._alloc = alloc
        self._key: tuple | None = None
        self._buffers: tuple = ()

    @property
    def key(self) -> tuple | None:
        return self._key

    def get(
        self,
        input_height: int,
        input_width: int,
        strides: Iterable[int] = STRIDES,
    ) -> tuple:
        # 先物化 strides（可能是 generator），再算 specs 顺带校验入参；
        # 非法尺寸不应该把缓存写脏。
        st = tuple(int(s) for s in strides)
        specs = output_buffer_specs(input_height, input_width, st)
        key = (int(input_height), int(input_width), st)
        if key != self._key:
            self._buffers = tuple(
                self._alloc(spec.shape, spec) for spec in specs
            )
            self._key = key
        return self._buffers
