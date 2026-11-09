"""0 检出帧的整帧朝向兜底（只认 90 度倍数）。

stdlib-only：``VideoManager`` 需要 torch / OpenCV 才能 import，而「这一帧
要不要试旋转、试出哪个角度」是纯逻辑，可以脱离 GPU 栈做行为测试。

背景：手机竖拍、横竖颠倒的视频里，RetinaFace 对横躺脸召回显著下降。
GUI 时代的 ``OrientAutoSwitch`` 靠跨帧状态 + 眼轴倾角 EMA 处理，但：

1. ``detected_orient_angle`` / ``orient_fine_angle`` / ``orient_miss_streak``
   是 VideoManager 实例属性，而 headless 的多个 worker 共享同一个实例，
   ``workers > 1`` 下会竞态（README 早已写明不要同时开）。
2. EMA 产生的亚 90 度角度配上 ``rotate(..., expand=True)`` 的正、逆两次
   调用会让画布越滚越大，输出帧尺寸逐帧漂移；编码器却把输出流宽高写死
   成输入尺寸。
3. 它默认关闭且没有 CLI 开关，只能靠 ``--params saved_parameters.json``。

这里改成**逐帧无状态**的兜底：只在第一遍 0 检出时才试 90/180/270，命中
后把整帧转正重新检测，检测、识别、换脸都在旋转域里跑完，最后逆旋回原
尺寸。正常视频第一遍就检出，零额外推理、零旋转，输出逐像素不变。

只用 90 的倍数，因此 ``expand=True`` 的 H×W → W×H → H×W 严格可逆；不做
亚 90 度微调（歪头交给 ``Models.recognize`` 的 5 点相似变换）；无跨帧
状态，2N 乱序窗口 / ``workers > 1`` 下安全。

显式传 ``--probe-angles`` 时改走另一条路径：img 全程不动，只把旋转副本
喂给 detector，5 点 kps 经 ``map_kps_back`` 回到原帧坐标再交给换脸链路，
因此 45/22.5 度也不会让输出画布越滚越大。不传该参数时上面的旧路径
逐字保留，保证 workers=1 下逐像素回归。
"""

from __future__ import annotations

import math as _math

# 0 度已经由第一遍检测覆盖，不重复推理。
PROBE_ANGLES: tuple[int, ...] = (90, 180, 270)


def should_probe(
    *,
    detected: int,
    enabled: bool,
    already_rotated: bool = False,
) -> bool:
    """是否在这一帧做 4 向朝向探测。

    - ``detected > 0``：正常帧，零额外推理（默认路径逐像素不变的关键）。
    - ``enabled=False``（``--no-auto-orient``）：回滚到旧行为。
    - ``already_rotated``：GUI 时代的 ``OrientAutoSwitch`` / ``OrientSwitch``
      已经在检测前转过一次，不再叠加第二次旋转。
    """
    return bool(enabled) and int(detected) == 0 and not already_rotated


def choose_angle(counts: dict[int, int], angles=None) -> int:
    """从各角度的检出脸数里挑胜出角度，``0`` = 不旋转。

    平手或全部为 0 一律取 0——证据不足时宁可保持原帧。默认只认
    ``PROBE_ANGLES`` 里的角度，其它 key 静默忽略（防止调用方把 0 度或
    任意角度混进来）。严格 ``>`` 才更新，所以平手时取最小角度。

    ``angles`` 供 ``--probe-angles`` 的 8/16 向路径显式传入自己的候选轴；
    不传时行为与旧版逐字一致（现有离线断言依赖这一点）。
    """
    axes = PROBE_ANGLES if angles is None else angles
    best_angle = 0
    best_count = 0
    for angle in axes:
        count = int(counts.get(angle, 0))
        if count > best_count:
            best_count = count
            best_angle = angle
    return best_angle


# ===== --probe-angles：只旋转检测副本，不旋转用于换脸的原帧 =============
#
# 旧路径把整帧转正、在旋转域里换脸、末尾再逆旋回来，只有 90 倍数可逆。
# 新路径（显式传 --probe-angles 才走）让 img 全程不动，只生成一份旋转
# 副本喂给 detector，再把 5 点 kps 逆映射回原帧坐标。只要 kps 与传入
# swap_core 的 img 同坐标系，SimilarityTransform / paste_back 都不需要
# 知道图被转过，输出尺寸天然等于输入尺寸。
#
# 关键取舍：旋转副本不用 torchvision 的 rotate(expand=True)，而是用自己
# 构造的仿射 A 配 affine_grid + grid_sample 生成。这样 A 与它的逆完全由
# 本模块定义，不依赖私有 API，也不需要在 GPU 上「现场标定 A」——那是
# 原方案里唯一没法离线验证的环节。纯几何部分在 tests/test_probe_angles.py
# 里逐角度做往返自校验。
#
# 术语沿用 CONTEXT.md：中文行文，领域术语保留英文原词。


def parse_probe_angles(spec):
    """把 ``--probe-angles`` 的步长字符串解析成非零探测轴（升序）。

    ``None`` / 空串 -> ``None``，表示走旧的整帧旋转路径（逐像素行为不变）。
    ``"45"`` -> 8 向，``"22.5"`` -> 16 向；0 度由第一遍检测覆盖，不重复推理。
    步长必须整除 360 且落在 (0, 180]，否则抛 ``ValueError``。
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    try:
        step = float(text)
    except ValueError as exc:
        raise ValueError(f"probe-angles 必须是角度步长，得到 {spec!r}") from exc
    if not (0.0 < step <= 180.0) or not _math.isfinite(step):
        raise ValueError(f"probe-angles 步长须在 (0, 180]，得到 {step!r}")
    n = 360.0 / step
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"probe-angles 步长须整除 360，得到 {step!r}")
    n = int(round(n))
    return tuple(round(k * step, 6) for k in range(1, n))


def matmul3(a, b):
    """3x3 矩阵乘，纯 stdlib（不引 numpy，保持本模块可离线测试）。"""
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def rotation_affine(angle, src_hw):
    """「原帧像素坐标 -> 旋转画布像素坐标」的 3x3 仿射 A。

    返回 ``(A, (H_out, W_out))``。旋转方向是视觉逆时针（图像坐标 y 向下），
    画布取旋转后四角的外接矩形（等价于 expand=True），两个中心对齐。
    因为我们自己定义 A，它的逆就是 kps 的回程，不需要猜 torchvision 的
    符号、中心点或 expand 平移量。
    """
    h, w = float(src_hw[0]), float(src_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid src_hw {src_hw!r}")
    rad = _math.radians(float(angle))
    c, s = _math.cos(rad), _math.sin(rad)
    out_w = int(_math.ceil(abs(c) * w + abs(s) * h - 1e-9))
    out_h = int(_math.ceil(abs(s) * w + abs(c) * h - 1e-9))
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    ncx, ncy = (out_w - 1.0) / 2.0, (out_h - 1.0) / 2.0
    a = (
        (c, s, ncx - c * cx - s * cy),
        (-s, c, ncy + s * cx - c * cy),
        (0.0, 0.0, 1.0),
    )
    return a, (out_h, out_w)


def invert_affine(a):
    """3x3 仿射求逆（纯旋转 + 平移，det 恒为 1，数值稳定）。"""
    (m00, m01, m02), (m10, m11, m12), _ = a
    det = m00 * m11 - m01 * m10
    if abs(det) < 1e-12:
        raise ValueError("singular affine matrix")
    i00, i01 = m11 / det, -m01 / det
    i10, i11 = -m10 / det, m00 / det
    return (
        (i00, i01, -(i00 * m02 + i01 * m12)),
        (i10, i11, -(i10 * m02 + i11 * m12)),
        (0.0, 0.0, 1.0),
    )


def apply_affine(a, x, y):
    """3x3 仿射作用到单个 (x, y)。"""
    return (
        a[0][0] * x + a[0][1] * y + a[0][2],
        a[1][0] * x + a[1][1] * y + a[1][2],
    )


def map_kps_back(kpss, inv_a):
    """把旋转画布坐标的 ``(N,5,2)`` kps 逆映射回原帧坐标。

    接受 numpy array 或嵌套 list，返回 list of list of ``(x, y)``，由调用方
    用 ``np.asarray(..., dtype=np.float32)`` 转回。刻意不 import numpy，
    这样几何往返能在本机（无 GPU、无 numpy）做离线测试。
    """
    out = []
    for face in kpss:
        out.append([apply_affine(inv_a, float(x), float(y)) for x, y in face])
    return out


def normalized_grid_matrix(inv_a, src_hw, dst_hw):
    """把「输出画布像素 -> 输入原帧像素」的 inv(A) 转成 grid_sample 的 2x3。

    ``affine_grid(theta, align_corners=False)`` 的 theta 作用在归一化输出
    坐标上，产出归一化输入坐标。这里把像素空间矩阵夹在两头的归一化变换
    之间：``M = Q_in @ inv(A) @ P_out``，其中 P_out 把归一化输出还原成
    输出像素，Q_in 把输入像素换成归一化输入。
    """
    h_in, w_in = float(src_hw[0]), float(src_hw[1])
    h_out, w_out = float(dst_hw[0]), float(dst_hw[1])
    p_out = (
        (w_out / 2.0, 0.0, w_out / 2.0 - 0.5),
        (0.0, h_out / 2.0, h_out / 2.0 - 0.5),
        (0.0, 0.0, 1.0),
    )
    q_in = (
        (2.0 / w_in, 0.0, 1.0 / w_in - 1.0),
        (0.0, 2.0 / h_in, 1.0 / h_in - 1.0),
        (0.0, 0.0, 1.0),
    )
    m = matmul3(q_in, matmul3(inv_a, p_out))
    return ((m[0][0], m[0][1], m[0][2]), (m[1][0], m[1][1], m[1][2]))


# ===== 目标人物未匹配帧的受控补探测 + 跨角度合并 =========================
#
# 原方案 S5 用常数 K 放宽到 detected>0 的所有帧，单人脸视频会近逐帧跑
# 7 次额外检测（成本 ×8）。这里收窄成：只有指定了 --target-face、本帧
# 已经有脸、却没有任何一张匹配上目标人物时，才补做多向探测。补回的脸
# 只追加不替换，0 度第一遍检出的 kps 永远优先保留。
#
# 合并去重刻意放在 stdlib-only 模块里，阈值与仲裁策略都能离线做行为测试。

# 群像镜头不补探测：帧里脸数超过这个上限时，多向检测的性价比太低。
PROBE_ON_MISS_MAX_FACES = 3


def should_probe_on_miss(
    *,
    enabled: bool,
    has_targets: bool,
    detected: int,
    matched_any: bool,
    already_rotated: bool = False,
    max_faces: int = PROBE_ON_MISS_MAX_FACES,
) -> bool:
    """是否要为「目标人物本帧未匹配上」补做多向探测。

    - ``detected == 0``：0 检出帧由 ``should_probe`` 的朝向兜底负责，
      不在这里重复跑两套探测。
    - ``matched_any``：目标人物已经匹配上了，零额外推理（正常帧逐像素不变）。
    - ``detected > max_faces``：群像镜头跳过，避免近逐帧 ×N 检测。
    - ``already_rotated``：GUI 的 OrientAutoSwitch / OrientSwitch 或旧的
      整帧旋转兜底已把 img 转过，kps 不在原帧坐标域，补探测会贴歪。
    """
    n = int(detected)
    return bool(
        enabled
        and has_targets
        and n > 0
        and n <= int(max_faces)
        and not matched_any
        and not already_rotated
    )


def kps_rms(a, b) -> float:
    """两组 5 点的 RMS 距离（像素）；长度不一致时按较短的那组算。"""
    pairs = list(zip(a, b))
    if not pairs:
        return float("inf")
    total = 0.0
    for (x1, y1), (x2, y2) in pairs:
        total += (float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2
    return _math.sqrt(total / (2.0 * len(pairs)))


def kps_scale(kps) -> float:
    """5 点包围盒对角线，用来把去重阈值归一化到脸的尺度。"""
    pts = list(kps)
    if not pts:
        return 0.0
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return _math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def merge_probe_landmarks(base_kpss, candidates, rms_ratio=0.05,
                          max_faces=10):
    """跨角度去重，返回**只属于新增脸**的 kps 列表（原帧坐标）。

    ``base_kpss`` 是第一遍 0 度检出的 5 点，无条件保留——它没经过插值，
    kps 最准；45 度副本里脸被转正、confidence 可能更高，但经历 BILINEAR
    插值 + 画布膨胀，5 点回归误差更大，不该覆盖它。

    ``candidates`` 是 ``[(angle, kpss(N,5,2), scores(N,))]``。按 score
    降序展平后逐个与已保留簇比 5 点 RMS；阈值是参考簇尺度的
    ``rms_ratio``（默认脸宽的 5%，需真机标定）。判为同一张脸就丢弃，
    避免它被 ``swap_core`` 换两次（第二次会在已换过的像素上再换）。

    返回新增部分，调用方只需为它们补算 embedding 再追加到 ret。
    """
    kept = [list(face) for face in base_kpss]
    kept_scale = [kps_scale(face) for face in kept]
    base_count = len(kept)
    if max_faces <= base_count:
        return []

    flat = []
    for angle, kpss, scores in candidates:
        for i, face in enumerate(kpss):
            try:
                score = float(scores[i]) if scores is not None else 0.0
            except (IndexError, TypeError, ValueError):
                score = 0.0
            flat.append((score, float(angle), list(face)))
    # score 降序；平手时小角度优先（90 倍数副本是像素重排，无插值损失）。
    flat.sort(key=lambda item: (-item[0], item[1]))

    for _score, _angle, face in flat:
        if len(kept) >= max_faces:
            break
        scale = kps_scale(face)
        duplicated = any(
            kps_rms(face, ref) <= rms_ratio * max(ref_scale, 1.0)
            for ref, ref_scale in zip(kept, kept_scale)
        )
        if duplicated:
            continue
        kept.append(face)
        kept_scale.append(scale)
    return kept[base_count:]
