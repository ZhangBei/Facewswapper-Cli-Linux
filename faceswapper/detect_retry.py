"""漏检帧的检测重试计划。

stdlib-only：``VideoManager`` 需要 torch / OpenCV 才能 import，而「这一帧
要不要重试、用什么参数重试」是纯逻辑，可以脱离 GPU 栈做行为测试。

背景：``run_detect`` 返回空 ``kpss`` 的帧直接透传（不换脸）。侧脸/小脸
漏检常是「这个输入尺寸下过不了置信度」，升一档输入尺寸就能检出。只在
**0 检出**的帧上追加推理——已有检出的帧绝不重试，正常路径行为逐字不变，
开销为零（一次布尔判断）。

防误检的两道闸：
1. 重试轮用更高的分数阈值（+0.2，即 20 分）：背景误检通常是低置信度，
   第一遍用用户阈值没检出的帧，第二遍用更高阈值检出，误检增量严格小于
   正检。
2. GUI 时代的 ``OrientAutoSwitch`` 开启时不重试——它的
   ``orient_miss_streak`` 会在连续漏检后做 4 向重探，双开会在同一帧跑
   两套试探。注意这与默认开启、逐帧无状态的整帧朝向兜底
   （``orientation.py``）是两回事：兜底排在重试之前，命中后帧上已有脸，
   本函数根本不会被调用；兜底也没探到脸时，仍会轮到这里的升档重试。

``--detect-retry-quality``（默认关闭）把触发从「0 检出」放宽到「检出了
但明显不可靠」：用本帧最高 confidence 与最差 5 点构型质量做门控，仍用
升档 + 更高阈值，重检结果只追加不替换。这修的是侧脸方案里「一张正脸 +
一张侧脸」的混合帧——旧逻辑 ``detected > 0`` 直接放过。质量代理刻意做成
旋转不变，避免把 roll 歪头误判成坏 kps（``frontalness`` 用水平 x 偏移，
有这个污染）。不传该开关时行为逐字不变。
"""

from __future__ import annotations

import math as _math
from typing import Any


# 重试轮在用户阈值上抬升的分数（0-100 刻度）。
RETRY_SCORE_BUMP = 20.0
# 重试用的检测输入尺寸。要求是 32 的倍数（检测器动态输入约束），取
# DetectInputSizeTextSelModes 的最大档。
RETRY_INPUT_SIZE = 960

# --detect-retry-quality 的默认门控。两个值都需要真机标定：太松会把
# 正常帧也重检（成本从稀有变常见），太紧救不到混合帧。
DEFAULT_RETRY_MIN_CONF = 0.60
DEFAULT_RETRY_MIN_KPS_QUALITY = 0.50
DEFAULT_RETRY_QUALITY_SPEC = "0.60:0.50"

# --align-for-swap 的默认启用侧度：只有 yaw_sideness >= 这个值才把 68 点
# 派生的 5 点交给 swap_core，正脸帧保持原 kps 以保住逐像素回归。需真机标定。
DEFAULT_ALIGN_FOR_SWAP_SIDENESS = 0.35

# 5 点构型的三个旋转不变量各自的「正常区间」与容差。
# (下限, 上限, 容差)，全部以眼距 L 归一化。
_KPS_BANDS = (
    (0.40, 1.30, 0.60),   # 鼻尖相对眼中点，沿脸轴的垂直投影
    (0.90, 2.10, 0.80),   # 嘴中点相对眼中点的垂直投影
    (0.45, 1.10, 0.40),   # 嘴宽 / 眼距
)


def parse_retry_quality(spec):
    """把 ``--detect-retry-quality`` 解析成 ``(min_conf, min_kps_quality)``。

    ``None`` / 空串 -> ``None``（关闭，旧行为）；``"0.60:0.50"`` -> 元组。
    两个量都必须落在 0-1，否则抛 ``ValueError``。
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(
            "--detect-retry-quality 须为 MIN_CONF:MIN_KPS_QUALITY，"
            f"得到 {spec!r}"
        )
    try:
        min_conf, min_quality = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"--detect-retry-quality 含非数值：{spec!r}"
        ) from exc
    for name, value in (("MIN_CONF", min_conf),
                        ("MIN_KPS_QUALITY", min_quality)):
        if not _math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} 须在 0-1，得到 {value!r}")
    return (min_conf, min_quality)


def _band_score(value: float, lo: float, hi: float, tol: float) -> float:
    if lo <= value <= hi:
        return 1.0
    d = lo - value if value < lo else value - hi
    return max(0.0, min(1.0, 1.0 - d / tol))


def local_frame(kps):
    """建立以两眼连线为 u 轴、垂直向下为 v 轴的局部坐标系。

    返回 ``(pts, ux, uy, vx, vy, L, mid_x, mid_y)``；退化输入（不足 5 点、
    眼距为 0、非有限值）返回 None。投影后除以眼距 L，对 roll 旋转与尺度
    都不变——这是 ``kps_quality`` 与 ``yaw_sideness`` 共同的地基。
    """
    try:
        pts = [(float(p[0]), float(p[1])) for p in kps]
        if len(pts) != 5 or any(not _math.isfinite(v) for p in pts for v in p):
            return None
        e_l, e_r = pts[0], pts[1]
        ex, ey = e_r[0] - e_l[0], e_r[1] - e_l[1]
        L = _math.hypot(ex, ey)
        if not _math.isfinite(L) or L < 1e-6:
            return None
        ux, uy = ex / L, ey / L
        return (pts, ux, uy, -uy, ux, L,
                (e_l[0] + e_r[0]) / 2.0, (e_l[1] + e_r[1]) / 2.0)
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def project(frame, p):
    """把原帧点投到局部坐标并除以眼距 L，返回 ``(u, v)``。"""
    _pts, ux, uy, vx, vy, L, mid_x, mid_y = frame
    dx, dy = p[0] - mid_x, p[1] - mid_y
    return ((dx * ux + dy * uy) / L, (dx * vx + dy * vy) / L)


def kps_quality(kps) -> float:
    """5 点构型质量 0-1，衡量「像不像一张几何自洽的脸」。

    刻意做成 **roll 旋转不变**（局部坐标系见 ``local_frame``），不会像
    ``frontalness``（只看水平 x 偏移）那样把歪头污染成「侧脸 / 坏 kps」。

    同时刻意**不惩罚 yaw**：鼻尖沿 u 的偏移是偏航信号，不是几何错误，
    拿它扣分会把每张侧脸都判成低质量、触发无意义的重检。侧脸嘴宽因
    透视压缩变小会轻微扣分，这是可接受的（侧脸 kps 确实更不可靠）。
    """
    frame = local_frame(kps)
    if frame is None:
        return 0.0
    pts = frame[0]
    _nose_u, nose_y = project(frame, pts[2])
    ml_x, ml_y = project(frame, pts[3])
    mr_x, mr_y = project(frame, pts[4])
    mouth_y = (ml_y + mr_y) / 2.0
    mouth_w = _math.hypot(mr_x - ml_x, mr_y - ml_y)

    values = (nose_y, mouth_y, mouth_w)
    scores = [_band_score(v, *band) for v, band in zip(values, _KPS_BANDS)]
    if any(not _math.isfinite(s) for s in scores):
        return 0.0
    return sum(scores) / len(scores)


def yaw_sideness(kps) -> float:
    """0 = 正脸，1 = 全侧；**roll 旋转不变**的偏航几何代理。

    鼻尖沿眼轴 u 的偏移 / 眼距，与 ``frontal_match.frontalness`` 的 0.6
    归一化同源，但投在局部坐标系里：歪 45° 的正脸仍判 0，不会被
    ``frontalness`` 那样误判成 0.58 侧度。仍是大角度饱和的线性代理
    （约 60° 后失去分辨率），只用于「要不要启用 68 点替换几何」这类
    粗门控，不当精确角度用。退化输入返回 0.0（证据不足时保持原 kps）。
    """
    frame = local_frame(kps)
    if frame is None:
        return 0.0
    nose_u, _nose_v = project(frame, frame[0][2])
    return max(0.0, min(1.0, abs(nose_u) / 0.6))


def align_kps_for_swap(face_kps, align_kps, min_sideness):
    """--align-for-swap 的纯决策：返回该交给 ``swap_core`` 的 5 点。

    三道门任一不过就保持检测器 5 点（正脸帧逐像素不变的关键）：

    1. 开关关闭（``min_sideness is None``）或没有 68 点结果；
    2. 侧度不够（旋转不变的 ``yaw_sideness``）；
    3. 68 点转出的 5 点几何不如原 5 点自洽——侧脸遮挡会让不可见的
       嘴角/眼被算歪，方案 P2 明确要求这道质量门。

    抽成 stdlib 纯函数是为了能离线验证门控，而不是靠读 GPU 栈源码。
    """
    if min_sideness is None or align_kps is None:
        return face_kps
    if yaw_sideness(face_kps) < float(min_sideness):
        return face_kps
    if kps_quality(align_kps) < kps_quality(face_kps):
        return face_kps
    return align_kps


def retry_plan(
    *,
    detected: int,
    input_size: Any,
    detect_score: float,
    auto_orient: bool = False,
    enabled: bool = False,
    quality_enabled: bool = False,
    max_conf: float | None = None,
    frame_kps_quality: float | None = None,
    min_conf: float = DEFAULT_RETRY_MIN_CONF,
    min_kps_quality: float = DEFAULT_RETRY_MIN_KPS_QUALITY,
) -> tuple[int, float] | None:
    """重试参数 ``(input_size, score)``；不该重试返回 None。

    - ``quality_enabled=False``（默认）：只在 0 检出帧重试，``detected>0``
      一律放过——与旧版逐字一致。
    - ``quality_enabled=True``：``detected>0`` 时，只有本帧最高 confidence
      低于 ``min_conf`` **或** 最差 5 点质量低于 ``min_kps_quality`` 才重试。
      两道防误检闸（更高阈值 +0.2、auto-orient 让位）仍然生效。
    - 输入尺寸已升到最大档（>= RETRY_INPUT_SIZE）时再升没有意义，放弃。
    """
    if not enabled or auto_orient:
        return None
    n = int(detected)
    if n > 0:
        if not quality_enabled:
            return None
        low_conf = (max_conf is not None
                    and float(max_conf) < float(min_conf))
        bad_kps = (frame_kps_quality is not None
                   and float(frame_kps_quality) < float(min_kps_quality))
        if not (low_conf or bad_kps):
            return None
    if int(input_size) >= RETRY_INPUT_SIZE:
        return None
    score = min(1.0, float(detect_score) + RETRY_SCORE_BUMP / 100.0)
    return RETRY_INPUT_SIZE, score
