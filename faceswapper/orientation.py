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
"""

from __future__ import annotations

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


def choose_angle(counts: dict[int, int]) -> int:
    """从各角度的检出脸数里挑胜出角度，``0`` = 不旋转。

    平手或全部为 0 一律取 0——证据不足时宁可保持原帧。只认
    ``PROBE_ANGLES`` 里的角度，其它 key 静默忽略（防止调用方把 0 度或
    任意角度混进来）。严格 ``>`` 才更新，所以平手时取最小角度。
    """
    best_angle = 0
    best_count = 0
    for angle in PROBE_ANGLES:
        count = int(counts.get(angle, 0))
        if count > best_count:
            best_count = count
            best_angle = angle
    return best_angle
