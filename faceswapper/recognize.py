"""逐帧 ArcFace 识别调用的开关判定。

stdlib-only：``VideoManager`` 需要 torch / OpenCV 才能 import，而「这一次
``run_recognize`` 的结果到底有没有读取方」是一条纯逻辑，可以脱离 GPU 栈
做行为测试。

headless 的 video 路径把所有检测到的脸都换成同一个 ``force_source_embedding``。
只有当 ``--target-face`` 给出参考身份、需要用相似度筛掉其他人的时候，逐脸
算出的 embedding 才会被读到。其余情况下每帧 N 次 ArcFace 推理的结果全是死数据。
"""

from __future__ import annotations

from typing import Any


def should_skip_recognize(
    *,
    force_source: Any | None,
    targets: Any | None,
    opt_out: bool = False,
) -> bool:
    """是否可以跳过逐脸 ``run_recognize``。

    - ``opt_out``：用户显式要求保留旧路径（``--force-recognize``），一律不跳。
    - ``force_source is None``：图片 / GUI 的 ``found_faces`` 匹配路径，逐脸
      embedding 会被 ``findCosineDistance`` 读取，不能跳。
    - ``targets`` 非空：``--target-face`` 相似度筛选需要 embedding，不能跳。
    - 其余（headless video、未指定 target-face）：embedding 没有任何读取方，
      可以安全跳过，像素输出完全不变。
    """
    if opt_out or force_source is None:
        return False
    if targets:
        return False
    return True
