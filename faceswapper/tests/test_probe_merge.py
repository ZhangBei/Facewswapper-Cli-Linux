"""--probe-on-target-miss 的离线测试（无 GPU / torch / numpy / OpenCV）。

覆盖三步最小落地顺序里可离线验证的部分：

1. 受控触发门控 ``should_probe_on_miss`` 的全部条件。
2. 跨角度合并去重 ``merge_probe_landmarks``：0 度检出优先保留、5 点 RMS
   判同脸、score 降序仲裁、max_faces 截断。
3. 接线：``with_scores`` 默认关闭、SCRDF 的 det/kpss 截断已修、新开关
   默认 False 时旧主链路逐字保留。

    python -m faceswapper.tests.test_probe_merge
"""

from __future__ import annotations

import sys
from pathlib import Path

from faceswapper.orientation import (
    PROBE_ON_MISS_MAX_FACES,
    kps_rms,
    kps_scale,
    merge_probe_landmarks,
    should_probe_on_miss,
)

ROOT = Path(__file__).resolve().parents[2]
FACE_A = [(100.0, 100.0), (140.0, 100.0), (120.0, 125.0),
          (105.0, 150.0), (135.0, 150.0)]
FACE_B = [(500.0, 150.0), (540.0, 150.0), (520.0, 175.0),
          (505.0, 200.0), (535.0, 200.0)]


def shifted(face, dx, dy):
    return [(x + dx, y + dy) for x, y in face]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise SystemExit(1)


def test_gate_truth_table() -> None:
    ok = dict(enabled=True, has_targets=True, detected=2, matched_any=False)
    if not should_probe_on_miss(**ok):
        fail("the canonical miss case must probe")
    if PROBE_ON_MISS_MAX_FACES != 3:
        fail(f"max_faces default changed: {PROBE_ON_MISS_MAX_FACES}")
    blocked = (
        dict(enabled=False),
        dict(has_targets=False),
        dict(detected=0),                 # 0 检出帧归 should_probe 管
        dict(detected=PROBE_ON_MISS_MAX_FACES + 1),  # 群像镜头跳过
        dict(matched_any=True),          # 正常帧零额外推理
        dict(already_rotated=True),      # kps 不在原帧坐标域
    )
    for override in blocked:
        kw = dict(ok)
        kw.update(override)
        if should_probe_on_miss(**kw):
            fail(f"gate must reject {override!r}")
    # 自定义上限
    if not should_probe_on_miss(**{**ok, "detected": 5, "max_faces": 5}):
        fail("custom max_faces must be honoured")
    print("  ok  gate: only target-miss frames with <=3 faces, never rotated ones")


def test_kps_metrics() -> None:
    if abs(kps_rms(FACE_A, FACE_A)) > 1e-12:
        fail("identical kps must have zero RMS")
    expect = (2.0 ** 2 + 1.0 ** 2) / 2.0
    if abs(kps_rms(FACE_A, shifted(FACE_A, 2, 1)) ** 2 - expect) > 1e-9:
        fail("kps_rms must average over both coordinates")
    if abs(kps_scale(FACE_A) - 64.031242374) > 0.01:
        fail(f"kps_scale unexpected: {kps_scale(FACE_A)}")
    if kps_scale([]) != 0.0 or kps_rms([], []) != float("inf"):
        fail("degenerate kps must not crash the merge")
    print("  ok  kps_rms / kps_scale: numeric and degenerate inputs")


def test_merge_keeps_base_and_dedups() -> None:
    # 0 度检出的 FACE_A 永远保留；45/90/135 度副本都只是它的重复检出。
    same = [shifted(FACE_A, 2, 1)]
    cands = [
        (45.0, same, [0.90]),
        (90.0, same, [0.95]),
        (135.0, [shifted(FACE_A, -2, 1)], [0.80]),
    ]
    if merge_probe_landmarks([FACE_A], cands) != []:
        fail("repeated detections of the same face must be dropped")
    # 只返回新增部分，不含 base。
    added = merge_probe_landmarks([FACE_A], [(90.0, [FACE_B], [0.7])])
    if len(added) != 1 or kps_rms(added[0], FACE_B) > 1e-9:
        fail("merge must return only newly found faces, in frame coords")
    if added[0] == FACE_A:
        fail("merge must not return the base face")
    print("  ok  merge: base kept, duplicates dropped, only new kps returned")


def test_merge_score_order_and_cap() -> None:
    # score 降序：上限只容 1 张新脸时应保留高分的 FACE_B。
    cands = [(45.0, [FACE_A], [0.50]), (90.0, [FACE_B], [0.90])]
    added = merge_probe_landmarks([], cands, max_faces=1)
    if len(added) != 1 or kps_rms(added[0], FACE_B) > 1e-9:
        fail("higher-score candidate must win when the cap is tight")
    # 平手时小角度优先（90 倍数是像素重排，无插值损失）。
    tied = [(90.0, [FACE_B], [0.8]), (45.0, [FACE_A], [0.8])]
    tied_added = merge_probe_landmarks([], tied, max_faces=1)
    if kps_rms(tied_added[0], FACE_A) > 1e-9:
        fail("tie must prefer the smaller probe angle")
    # base 已经占满名额时不再追加。
    if merge_probe_landmarks([FACE_A], cands, max_faces=1) != []:
        fail("merge must respect max_faces as a total cap")
    # 空候选 / 退化脸（scale=0）不应崩。
    if merge_probe_landmarks([FACE_A], []) != []:
        fail("empty candidates must add nothing")
    degenerate = [(x, y) for x, y in [(10.0, 10.0)] * 5]
    if len(merge_probe_landmarks([], [(45.0, [degenerate, degenerate],
                                      [0.5, 0.6])])) != 1:
        fail("degenerate kps must dedup without crashing")
    print("  ok  merge: score order, tie-break, total cap, empty/degenerate")


def test_default_off_wiring() -> None:
    """接线 + 默认零行为变化（源码断言，本机无法 import GPU 栈）。"""
    cli = (ROOT / "faceswapper" / "cli.py").read_text(encoding="utf-8")
    headless = (ROOT / "faceswapper" / "headless.py").read_text(encoding="utf-8")
    vm = (ROOT / "faceswapper" / "VideoManager.py").read_text(encoding="utf-8")
    models = (ROOT / "faceswapper" / "Models.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if '"--probe-on-target-miss"' not in cli:
        fail("cli.py missing --probe-on-target-miss")
    if cli.count("probe_on_target_miss=args.probe_on_target_miss") != 2:
        fail("the flag must reach both SwapOptions and VideoOptions")
    if "--probe-on-target-miss" not in readme:
        fail("README must document the new flag (test_docs_stay_in_sync)")
    if "probe_on_target_miss: bool = False" not in headless:
        fail("SwapOptions missing the field")
    if "self.vm.probe_on_target_miss = bool(" not in headless:
        fail("headless must inject the flag")
    if "self.probe_on_target_miss: bool = False" not in vm:
        fail("VideoManager must default the flag to False")
    if "def _probe_on_target_miss(" not in vm:
        fail("VideoManager missing the miss-probe method")
    if "if self.probe_on_target_miss and ret:" not in vm:
        fail("the miss-probe must be gated off by default")
    if "if ret and self.force_source_embedding is not None:" not in vm:
        fail("the original swap loop branch must stay intact")
    # 只追加不替换：新增脸走 append，不改已有 ret 条目。
    method = vm.split("def _probe_on_target_miss(", 1)[-1].split(
        "def probe_orientation(", 1)[0]
    if "ret.append(" not in method or "merge_probe_landmarks(" not in method:
        fail("miss-probe must append merged faces, never replace them")
    if "should_probe_on_miss(" not in method:
        fail("miss-probe must use the stdlib gate")
    print("  ok  wiring: flag threaded, default off, original swap loop intact")


def test_scores_wiring() -> None:
    """with_scores 默认关闭 + SCRDF 的 det/kpss 截断一致性。"""
    models = (ROOT / "faceswapper" / "Models.py").read_text(encoding="utf-8")
    if "def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5," not in models:
        fail("run_detect signature changed unexpectedly")
    if "input_size=640, with_scores=False):" not in models:
        fail("with_scores must default to False")
    if "if with_scores:" not in models or "return kpss, scores" not in models:
        fail("run_detect must return (kpss, scores) only when asked")
    if "return kpss\n" not in models:
        fail("run_detect default path must still return kpss alone")
    # 空检出早退分支必须同时给空 scores。
    if "np.zeros((0,), dtype=np.float32)" not in models:
        fail("empty-detection branch must return empty scores")
    # Retinaface 路径：keep 同时作用于 kpss 与 scores。
    if "all_scores[keep].cpu().numpy()" not in models:
        fail("Retinaface scores must follow the same keep indices")
    # SCRDF 路径：修复 max_num 只截 kpss 不截 det 的遗留不一致。
    scrdf = models.split("def detect_scrdf(", 1)[-1].split(
        "def recognize(", 1)[0]
    if "det = det[bindex]" not in scrdf:
        fail("SCRDF max_num branch must truncate det alongside kpss")
    if "det[:, 4].astype(np.float32, copy=False)" not in scrdf:
        fail("SCRDF must return det[:,4] as scores")
    print("  ok  scores: opt-in by default, empty branch, SCRDF truncation fixed")


def main() -> int:
    print("faceswapper probe-on-target-miss tests")
    test_gate_truth_table()
    test_kps_metrics()
    test_merge_keeps_base_and_dedups()
    test_merge_score_order_and_cap()
    test_default_off_wiring()
    test_scores_wiring()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
