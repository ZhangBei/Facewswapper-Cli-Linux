# 侧脸（ROP / 大 yaw / pitch）改造路线图与下一步计划

日期：2026-11-09

## 状态

- **RIP（roll，画面内歪头）系列已完成**：`--probe-angles`（8/16 向、只转检测副本 + kps 逆映射）、`--probe-on-target-miss`（受控补探测 + 跨角度合并去重）。
- **ROP（yaw / pitch，侧脸）系列已完成两个代码缺口**：P1 质量门控重检（`--detect-retry-quality`）、P2 68 点复用到 `swap_core`（`--align-for-swap`）。
- **Step 0 量化闭环已补齐**：`tools/measure_profile_faces.py` 提供采集 / 分析 / 判据，但**真机 yaw 分档曲线尚未采集**（本地无 GPU）。
- 全套离线测试：27 模块 25 PASS / 1 SKIP / 1 FAIL。FAIL 是既存的 `test_encoder`，与本系列改动无关；SKIP 是本机缺 numpy。

---

## 1. 边界：两个不同的问题

| | RIP（前几轮） | ROP（本轮） |
|---|---|---|
| 术语 | rotation-**in**-plane，绕视线轴 | rotation-**off**-plane，yaw / pitch |
| 现象 | 歪头、横躺 | 正脸 → 半侧 → 全侧 |
| 整帧旋转能否救 | 能（转正后脸变正） | **不能**，转 90° 脸仍是侧脸 |
| 项目对策 | `--probe-angles` / `--probe-on-target-miss` | 本文 |

三个独立失效面（README §5.1 只覆盖前两个）：

| 层 | 失败表现 | 物理原因 |
|---|---|---|
| **L1 检测召回** | 该换的脸没检出 → 整帧不换 | SCRFD 训练分布偏正脸；小脸在 640 特征图只剩几像素 |
| **L2 识别 / 匹配** | `--target-face` 没匹配上 / 换错人 | ArcFace 正脸训练；5 点相似变换在侧脸上裁剪形变 |
| **L3 替换几何与身份** | 换上去了但侧脸歪、身份漂移 | `swap_core` 用 detector 5 点算 `SimilarityTransform`；inswapper 以正脸训练 |

一句话：**L1 决定「换不换」，L2 决定「换谁」，L3 决定「换得像不像」。**

---

## 2. 已落地清单

### 2.1 项目原有开关（零代码，README §5 已实施）

| 开关 | 层 | 成本 | 默认 |
|---|---|---|---|
| `--detect-input-size 960` | L1 | 检测 ×2.25（960²/640²） | 640 |
| `--detect-retry` | L1 | 仅 0 检出帧 +1 次检测（960、阈值 +0.2） | 关 |
| `--target-face`（可 append） | L2 | 每多 1 参考图 +1 次 512 维点积 | 无 |
| `--recognize-mirror` | L2 | 识别 ×2 | 关 |
| `--recognize-dim2` | L2 | 识别 ×4（2×2 网格） | 关 |
| `--recognizer glintr100` | L2 | 识别 ×4，需下载 `arcface100` | `w600k_r50` |
| `--align 68-point` | L2 | 识别前 +1 次 2dfan4，需下载 `landmark68` | `5-point` |
| `--pose-compensation N` | L2 判定 | 零推理增量；`N>0` 时参考图多 1 趟检测+识别 | 0 |

累计：mirror × dim2 = **×8**；再叠 glintr100 = **×32**。

### 2.2 本系列新增（默认全部关闭，不传即逐像素不变）

| 开关 | 层 / 问题 | 关键约束 | 离线验证 |
|---|---|---|---|
| `--probe-angles STEP` | RIP / 0 检出帧扩角度 | `45`=8 向、`22.5`=16 向；不传走旧整帧旋转路径 | 仿射往返 0–345° 逐 15°，误差 6.82e-13 px |
| `--probe-on-target-miss` | RIP / 目标人物漏检 | 必须 `--target-face`；帧内 ≤3 张脸才触发 | 6 项测试 |
| `--detect-retry-quality [SPEC]` | L1 / 一正一侧混合帧 | 需 `--detect-retry`；`SPEC`=`MIN_CONF:MIN_KPS_QUALITY`，默认 `0.60:0.50` | 7 项测试 |
| `--align-for-swap [MIN_SIDENESS]` | **L3** / 侧脸贴歪 | 强制 `--align 68-point` + `--target-face`，仅 video；默认 `0.35` | 5 项测试 |

`--detect-retry-quality` 的质量代理与 `--align-for-swap` 的姿态代理都建立在**旋转不变的局部坐标系**（两眼连线为 u 轴）上：歪 45° 正脸不会像 `frontalness`（只看水平 x 偏移）那样被误判成侧脸 / 坏 kps。

### 2.3 配套工具

| 工具 | 用途 |
|---|---|
| `tools/probe_roll_recall.py` | RIP 收益量化：θ_tol 扫表 + 真机量 θ_tol |
| `tools/measure_profile_faces.py` | ROP Step 0：`--emit-template` / `--collect` / `--analyze` / `--self-check` |

---

## 3. 硬约束（改动前必须知道）

1. **默认零行为变化是所有改动的底线**：不传任何新开关时，输出必须与改造前逐像素一致（PSNR=`inf`）。每一步都保留「去掉开关即回滚」的路径。
2. **只读跨帧表允许，在线可变状态不允许**：`frame_order.py` 的 2N 有序窗口下帧会跨 worker 乱序完成，任何 EMA / `orient_miss_streak` 都会让输出依赖调度顺序。README §5.6 已明确放弃跨帧 embedding 回拉。
3. **不要重提已放弃方案**：降 `--detect-score`（误检把背景当脸换）、retry 叠 4 朝向、3D 对齐、106 点对齐、跨帧在线状态。
4. **`--detect-score` 有 unit bug**：`cli.py:176` 把 `args.detect_score * 100` 写进 `DetectScoreSlider`，再除 100 得 50.0，detector 恒 0 检出。**任何 A/B 都不要传这个 flag**（`measure_profile_faces.py --collect --detect-score` 是独立的 0-1 参数，不受影响）。
5. **未指定 `--target-face` 时识别层 5 个开关全部零收益**：`should_skip_recognize()` 跳过逐脸识别，`--recognize-*` / `--align` / `--pose-compensation` 没有读取方。

---

## 4. 下一步 Step 0：先把 yaw 分档曲线拿到手

方案 §4 第 1 条：**「没有这条 `yaw → 检出率` 曲线，下面所有选型都是猜」**。文献里没有现成的分档召回曲线（WIDER FACE 的 `atypical pose` 定义是 yaw>90°，评估不了 30–60° 常见侧脸）。

### 4.1 采集（GPU 机器）

```bash
# 1) 看表头与采集规范
python tools/measure_profile_faces.py --emit-template

# 2) 按 yaw 分档准备图片，每档 50-100 张，控制脸尺寸与分辨率一致
#    文件名写 yawNN 可自动分档，例如 yaw35_001.jpg -> 30-45 档
python tools/measure_profile_faces.py --collect frames/ \
    --models ./models --csv yaw.csv --detect-input-size 640

# 3) 分析
python tools/measure_profile_faces.py --analyze yaw.csv
```

`--collect` 用项目自己的 `Models.run_detect(..., with_scores=True)`，与生产路径同源；重试轮沿用 `detect_retry` 的 960 / +0.2，`kps_quality` 也来自同一个 stdlib 模块。纯 stdlib 部分（表头、分析、自校验）本地可跑：

```bash
python tools/measure_profile_faces.py --self-check
```

### 4.2 CSV 列与判据

`yaw_bin, img_id, face_px, detected_count, max_conf, is_zero_detect, detect_input_size, detect_score, retry_detected_count, retry_max_conf, kps_quality, target_match_score, matched`

`--analyze` 自动给出分档表（零检出率、重试救回率、平均脸尺寸 / max_conf / kps 质量）与以下判据：

| 观察 | 结论 |
|---|---|
| `>45°` 档零检出高且脸尺寸不小（≥ 低 yaw 参照的 80%） | 姿态本身 → 走 L1/L2/L3 |
| 漏检集中在小脸档 | **只需 `--detect-input-size 960`**，别动权重 |
| `yaw>45°` 占比 **< 5%** | 所有训练 / 换模型方案（P4 / P5）**直接排除** |
| 重试轮平均救回 < 20% | `+0.2` 阈值假设不成立，`--detect-retry` 收益存疑；用 `retry_max_conf` 列核对 |
| `yaw>45°` 占比 **> 10%** 且 T1–T5 / P1 / P2 全部到顶 | 才评估 P4 |

---

## 5. A/B 序列（每步可停、每步可回滚）

每档单独跑，不要一次全开。统一记录：**该换没换帧数 / 误换帧数 / 命中帧率（命中数 ÷ 总帧数）/ 逐帧耗时 / 不传开关时 PSNR=`inf`**。

```bash
# T0 回归底线
python -m faceswapper.tests
python tools/bench_gpu.py --run --psnr

# T1 检测层（L1）
... --source face.jpg --target-face person.jpg --target in.mp4 --output out_t1.mp4 \
    --detect-input-size 960 --detect-retry

# T2 识别层最低成本（L2）—— 用 65/comp10 而非 README 的 55/comp15
... --target-face person.jpg --recognize-mirror \
    --pose-compensation 10 --target-threshold 65

# T3 / T4 按成本递进，一档一档加
... + --recognize-dim2
... + --recognizer glintr100 --align 68-point

# T5 多角度参考图 gallery（any(>=) 是并集，阈值必须重标定）
... --target-face front.jpg --target-face yaw30.jpg --target-face yaw60.jpg \
    --target-threshold 65 --pose-compensation 10
```

`65 / comp 10` 的理由：`score = 100-(1-cos)*50`，默认 55 ⇔ `cos≥0.10` 已极松；`65/comp10` 把正脸线收紧到 `cos≥0.30`、全侧脸回到 55，是「收紧正脸误换」而不是整体放宽。

---

## 6. 本轮新增代码的真机验收与阈值标定

| 项 | 默认 | 要标定的量 | 验收方法 |
|---|---|---|---|
| `--detect-retry-quality` | `0.60:0.50` | `MIN_CONF`、`MIN_KPS_QUALITY` | 统计「触发帧占比 / 补回脸数 / 新增误检」；太松会把正常帧也重检 |
| `--align-for-swap` | `0.35` | `MIN_SIDENESS` | 大 yaw 视频 A/B，看贴回边缘与身份一致性；**正脸帧 PSNR 必须为 `inf`** |
| `--probe-on-target-miss` | 关 | 触发率 | 目标人物缺席帧也会触发，建议 `--every-nth 20` 先抽样 |
| `--probe-angles` | 不传 | θ_tol | `tools/probe_roll_recall.py --probe`；θ_tol ≥ 45° 时 8 向增益恒 0 |

开关叠加提示：`--probe-on-target-miss`、`--detect-retry-quality`、朝向兜底会在同一帧各自追加推理，目前**没有互斥**，建议先单独 A/B 再组合。

---

## 7. 剩余缺口：现在不做的理由

| 方案 | 层 | 触发条件 | 当前不做的原因 |
|---|---|---|---|
| **P3** 跨帧只读 identity / yaw 表 | L1+L2 | 「侧脸成段失效」被 yaw 曲线证实 | ① 实现步骤依赖 `路线B改造可行性评估与设计方案.md` §3，本仓库没有该文件；② 两遍解码、预扫描与正式两遍必须严格帧对齐；③ 主要风险是把「漏换」变成「误换」，镜头切换 / 人物快速移动时邻帧 kps 失效；④ 方案 §4 把它排在 Step 0 / T1–T5 之后 |
| **P4** finetune SCRFD 补 yaw 分布 | 全部 | `yaw>45°` 占比 >10% 且以上全部到顶 | 需 3DMM 合成 + 5 点标注同步变换、4 GPU / 25.8 万 iter；正脸 AP 可能回退 1–2 点，可能比救回的侧脸帧更亏 |
| **P5** 换模型 / 正面化 | — | — | **已排除**：WeaveFace 同协议 Hard 87.14 vs SCRFD-10GF 90.81（退步 3.67 点）；HF 现成「侧脸鲁棒 + 5 点」权重实查为空；接口不兼容 ORT 9 输出 |

P3 的侧脸特化要求（若将来做）：预扫描表每帧多存 `yaw_bin` 与 `kps_quality`，邻帧取身份时**只对 `yaw_bin` 相近的邻帧生效**，避免把正脸邻帧的 kps 用到全侧帧上贴歪。只读表允许，在线可变 EMA 不允许。

---

## 8. 决策矩阵（速查）

| 输入 | 决策 |
|---|---|
| `yaw>45°` 占比 < 5% | 只做零成本 A/B，不动权重、不训练、不换模型 |
| 漏检集中在小脸 | 只加 `--detect-input-size 960` |
| 大 yaw 零检出且脸不小 | T1 → T2 → T3/T4 → T5 逐档；混合帧仍漏才调 P1 |
| 侧脸贴回几何明显歪 | `--align-for-swap`（前提：`--align 68-point` + `--target-face`） |
| 目标人物整段侧脸、逐帧盲试都无效 | 才评估 P3 |
| `yaw>45°` >10% 且全部到顶 | 才评估 P4 |

---

## 9. 诚实边界

- 本地**无 GPU**：本系列所有改动只做了离线验证，未跑任何真机检测 / 换脸 A/B、未下载权重、未做训练。所有倍率来自 README §5 自述与面积比算术，**收益数字全部未知**。
- `frontalness` 在约 59.7° yaw 饱和，60° 与 90° 侧脸拿同一补偿；本轮新代理 `yaw_sideness` 修了 roll 污染，但**同样是大角度饱和的线性代理**，只做粗门控，不当精确角度用。
- 68→5 点索引与 `arcface_dst` 模板的一致性只在 README 声称成立，**未真机核对**；`inswapper` 的正脸训练偏差造成的身份漂移仍在（`--restorer` 只修画质不修姿态一致性）。
- 「yaw 0–57° / pitch ±56.25°」来自识别侧选帧研究，不能直接当检测阈值。
- 权重许可（InsightFace / FaceFusion / ModelScope / 社区 ONNX）需落地前确认；项目无 CPU 路径，全部方案假定 GPU 在线环境。
- 既存 `test_encoder` FAIL 与既存未提交改动（README 大幅删减、`config.py`、`test_config.py` 等）均非本系列所为。

---

## 10. 回滚

所有新增开关默认关闭，**去掉开关即回到旧行为逐像素一致**。若需整体回退，按落地顺序逆序移除：`--align-for-swap` → `--detect-retry-quality` → `--probe-on-target-miss` → `--probe-angles`。每个开关都可独立关，互不影响。
