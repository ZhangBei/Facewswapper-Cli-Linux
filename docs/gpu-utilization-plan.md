# 提升 GPU 与 VRAM 利用率的实施方案

日期：2026-11-05（最近核对：2026-11-06 五次核对，含 nsys / A/B / 30s 正式基线 / 7.1 受控实验 / P1 安全修复）

## 状态

- 阶段 0 **已完成**：10 秒矩阵两轮 + 30 秒 / 750 帧正式基线，`elapsed`、
  `avg fps`、`--profile` 全部可复现，见「首批 / 第二轮真机数据」。
- 阶段 1 的 `--workers N` 已实现并通过两轮真机矩阵：30 秒片段
  `w1 3.3 → w2 5.0 → w4 5.4 fps`（加速比 `1.0 / 1.503 / 1.631`），
  w2 / w4 对 w1 的逐帧 PSNR 均为 `inf`。GPU / VRAM 曲线仍受同机 4 个
  外部进程污染，本环境无隔离条件。
- 阶段 2 `--encode-thread`：真机 A/B 为**负收益**（64.8s / 3.9 fps vs
  基线 60.9s / 4.1 fps），PSNR=`inf`。不建议开启，见 5.4。
- 阶段 3 `--defer-d2h` / `--arena-strategy pow2`：真机 A/B 与基线持平或
  略负（63.0s / 4.0 fps），PSNR=`inf`。`nsys` 已定位 host stall：
  `cudaStreamSynchronize` 占 CUDA API 时间 75.6%，而 `syncvec_drain` 仅
  0.323s、D2H memcpy 仅 0.062s —— **原 B2 把 `syncvec.cpu()` 当作主要
  瓶颈的判断不成立**，见「第二轮真机数据」。
- 阶段 4：7.1 **受控解冻实验已做**（进程内临时开闸、不改文件），结论是
  **当前模型不可解冻**——`inswapper_128.fp16.onnx` 为 `static-batch=1`，
  N=3 的 batch 被 ONNX 拒绝（`Got: 3 Expected: 1`），异常回退后触发
  `CUDA failure 700` 崩溃；闸门关闭的对照实验正常完成。该安全缺陷
  **已修并复测**（补动态 batch 判据 + 修正 session 探测顺序，误开闸门
  现在干净回退、PSNR=`inf`）。两个闸门仍为 `False`，7.1 / 7.2 均被
  模型能力阻断，不再是「待解冻」。4.2 跳过逐脸
  recognize 已由 nsys 间接证实（`:run_recognize` 全程仅 1 次），仍是唯一
  默认生效的行为变化。详见「第二轮真机数据」的 7.1 受控实验。
- 阶段 5：5a NVENC 代码已落地，真机**明确报错不可用**——
  `avcodec_open2("h264_nvenc")` 抛 `PermissionError`（容器 NVENC 权限 /
  驱动 API 受限），符合「不静默回退」设计，默认 `libx264` 不受影响。
  5b GPU 常驻解码仍**未验证**：`torchcodec` 未安装。
- 阶段 6 `--reuse-detect-buffers` 与阶段 7 `--zero-copy-frames`：均已做
  真机 A/B（含组合），PSNR 全部 `inf`（逐像素一致），但**无可测量吞吐收益**
  （3.9–4.0 fps vs 基线 4.1）。nsys 给出收益上限：`cudaMalloc` + `cudaFree`
  约 0.07s、H2D + D2H memcpy 合计约 0.25s，相对 36s 总用时不足 1%。
  两阶段验收清单的「逐帧 PSNR 一致」项已达成，见第 14 / 15 节。
- 代码层 11 个阶段 / 子项全部收口；**真机侧 11 项均已给出结论**：
  0 / 1 / 2 / 3 / 4.2 / 5a / 6 / 7 已实测，4.1 / 4.3 被 `static-batch=1`
  模型阻断，5b 缺可选依赖且收益上限 <1%。最重要的结论是：**除
  `--workers 2` 的约 1.5× 外，其余优化在真机上均无收益或不可行**；
  唯一对症的「减少 ORT 调用次数」需要换支持动态 batch 的 swapper 模型，
  而不是继续改当前代码。
- 环境：Tesla T4 / 15 GiB / 驱动 580，运行时栈已由
  `tools/install.py default` 补齐，`check_env.py` 9 项全 ok；但同机持续有
  4 个外部 GPU 进程占约 5.6 GiB，**GPU 利用率 / VRAM 曲线无法隔离采集**，
  这一项在本环境属于不可完成，而非尚未执行。
- 离线测试 `python -m faceswapper.tests --strict` 18/18 PASS；四个闸门
  均为 `False`；`compileall` 与 `git diff --check` 通过。

### 完成情况核对（2026-11-06 五次核对）

对照代码库与两轮真机结果逐项复查：**代码层 11 个阶段 / 子项全部收口**；
真机侧 11 项**均已给出结论**（含 7.1 受控解冻实验）。其中 0 / 1 / 2 /
3 / 4.2 / 5a / 6 / 7 拿到实测数据，4.1 / 4.3 被当前 `static-batch=1`
模型阻断，5b 缺可选依赖。**核心结论**：除 `--workers 2` 的约 1.5× 外，
所有优化在真机上均无收益或不可行；要继续提升吞吐，前提是换支持动态
batch 的 swapper 模型，见「第二轮真机数据」。

| 阶段 | 代码 | 离线测试 | 默认行为 | 真机验证 |
|---|---|---|---|---|
| 0 基线 / `avg fps` / `--profile` | 已完成 | PASS | 默认生效 | **已完成**（10s × 2 + 30s / 750 帧正式基线） |
| 1 `--workers N` | 已完成 | PASS | `workers=1`（串行） | **基本完成**（30s 矩阵、PSNR=inf；GPU 曲线本环境不可归因） |
| 2 `--encode-thread` | 已完成 | PASS | 关闭 | **已完成**：负收益，PSNR=inf |
| 3 `--defer-d2h` / `--arena-strategy` | 已完成（保守版） | PASS | 均关闭 | **已完成**：持平，PSNR=inf，nsys 已定位 |
| 4.1 单帧多脸 batch | 已编写、闸门关闭 | PASS + AST 闸门（新增 2 项） | `EXPERIMENTAL_FACE_BATCH=False` | **模型阻断，不可解冻**；但安全缺陷已修并复测：误开闸门干净回退、PSNR=`inf` |
| 4.2 跳过逐脸 recognize | 已完成 | PASS | **默认生效**，`--force-recognize` 回滚 | **间接验证**（nsys `:run_recognize` 仅 1 次；收益数字未单测） |
| 4.3 跨帧 batch | 已编写、未挂载主路径 | PASS + AST 未挂载 | `EXPERIMENTAL_CROSS_FRAME_BATCH=False` | **受模型阻断**（同样依赖动态 batch），放最后 |
| 5a NVENC 编码 | 已完成 | PASS | 关闭（默认 `libx264`） | **已完成**：真机不可用，明确报错 |
| 5b GPU 常驻解码 | 已完成 | PASS | 关闭（默认 `pyav`） | **未做**（`torchcodec` 未安装） |
| 6 detector 输出缓冲复用 | 已完成 | PASS | 关闭，`trt` 自动关 | **已完成**：PSNR=inf，无收益 |
| 7 帧路径零拷贝 | 已完成 | PASS | 关闭 | **已完成**：PSNR=inf，无收益 |

核对依据：`faceswapper/tests/` 下 18 个离线模块（新增
`test_suite_runner`），`python -m faceswapper.tests --strict` 实测 18/18
PASS；四个实验 / 优化闸门均为 `False`；`compileall` 与
`git diff --check` 通过。

11-06 后续复核修正：`test_install` 里两处「假设本机无 torch / bare
machine」的断言在装了 CUDA 栈后会假失败，已改为注入 `find_spec` 与临时
requirements 文件的行为测试；`test_target_match` 早期因缺 numpy 的环境性
失败已随环境补齐消失（runner 仍保留 `SKIP` 分类）。验收脚本同时修掉两处
真机才暴露的缺陷：`build_psnr_command` 原用 `-loglevel error` 吞掉 PSNR
统计行（现改为 `-nostats -loglevel info` 并解析 `psnr_average`），
`build_gpu_monitor_command` 原用 `-s u,m` 被 dmon 拒绝（现改为 `-s um`）。
三次核对新增：`nsys` 轨迹、30 秒 / 750 帧正式基线、阶段 2 / 3 / 6 / 7
四组 A/B（含组合）与 5a NVENC 探测均已执行，结果见「第二轮真机数据」；
18 个离线模块仍为 18/18 PASS，四个闸门仍为 `False`。

三点需要特别留意：

1. **唯一默认改变行为的项是 4.2**。其余新开关全部默认关闭或闸门为 False，
   不传开关即回到基线路径；这也是本文所有回滚说明的前提。
2. **7.1 / 7.2 不只是「冻结」，而是被模型能力阻断**：受控实验证明当前
   `inswapper_128.fp16.onnx` 为 `static-batch=1`，当时 N=3 的 batch 被
   ONNX 拒绝、回退后触发 `CUDA failure 700`。该安全缺陷已修（动态 batch
   判据 + session 探测顺序），复测确认误开闸门会干净回退且 PSNR=`inf`；
   但两个闸门仍必须保持 `False`，要解冻只能先换支持动态 batch 的
   swapper 模型。
3. **性能改造基本证伪，别再堆开关**：nsys 定位到
   `cudaStreamSynchronize`（CUDA API 时间 75.6%）与逐脸 `ort_run_swapper`
   （225 次 / 11.77s）才是瓶颈，而 `syncvec_drain` / memcpy / 显存分配
   合计不足总用时 1%。阶段 2 / 3 / 6 / 7 的 A/B 全部持平或负收益，
   5a 环境不支持。唯一对症的「减少 ORT 调用次数」（每帧 3 张脸 →
   1 次 Run）在当前模型下无法实施，需先解决模型能力这一前置条件。
4. **GPU / VRAM 曲线在本环境不可完成**：同机 4 个外部 GPU 进程持续占用
   约 5.6 GiB 且 PID 随会话变化，无法隔离。不要再列为「待执行」，
   应记为环境限制；「GPU 利用率低」这一旧判断也始终未得到证实。

### 下一步动作（2026-11-06 五次核对，P1 安全修复已完成）

P0 / P1 / P2 中可执行的项**已全部完成**（30 秒正式基线、nsys、阶段 2 / 3 /
6 / 7 的 A/B、5a 探测、7.1 受控开闸实验、P1 安全修复与复测）。阶段 2 / 3 /
6 / 7 已由真机证伪，不再投入。剩余按价值排序：

| 优先级 | 动作 | 预期产出 | 状态 |
|---|---|---|---|
| P0 | 评估并换用支持动态 batch 的 swapper 模型（当前 `inswapper_128.fp16.onnx` 锁死 batch=1） | 解锁 7.1 / 4.3 的唯一前提 | 未做，需外部模型 |
| ~~P1~~ | ~~修 CUDA 700 回退 + 补动态 batch 判据~~ | 误开闸门时干净回退 | **已完成**（见下） |
| P2 | 5b：安装 `torchcodec>=0.4` 核对 frame 布局 / PTS 单位 | 可用性结论；H2D 收益上限 <1% | 低价值 |
| — | 4.3 跨帧 batch 解冻 | 上限更高，需重构帧流水线 | 放最后，同受模型阻断 |
| — | 隔离同机 GPU 进程后重采利用率 / VRAM 曲线 | 可归因的 GPU 数字 | **本环境不可完成** |

**P1 已完成**：`batch_eligibility()` 增加必填的 `batch_backend_available`
（与 7.2 同名同义、不设默认值），`Models.swapper_supports_dynamic_batch()`
显式触发 session 懒加载后返回模型能力；`run_swapper_batched` 把
`_get_swapper_session()` 提前到静态 batch 检查之前作为第二道防线。
新增 2 项离线行为测试（含接线 AST 检查），全量 18/18 PASS。真机复测
临时开闸跑 3 秒 / 75 帧片段：`done: 75 frame(s)`、无 CUDA 错误、输出与
闸门关闭的逐脸基线 **PSNR=`inf`**，即误开闸门现在会**干净回退**。

**当前卡点**不是缺代码或 GPU，而是模型能力：`inswapper_128.fp16.onnx`
把 batch 维锁死为 1，任何 batch 路线都走不通。
其次是 GPU 读数被外部负载污染。继续调 `workers`、编码线程、D2H、
分配器、memcpy 都不会有收益。

### 首批真机数据（2026-11-06）

**环境**：Tesla T4 / 15 GiB / 驱动 580.178.04；torch 2.10.0+cu128、
onnxruntime-gpu 1.31.1（`CUDAExecutionProvider`）、onnx 1.23.2、av 18.1.0；
`models/` 三个基础模型齐全，`tools/check_env.py` 9 项全 ok。

**样本**：按 3.4「用短片段」的建议，从 `/workspace/data/video/target.mp4`
（720p / 25 fps / 4844 帧）剪出前 10 秒，250 帧、720p、含音频；`--source`
用 `face10.jpg`。规模是冒烟级，**不是 3.4 建议的 30–60 秒正式基线**，
结论只用于判断趋势与推进顺序。

**命令**：

```bash
python tools/bench_gpu.py \
  --source /workspace/data/picture/face10.jpg \
  --target bench-out/smoke-10s.mp4 \
  --output-dir bench-out/matrix-10s-v2 \
  --workers 1,2,4 --run --monitor-gpu --check-frames --psnr
```

**结果**（跑两轮，数字可复现；括号内为第一轮）：

| workers | elapsed | avg fps | speedup | decode | swap | encode |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 81.7s (83.7s) | 3.1 (3.0) | 1.000 (1.000) | 2.57s | 64.61s | 3.99s |
| 2 | 60.9s (60.7s) | 4.1 (4.1) | 1.342 (1.379) | 3.49s | 99.30s | 5.33s |
| 4 | 56.7s (57.9s) | 4.4 (4.3) | 1.441 (1.446) | 4.65s | 180.07s | 6.08s |

**正确性**：三个用例 `rc=0`，`ffprobe` 均为 250 帧 / 10.000s / 25 fps、
`audio=yes`；w2 与 w4 对 w1 的逐帧 PSNR 均为 `inf`（逐像素完全一致）。
并行下 `--profile` 三段之和大于总用时，与文档说明一致，不要相加。

**结论**：

1. 阶段 0 的「通过标准」达成：拿到可复现数字，`workers=1` 可作回归基线。
2. 阶段 1 的保序与像素一致性通过真机验证，`--workers N` 不再是纯实验开关。
3. **w2→w4 收益几乎消失**（4.1 → 4.4 fps，加速比 1.34 → 1.44），正是
   4.2 节待验证步骤第 4 条预判的 GIL / Python 调度瓶颈。**不应继续加 N**。
   后续 nsys 已完成（见「第二轮真机数据」），结论是瓶颈在逐脸 ORT Run
   与 stream 同步；阶段 2 / 3 的 A/B 也证实无收益，方向应转向 7.1。
4. `encode` 仅 4–6s（约 5–8%），`swap` 占绝大部分，印证阶段 2 的保留意见：
   编码独立线程的增量可能很小甚至为负。开启前先看 `--profile` 的
   `encode=` 占比。
5. GPU / VRAM 曲线**已复采但不能用**：原 `-s u,m` 使 dmon 直接退出，
   改为 `-s um` 后在 w1 单用例上取到 87 个采样点（`avg sm 86.3%`、
   `max sm 100%`、`fb` 峰值 6895 MB）。但 `nvidia-smi --query-compute-apps`
   同时显示 **4 个其它 GPU 进程占约 5704 MiB**，且空闲时 dmon 的 sm 仍在
   25–93% 波动、fb 恒为 5707 MB。**这些数字不能归因于 faceswapper**，
   要判断本项目自身的利用率与显存，必须隔离同机负载后重采。这也说明
   「GPU 利用率低」不能靠这台机器的 `nvidia-smi` 直接下结论。

**后续结果**：30 秒正式基线、nsys、阶段 2 / 3 / 6 / 7 的 A/B 与 5a 探测
已于同日完成，见下节；本节的「仍缺」列表因此大幅收窄。报告见
`bench-out/matrix-10s-v2/bench_report.json`，GPU 日志见
`bench-out/monitor-check/gpu-baseline.log`。

### 第二轮真机验证（2026-11-06）

按上一版「下一步动作」的 P0 / P1 / P2 与 5a 执行。A/B 统一用 10 秒 /
250 帧片段、`workers=2`、`--extra-flag` 透传，对照基线为
`bench-out/matrix-10s-v2` 的 w2（60.9s / 4.1 fps）。

#### 30 秒正式基线（P0）

`bench-out/baseline-30s.mp4`（750 帧 / 30.000s / 720p / 含音频）：

| workers | elapsed | avg fps | speedup | decode | swap | encode | PSNR vs w1 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 226.9s | 3.3 | 1.000 | 7.59s | 191.42s | 12.56s | — |
| 2 | 151.0s | 5.0 | 1.503 | 9.34s | 280.03s | 13.56s | `inf` |
| 4 | 139.1s | 5.4 | 1.631 | 12.90s | 512.42s | 14.90s | `inf` |

三个用例 `rc=0`、`ffprobe` 均 750 帧、`psnr_average=inf`。与 10 秒片段
（3.1 / 4.1 / 4.4 fps）相比 fps 全面上升，**量化了启动开销**：短片段里
懒建 session 与 cuDNN benchmark 摊薄不掉，4.2 节第 3 条「短视频可能
得不偿失」得到证实。w2 → w4 仍只有 +0.4 fps，收益递减结论不变。报告见
`bench-out/baseline-30s/bench_report.json`。

#### nsys host stall（P1 核心卡点）

`nsys profile -t cuda,nvtx` 采集 3 秒 / 75 帧片段的 w1 路径，报告
`bench-out/nsys-w1.nsys-rep`（`elapsed 36.1s`，
`profile: decode=0.74s swap=21.73s encode=1.19s`）：

| 观测 | 数值 | 含义 |
|---|---|---|
| `cudaStreamSynchronize` | **13.41s / 2618 次，占 CUDA API 时间 75.6%** | host stall 主体，主机在等 stream |
| `ort_run_swapper`（NVTX） | 11.77s / 225 次，平均 52.3ms | 每帧 3 张脸 × 75 帧，逐脸 ORT Run |
| `swap_core` / `sc_polypass` | 17.05s / 13.16s（各 225 次） | 嵌套 range，与上面重叠计时，不要相加 |
| `detect_retinaface` | 5.67s / 76 次 | 每帧一次检测 |
| `syncvec_drain`（NVTX） | **0.323s / 225 次** | B2 假设的主要瓶颈，实测占比极小 |
| H2D + D2H memcpy | 0.181s + 0.062s = **0.243s** | 相对 36.1s 不足 1% |
| `cudaMalloc` + `cudaFree` | 0.033s + 0.041s = **0.074s** | 分配器开销同样可忽略 |
| `:run_recognize` | **仅 1 次 / 1.56s** | 4.2 的逐脸识别跳过确实生效 |

**结论**：瓶颈不是 B2 说的 `syncvec.cpu()`，也不是编码、D2H 或显存分配，
而是**逐脸 `ort_run_swapper` 调用及其内部同步**。这解释了阶段 2 / 3 / 6 / 7
为何全部无收益，也说明「减少 ORT 调用次数」是唯一对症的方向。但紧接着的
7.1 受控实验证明，当前 `static-batch=1` 模型走不通这条路，见下节。

#### 阶段 2 / 3 / 6 / 7 的 A/B（P1 / P2）

| 开关 | 报告 | elapsed | avg fps | PSNR vs 基线 w2 |
|---|---|---:|---:|---|
| 基线（无开关） | `matrix-10s-v2` w2 | 60.9s | 4.1 | — |
| `--encode-thread`（阶段 2） | `ab-stage2` | 64.8s | 3.9 | `inf` |
| `--defer-d2h --arena-strategy pow2`（阶段 3） | `ab-stage3` | 63.0s | 4.0 | `inf` |
| `--reuse-detect-buffers`（阶段 6） | `ab-stage6` | 62.6s | 4.0 | `inf` |
| `--zero-copy-frames`（阶段 7） | `ab-stage7` | 63.3s | 3.9 | `inf` |
| 阶段 6 + 7 组合 | `ab-stage67` | 62.8s | 4.0 | `inf` |

五组输出与基线**逐像素完全一致**（`average:inf`），正确性验收通过；
但吞吐全部持平或略降（差异在噪声内），**没有任何一组有可测量收益**。
这与 nsys 给出的收益上限一致，不需要再重复 A/B。报告在
`bench-out/ab-*/bench_report.json`。

#### 5a NVENC（P3 的一部分）

`h264_nvenc` 在 `ffmpeg -encoders` 里可见，但真机
`avcodec_open2("h264_nvenc", {'rc':'vbr','cq':'23','preset':'p1'})`
抛 `av.error.PermissionError: [Errno 1] Operation not permitted`，
FFmpeg 附带日志 `The minimum required Nvidia driver for nvenc is 570.0
or newer`（本机驱动 580，更可能是容器内 NVENC 设备权限 / 驱动 API
访问受限）。CLI 明确报错、退出码非 0、**没有静默回退到 libx264**，
符合 8.2 的设计。结论：**本环境 5a 不可用**，`p1..p7` 兼容性无从核对。
报告见 `bench-out/ab-nvenc/bench_report.json`。

#### 7.1 受控解冻实验（P0，结论：当前模型不可解冻）

用**进程内 monkey-patch** 临时把 `EXPERIMENTAL_FACE_BATCH` 置 `True`
（`python -c` 里改 `faceswapper.VideoManager` 的模块属性），不改任何源文件，
进程结束即自动恢复；跑 10 秒 / 250 帧多人脸样本、`workers=1`，参数为默认
`inswapper-128` / `ultrafast` / `crf 23` / `--profile`。

结果**没有产出输出文件**，日志依次为：

1. `[Models] inswapper: CUDAExecutionProvider (io=float32 static-batch=1)`
   —— 模型 `inswapper_128.fp16.onnx` 的 `target` 输入第 0 维锁死为 1。
2. `[Models] swapper batched call rejected (RuntimeError: ... Got invalid
   dimensions for input: target ... index: 0 Got: 3 Expected: 1);
   falling back to per-call for the rest of the session` —— N=3 的 batch
   被 ONNX 拒绝，`run_swapper_batched` 的异常回退路径被触发。
3. `CUDA failure 700: an illegal memory access was encountered` 与
   `CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR`，进程崩溃。

**对照实验**：闸门保持 `False`、同一 3 秒 / 75 帧片段正常完成
（`done: 75 frame(s)`、`elapsed 39.0s`、无 CUDA 错误），排除同机 4 个
外部 GPU 进程造成偶发故障。实验后两个闸门仍为 `False`（文件未改），
`bench-out/face-batch-on.mp4` 未生成。

**结论**：7.1 的 `batch_eligibility()` 判了脸数 / 模型类型 / pass 开关 /
`pipeline_size` / bbox 重叠，**唯独没判「模型是否支持动态 batch」**
（7.2 的硬条件里有这一条）。当前基础模型是 `static-batch=1`，因此
**7.1 与 7.2 两条 batch 路线在本模型下都不具备解冻条件**；当时强行开闸
不只是「无收益」，而是会触发 CUDA 非法内存访问。

**已修并复测（P1 安全修复）**：`batch_eligibility()` 补上必填的
`batch_backend_available`（与 7.2 同名同义、不设默认值），`Models` 新增
`swapper_supports_dynamic_batch()` 显式触发 session 懒加载，
`run_swapper_batched` 把 `_get_swapper_session()` 提前到静态 batch 检查
之前作为第二道防线。同一 3 秒 / 75 帧片段临时开闸复测：
`done: 75 frame(s)`、无 CUDA 错误、对闸门关闭的逐脸基线
**PSNR=`inf`**——误开闸门现在会在判定阶段干净回退，不再进入 N>1 的
`io_binding`。闸门仍保持 `False`（这是模型能力限制，不是代码缺陷）。
要继续这个方向，唯一前提是换成支持动态 batch 的 swapper 模型。

#### 本轮仍未完成

- 4.1 / 4.3 解冻：**不是待执行，而是被模型阻断**。受控实验证明
  `inswapper_128.fp16.onnx` 为 `static-batch=1`；当时的 CUDA 700 崩溃
  已通过 P1 安全修复消除（误开闸门现在干净回退），但两个闸门仍为
  `False`。要真正解冻仍需先换支持动态 batch 的 swapper 模型。
- 5b：`torchcodec` 未安装。nsys 显示 H2D memcpy 仅 0.181s / 36s，
  GPU 常驻解码的收益上限 <1%，性价比低。
- GPU / VRAM 曲线：同机 4 个外部进程持续占约 5.6 GiB，**无法隔离，
  属环境限制而非待执行项**。

适用范围：headless CLI 的 `video` / `swap` 路径。文中代码位置以函数名定位，
括号里的行号来自 2026-11-05 的代码快照；阶段 6 / 7 及后续改动后请勿依赖
行号。术语沿用 [CONTEXT.md](../CONTEXT.md) 的约定：中文行文，领域术语保留
英文原词。

---

## 1. 目标与判定标准

**目标**是提高吞吐（帧/秒）与 GPU SM 利用率。**VRAM 占用上升是并发的副产物，
不是优化目标本身**；单纯把显存池调大不算提速。

| 指标 | 采集方式 | 建议达标线 |
|---|---|---|
| 总用时 | `cli.py::main()` 打印的 `elapsed: HH:MM:SS.s` | 相对基线下降 |
| 平均处理速度 | `elapsed:` 行附带的 `(avg N fps)`；分母是编码循环实际帧数 | 随 worker 数上升；实测 w2 4.1 → w4 4.4 fps 已趋平，见「首批真机数据」 |
| 分段耗时 | `video --profile` 打印的 `decode/swap/encode` 三段 | 定位瓶颈，不与总用时相加 |
| GPU 利用率 | `nvidia-smi dmon -s um -d 1`（`-s u,m` 非法，见 3.4） | 峰值 >70%，不再周期性归零；**须先隔离同机其它 GPU 进程，否则读数不可归因** |
| VRAM 峰值 | `nvidia-smi --query-gpu=memory.used --format=csv -l 1` 或 dmon 的 `fb` 列 | 随并发度线性上升，无 OOM；同机负载会抬高基线，须隔离后读 |
| 输出正确性 | `ffprobe`、`elapsed`、前若干帧 PSNR | 与 `workers=1` 一致 |

**硬约束**：不得破坏 PTS / VFR 时序、音画同步、逐帧像素结果。`headless.py::_encode_video`
的注释与 `tests/test_timing.py` 就是这个契约的守门人，任何改动都要能过它们。

---

## 2. 现状与瓶颈

| # | 瓶颈 | 位置 |
|---|---|---|
| B1 | 解码（PyAV，CPU）、GPU 推理、libx264 编码全在**同一个线程逐帧串行** | `headless.py::_encode_video` |
| B2 | `Shared` 模式下每次 ORT 调用前做一次阻塞式 `syncvec.cpu()`；每帧结束还有 `result.detach().cpu().numpy()`。**11-06 nsys 修正**：`syncvec_drain` 仅 0.323s / 225 次、D2H memcpy 仅 0.062s，**这条不是主要瓶颈**；真正的 host stall 是 `cudaStreamSynchronize`（13.41s / 2618 次，占 CUDA API 时间 75.6%），集中在逐脸 `ort_run_swapper` | `Models.py::_should_drain_syncvec`、`headless.py::_swap_frame` |
| B3 | ORT session 单份 `Shared`；CUDA EP 用 `arena_extend_strategy=kSameAsRequested` 按需分配 VRAM | `Models.py::__init__`、`Models.py::_cuda_ep_provider_options` |
| B4 | 换脸按脸逐个 ORT Run，没有跨帧 batch；生产路径上 `run_swapper_batched` 只在 polyphase 内部使用（阶段 4 的冻结骨架另有 3 处调用，闸门关闭、不进帧路径） | `VideoManager.py::_swap_video_inner`、`Models.py::run_swapper_batched` |
| B5 | `ThreadsSlider` 是零读取方的遗留参数；`_model_session_mode` 原本没有 setter | `parameters.py`、`Models.py` |
| B6 | 可选 GPU 解码依赖被注释掉，但 `VideoManager.py::_swap_video_inner` 已接受 CUDA tensor 输入 | `requirements.txt`、`VideoManager.py` |
| B7 | 逐脸 `run_recognize` 的结果在 headless 且无 `--target-face` 时完全没有读取方 | `VideoManager.py::_swap_video_inner`（已由 7.1 的替代优化跳过，`--force-recognize` 可回滚） |

以上是**改造前的基线现状**（`workers=1`、单份 `Shared` session），不是当前
流水线：阶段 1 的 `--workers > 1` 已让多帧并行，11-06 真机矩阵里 `swap`
段累计耗时随 worker 数上升（`64.61s → 99.30s → 180.07s`），说明瓶颈更像
Python 层调度 / 逐脸 ORT 调用，而非单纯的「GPU 空闲」。另外「GPU 和
VRAM 都很低」这一旧判断在本机无法证实：同机还有 4 个其它 GPU 进程，
`nvidia-smi` 读数被污染，见「首批真机数据」结论 5。

---

## 3. 阶段 0：建立基线（先做，不可跳过）

### 3.1 已完成

`cli.py::main()` 用 `time.monotonic()` 在命令入口计时，成功结束后打印
`elapsed: HH:MM:SS.s`；仅 `swap` / `video` 打印，失败、`Ctrl+C`、异常退出均不打印。
格式化函数在 `cli.py::_format_elapsed()`，测试在 `tests/test_cli.py::test_format_elapsed()`。

### 3.2 已实现：平均处理速度

输出为 `elapsed: 00:02:34.6 (avg 30.6 fps)`，格式化函数
`cli.py::_format_avg_fps()`，测试在 `tests/test_cli.py`。

帧数没有直接用 `report["frames"]`——它来自 `stream.frames`，容器不声明时
可能为 0 或只是按 duration 估算。`_encode_video` 在编码循环里实际计数，
经 `report["processed_frames"]` 传给 `main()`，与 `elapsed` 共用同一个时间
基准。`done:` 行仍用原来的元数据值，输出语义不变。零帧、零耗时、缺帧数
一律不输出 fps；`swap` 没有帧数概念，只打印 `elapsed:`。

### 3.3 已实现：分段计时开关 `--profile`

`video --profile` 结束时打印
`profile: decode=Xs swap=Ys encode=Zs`：

- `decode`：取帧、`to_ndarray` 与 PTS 计算，在 `decoded()` 里累计。串行路径
  的换脸也在 `decoded()` 内，但它归 `swap` 段，计时在换脸前后分别结清。
- `swap`：`timed_swap()` 包住 `_swap_frame()`，串行与并行路径共用同一个
  计时入口，`ordered_swap_frames` 的签名不变。worker 线程并发记账由一把
  `threading.Lock` 保护。
- `encode`：`VideoFrame.from_ndarray` + `encode` + `mux`，在编码主循环里累计。

`--workers > 1` 时三段在不同线程上重叠，之和会大于总用时，只用来定位瓶颈
在哪一段，不要相加。默认关闭，关闭时只多一次布尔判断。需要更细粒度时用
项目已有的 `_nvtx.py` 配合 `nsys profile -t cuda,nvtx` 看 stream 空隙。

### 3.4 基线矩阵

用同一段 30–60 秒的视频，避免每次都跑几千帧：

```bash
nvidia-smi dmon -s um -d 1 > gpu-baseline.log &
for w in 1 2 4; do
  python -m faceswapper.cli --models ./models video \
    --source /workspace/data/picture/face10.jpg \
    --target /workspace/data/video/target.mp4 \
    --output /tmp/out-w$w.mp4 \
    --workers $w --preset ultrafast --crf 23 >> bench.log 2>&1
done
```

记录 `elapsed`、`avg fps`、GPU 利用率曲线、VRAM 峰值、`ffprobe` 帧数与时长；
用 `ffmpeg -i out-w1.mp4 -i out-w4.mp4 -filter_complex psnr -f null -` 比对像素。

上面这段已封装成 `tools/bench_gpu.py`，**默认 dry-run**（只打印命令，不碰
GPU、不写文件），加 `--run` 才执行：

```bash
python tools/bench_gpu.py --source face.jpg --target in.mp4
python tools/bench_gpu.py --source face.jpg --target in.mp4 --run \
    --workers 1,2,4 --monitor-gpu --check-frames --psnr
```

命令构造与结果解析在 stdlib-only 的 `faceswapper/bench_plan.py`，由
`tests/test_bench_plan.py` 离线验证（含 dry-run 端到端）；脚本只负责编排
子进程。它会自动解析 `elapsed` / `avg fps` / `profile` 三段、`ffprobe`
帧数、与 `workers=1` 基线的加速比和 PSNR，结果写到
`<output-dir>/bench_report.json`。`--extra-flag` 可把
`--encode-thread` / `--encoder nvenc` 等开关透传给每次运行，便于做 A/B。

**通过标准**：拿到可复现的数字，并以 `workers=1` 为回归基线，确认其它
worker 数与它逐像素一致。改造前的输出未留存、无法直接比对，故以 w1 自身
两轮可复现加 w2/w4 对 w1 的 PSNR 作为等价验收。

> **核对状态（2026-11-06 三次核对）**：`--run` 累计执行 10 秒片段两轮
> （250 帧）、`w1` GPU 采样复测一轮、30 秒 / 750 帧正式基线一轮、以及
> 阶段 2 / 3 / 6 / 7 的四组 A/B。所有 w2/w4 对 w1 的逐帧 PSNR 均为
> `inf`，**「通过标准」已达成，正式基线已建立**，见「首批 / 第二轮
> 真机数据」。唯一无法补齐的是隔离负载后的 GPU / VRAM 曲线：同机
> 4 个外部 GPU 进程无法移除，属环境限制。

---

## 4. 阶段 1：`--workers N`（已实现，吞吐与 PSNR 已真机验证）

### 4.1 实现清单

| 层 | 改动 |
|---|---|
| `cli.py` | `video` 新增 `--workers`（默认 1，内部 clamp 到 1–8），传入 `VideoOptions` |
| `headless.py` | `VideoOptions.workers`；`workers>1` 时调用 `Models.set_model_session_mode("Per-Thread")` |
| `Models.py` | 恢复 `set_model_session_mode()`，激活已有的 thread-local session 分发 |
| `headless.py` | `_encode_video` 拆出有序生成器 `decoded()`；调用 `frame_order.ordered_swap_frames()` |
| `frame_order.py` | 有序并行窗口（stdlib-only），可脱离 GPU 栈做行为测试 |
| `tests/test_frame_order.py` | 乱序完成仍保序、every-nth 透传、异常传播、线程池收尾 |
| `tests/test_cli.py` | `test_video_workers_flag()`、`test_parallel_path_wiring()`（AST 接线检查） |

机制：N 个 worker 线程各自懒加载一份 ORT session 与一条 CUDA stream（仓库里
`_sessions_tls`、`VideoManager.py::_get_worker_stream`、
`Models.py::_should_drain_syncvec` 的 stream 判断早已为此准备）；解码与编码留在主线程；
`ordered_swap_frames`（`faceswapper/frame_order.py`，stdlib-only）用
`ThreadPoolExecutor` 加最多 `2N` 帧的有序窗口，**只有队首
future 完成才交给编码**，因此 PTS、VFR 时序、音画对齐与串行路径一致。`--every-nth`
跳过的帧不提交、原样透传。

### 4.2 验证步骤（2026-11-06 更新：1 / 2 / 4 已实测）

1. 用 3.4 的矩阵跑 `workers=1/2/4`，记录 `elapsed`、GPU 利用率、VRAM。
   **部分完成**：`elapsed` / `avg fps` / `--profile` 已采两轮；GPU 利用率与
   VRAM 曲线在修好 dmon 参数后已复采，但同机还有 4 个其它 GPU 进程，
   读数无法归因，需隔离重采。数字见「首批真机数据」。
2. `ffprobe -count_frames` 比对帧数与时长，检查音画偏移是否与 `workers=1` 一致。
   **已完成**：三个用例均 250 帧 / 10.000s / 25 fps / `audio=yes`；w2、w4
   对 w1 的逐帧 PSNR 均为 `inf`（逐像素一致）。PTS / VFR 与音画偏移的
   细项仍需在 VFR 片段上单独核对。
3. 观察启动开销：每个 worker 首次会懒建 3 份 session 并做一次 cuDNN EXHAUSTIVE
   benchmark，短视频可能得不偿失。**已量化**：同一路径下 10 秒片段
   `3.1 / 4.1 / 4.4 fps`，30 秒片段 `3.3 / 5.0 / 5.4 fps`，长片段 fps
   全面更高，说明启动开销在短片段里摊薄不掉，正式基线必须用 30–60 秒。
4. **重点看 GIL**：`_swap_frame` 里仍有大量 Python 层调度。若 N 增大而 `avg fps`
   不涨甚至下降，说明瓶颈不在 GPU kernel，应转阶段 2 / 3，而不是继续加 N。
   **已命中**：10s 片段 w2 4.1 → w4 4.4 fps，30s 片段 5.0 → 5.4 fps，
   收益递减。结论是不要再加 worker。后续 nsys 已把 host stall 定位到
   `cudaStreamSynchronize` / 逐脸 ORT Run，阶段 2 / 3 的 A/B 也无收益，
   剩余对症方向只有 7.1 单帧多脸 batch。
5. 已知限制：`workers>1` 时不要打开 `OrientAutoSwitch`（跨帧共享状态会竞态，默认关闭）；
   当前只作用于 `video`，`swap` 仍串行。

**回滚**：不传 `--workers` 即走原路径，无需回滚代码。

---

## 5. 阶段 2：编码流水线化（已实现，真机 A/B 为负收益）

### 5.1 目标

把 `out_stream.encode()` 与 `output.mux()` 移出主线程，让 CPU 编码与 GPU 推理解耦。
GPU 利用率曲线上的周期性归零主要来自这里。VRAM 不变，可与阶段 1 叠加。

### 5.2 实现说明

新增 `faceswapper/encoder.py`（与 `model_store.py` 一样刻意 stdlib-only），
把编码 worker 从重运行时里剥出来：`av` 模块由参数注入，因此队列保序能用
假容器做**行为测试**，不必读源码字符串。

- `encoder.py::encoder_worker()` 在编码线程内完成全部 `av.open` /
  `add_stream` / `encode` / `mux` / `close`，按 FIFO 消费
  `(display_index, rgb, pts)`，遇 `None` sentinel 先 flush 再退出。
  `pts_scale` 用本线程的 `out_stream.time_base` 计算，公式与串行路径一致。
- `encoder.py::submit_frames()` 用 `put(timeout=1.0)` 重试而非死等，每轮先检查
  `errors` / `stop_event`；编码线程一旦失败，主线程立刻抛 `EncoderFailure`
  （`HeadlessError` 子类，退出码 2），不会卡在满队列上。
- `headless.py::_encode_via_thread()` 建 `Queue(maxsize=2*workers)` 与非 daemon
  编码线程；正常情况下投完 sentinel 后 `join()`，异常路径不投 sentinel、
  改设 `stop_event` 让 worker 在 `get(timeout=1.0)` 超时后自行收尾；
  `finally` 保证 join，使 `output.close()` 先于音频 mux。
- `headless.py::_encode_video` 只把 `output` 的创建挪进分支：默认路径的
  `decoded()`、PTS 算法与编码主循环**一行未改**；`--encode-thread` 路径共用
  同一份 `decoded()` 生成器，没有复制第二套 PTS 逻辑。
- `progress` 回调移到编码线程按序触发；`--profile` 的 `encode` 段在编码线程
  累加，仍走同一把 `threading.Lock`。
- CLI：`video --encode-thread`（store_true，默认关）；不传即回滚到原路径，
  无需改代码。`VideoManager` / `Models` 未改动。

离线测试 `faceswapper/tests/test_encoder.py` 覆盖 7 帧 FIFO 保序、flush /
close、PTS 与 time_base、编码异常回传、满队列重试、stop_event 无 sentinel
退出。`tests/test_timing.py` 的 PTS 归一化 / VFR / offset 算术保持不变。
**真机比对已完成**：`--encode-thread` 在 10 秒 / 250 帧片段上为
64.8s / 3.9 fps，对照无开关基线 60.9s / 4.1 fps，属**负收益**；输出与
基线逐像素一致（PSNR=`inf`），帧数与音画均正常。见「第二轮真机数据」。
`workers=4 + --encode-thread` 的组合未单独测——既然单开关已负收益，
叠加不会反转结论。

### 5.3 风险与对策

| 风险 | 对策 |
|---|---|
| PyAV 对象跨线程 | 所有 `av.open` / `add_stream` / `encode` / `mux` / `close` 都在编码线程内 |
| 异常时死锁 | sentinel + `stop_event` + `get(timeout=1.0)` + `try/finally` |
| 内存膨胀 | `Queue(maxsize=2*workers)` 有界 |
| 进度回调时序 | 移到编码线程按序触发，语义与串行一致 |

### 5.4 预期（已由真机 A/B 证实为负收益）

GPU 利用率的周期性归零消失；`avg fps` 提升幅度取决于原编码耗时占比。配合
`--preset ultrafast` 时收益会更小，因为编码本来就不贵。11-06 的基线矩阵
给出了间接证据：`encode` 三段分别只有 `3.99s / 5.33s / 6.08s`，约占总用时
的 5–8%，而 `swap` 占绝大部分。**这支持下面的保留意见**：独立编码线程
可争取的空间很小。真机 A/B 也确认了这一点：`--encode-thread` 为
64.8s / 3.9 fps，比基线 60.9s / 4.1 fps 更慢，PSNR=`inf`。
**结论是保持默认关闭，不要开启**；nsys 也显示 `encode` 段在 3 秒片段
仅 1.19s，独立线程的调度开销已超过它能省下的时间。

**保留意见**：`--workers N` 的 2N 有序窗口在阶段 1 已让「主线程编码队首帧时
后续帧并行推进」，编码与推理本就部分重叠；窗口满时主线程阻塞在
`f.result()`，独立编码线程的增量只剩「提交新帧不再被 encode 阻塞」。另外
PyAV 的 `encode()` 是否在 Cython 层释放 GIL 未验证——若不释放，编码线程
反而会卡住主线程的 Python 循环，存在负收益可能（真机 A/B 已测到负收益，
与这个推测一致，但机制本身仍未证实）。因此该开关默认关闭，
应先用 `--profile` 看 `encode=` 占比、再决定是否开启，不要默认启用。

---

## 6. 阶段 3：减少每帧同步与显存分配开销（已实现保守版，真机 A/B 无收益）

### 6.1 同步（只搬位置，不删同步）

**红线完整保留**：没有删 `syncvec.cpu()`，没有删 `wait_stream`，没有动
`user_compute_stream`。`tests/test_cli.py::test_sync_red_lines_preserved` 用
AST 静态检查这三处仍在，防止后续改动误删。

- `headless.py::_swap_frame(rgb, *, defer=False)`：`defer=True` 时把 CUDA
  tensor 原样返回，不做 `.cpu().numpy()`；默认 `False` 时行为逐字不变。
- `headless.py::_finalize_frame()`：tensor 走 `detach().cpu().numpy()` 再转
  uint8（与原 `_swap_frame` 一致）；已是 numpy 的帧（`--every-nth` 跳过的）
  只保证 contiguous，不额外 astype，避免改变像素。
- `_encode_video` 在 `frames` 构造完之后包一层生成器统一 finalize，两条
  编码路径都拿到 numpy。顺序继续由 `VideoManager.swap_video` 里已有的
  `default_stream().wait_stream(worker_stream)` 保证：消费侧的 `.cpu()` 跑在
  默认 stream 上，一定读到已写完的缓冲。
- `defer_d2h` 只在 `workers > 1` 时生效（`parallel` 为假时自动退回原行为）。

**与原步骤的偏离**：文档 6.1 原本要求把 D2H 推迟到**编码线程**并用 CUDA
event 显式排序、把 stream/event 随帧传递。本次改为推迟到**帧的消费侧
（主线程）**，复用已有的 `wait_stream`，不新增 event。理由：`encoder.py` 是
stdlib-only，传入 torch stream 会破坏这条约束；而 `wait_stream` 本身已经提供
了顺序保证，再引入 event 属于重复排序。代价是收益可能低于预期——D2H 仍在
主线程串行，跨 worker 的默认 stream 隐式屏障也还在。

> **已核对（2026-11-06 三次核对）**：`nsys profile -t cuda,nvtx` 已完成，
> 结论见「第二轮真机数据」——host stall 主体是 `cudaStreamSynchronize`
> （占 CUDA API 时间 75.6%），而 `syncvec_drain` 仅 0.323s、D2H memcpy
> 仅 0.062s，**B2 把 `syncvec.cpu()` 当主要瓶颈的假设不成立**。
> `--defer-d2h --arena-strategy pow2` 的 A/B 为 63.0s / 4.0 fps（基线
> 60.9s / 4.1 fps），输出 PSNR=`inf`；`workers=1/2/4` 的输出比对已在
> 10 秒与 30 秒矩阵完成（均 `inf`）。开启 `--defer-d2h` 时 D2H 耗时不
> 计入 `--profile` 三段。开关默认关闭，实测无收益，不建议开启。

### 6.2 显存分配（做成可配置，默认不变）

- `Models.py::set_arena_strategy()` 在任何 session 懒加载前设置策略，只接受
  ORT 认识的两个值，拼错直接报错而不是静默回退。
- `_cuda_ep_provider_options()` 与 restorer 的内联 `cuda_options` 都改读
  `self._arena_strategy`，不再硬编码；默认仍是 `kSameAsRequested`。
- CLI `--arena-strategy same|pow2`（`add_common`，swap / video / inspect 共有），
  `_SwapperBase.__init__` 负责把它映射成 ORT 的值。
- `_make_session_options()` 的 `intra_op_num_threads` 逻辑无需改动：现状已是
  `Per-Thread=1`、`Shared=max(2, min(8, cores//2))`，CPU 侧不会超订。

`pow2` 只减少重复 `cudaMalloc`、让 VRAM 数字变大，**不是提速手段**；
nsys 实测 `cudaMalloc` + `cudaFree` 合计仅 0.074s，A/B 也确认无收益。

---

## 7. 阶段 4：batch 推理（收益最大、风险最高）

### 7.1 单帧多脸 batch —— 代码已编写，**闸门关闭、未启用**

文档中列的判据「paste-back 是否只作用于该脸 bbox」**成立**：
`VideoManager.py::swap_core` 的 paste_back 段只读写
`img[0:3, top:bottom, left:right]`（约 917–925 行）。但深入读代码后发现有
五条更硬的阻断条件，判据成立并不足以支撑实现：

1. **`run_swapper_batched` 的 N 不是脸数。** 它是 polyphase 的 phase 数
   （`dim²`）：`_polyphase_pass_v1` 把一张脸按 stride 抽稀成 N 个 128×128 tile
   一次推理。而 `swap_core` 里 `dim` 恒为 1（约 505–517 行），`n_phases=1`，
   `run_swapper_batched` 会走 `if n == 1` 直接退回 `run_swapper`——**这个
   batch 分支当前根本没被调用**。把它改造成「多脸 batch」等于重定义接口，
   不是「复用已有能力」。
2. **`pipeline_size` 逐脸不同。** 它由每张脸的 `scale` 算出（约 524–545 行），
   范围 128–swap_size，且随 `SwapperTypeTextSel` 变化；batch 输入要求统一的
   `(N, 3, 128, 128)`。
3. **`swap_core` 是 438 行单体函数**（约 499–927 行），串着输入 warp、latent、
   HighFidelity 2-pass、Strength 迭代、色彩匹配、mask 组合、restorer、
   paste-back。要 batch 必须把它拆成「prepare → batch infer → per-face finish」
   三段，或复制约 65 行 paste-back 形成两套实现。本地无法验证像素等价。
4. **输入阶段就有数据依赖，回填补不回来。** `pipeline_input_face` 是从**当前
   img** warp 出来的（约 571–579 行）；逐脸路径中后一张脸读到的是前一张已写回
   的像素（925 行）。预批量准备时所有脸读到的都是原始 img，bbox 重叠时结果
   与原来不同。文档第 3 步「按原顺序回填」只处理了写侧，读侧的差异无法补回。
5. **模型不通用。** hyperswap / inswapper-256 / 512 走不同 session 与归一化
   （约 505–520 行、`Models.py` 的 hyperswap pass），不能塞进同一个 128 的
   ONNX batch。
6. **基础模型不支持动态 batch（11-06 受控实验新发现，硬阻断）。**
   `inswapper_128.fp16.onnx` 的 `target` 输入第 0 维是常量 1，加载日志标
   `static-batch=1`。受控开闸实验中 N=3 的 batch 被 ONNX 以
   `Got: 3 Expected: 1` 拒绝，`run_swapper_batched` 异常回退后又触发
   `CUDA failure 700: an illegal memory access`，进程崩溃、无输出文件。
   这一条在 7.2 的硬条件里列了，但 7.1 的 `batch_eligibility()` 当时
   漏判——它只检查脸数 / 模型类型 / pass 开关 / `pipeline_size` / bbox
   重叠。**已修**：`batch_eligibility()` 增加必填的
   `batch_backend_available`，由 `Models.swapper_supports_dynamic_batch()`
   显式探测后传入；`run_swapper_batched` 另把 session 懒加载提前，作为
   第二道防线。详见「第二轮真机数据」的 7.1 受控实验与复测。

这六条决定了 7.1 **不能直接挂进主路径**：硬接会产出「看起来省了 ORT
调用、但多人脸/重叠人脸像素不对」的结果，在当前模型下更会直接触发
CUDA 非法内存访问。按「代码先编写进入、不启用」的要求，实现已落地
但闸门必须保持 `False`；解冻的前置条件已变成**换支持动态 batch 的
swapper 模型**，而不只是建 PSNR 基线。详见「第二轮真机数据」的
7.1 受控实验。

#### 已编写、未挂载的实现

- `face_batch.py`（stdlib-only）：`FaceCandidate` + `batch_eligibility()`，
  把上面五条变成可判定的条件——至少 2 张脸、仅 `inswapper-128`、无
  `StrengthSwitch` / `HighFidelitySwitch`、所有 `pipeline_size == 128`、
  bbox 两两不重叠。`tests/test_face_batch.py` 做真行为测试。
- `VideoManager.swap_core`（签名加 `batch_slot=None`）：默认 `None`，现有
  逐脸路径只多一次 bool 判断。传入时有两种模式：
  - `prepare`：在 `poly_pass` 之前把对齐裁剪 `(H,W,3) 0-1` 写进 batch
    buffer、记录 bbox 与 `pipeline_size`，随后 `return None`，**不写回 img**；
  - `finish`：把 `poly_pass` 换成一个只读 `batch_out[index]` 的闭包，让
    `*255 / clamp` 和 735 行之后的 mask / restorer / paste_back **全部复用
    原有约 190 行**，不复制第二套实现。
- `VideoManager.swap_core_batched()`：prepare 全部脸 → 一次
  `run_swapper_batched` → 按原顺序 finish 全部脸。判定不通过时（bbox 重叠、
  模型不符等）因为 prepare 没写回过 img，可以干净地退回逐脸路径。
- `VideoManager._face_bbox()`：与 paste_back 段同一套 bbox 算法，供 prepare
  提前判定重叠；**改动 paste_back 的 bbox 计算必须同步这里**。
- 闸门 `EXPERIMENTAL_FACE_BATCH = False`：`_swap_video_inner` 的 headless
  分支有一个**唯一**的接入点，被
  `if EXPERIMENTAL_FACE_BATCH and not targets` 包住，闸门关闭时短路到原逐脸
  循环，输出逐字不变。`--target-face` 的相似度筛选也不走 batch——
  `batch_eligibility` 新增 `target_face_filtering` 参数做第二道保险
  （`swap_core_batched` 传入 `bool(self.target_face_embeddings)`）。
  `tests/test_face_batch.py::test_batch_path_is_disabled` 用 AST 钉住
  「闸门为 False、**每一个** `swap_core_batched` 调用都在闸门分支内、
  `batch_slot` 默认 None」这三条。

#### 解冻步骤（受控实验已证当前模型不可解冻，以下步骤须先换模型）

> **执行前必读**：11-06 已在 `static-batch=1` 模型上按第 2 步临时开闸
> 做过受控实验，当时结果是 ONNX 维度拒绝 + `CUDA failure 700` 崩溃。
> 该安全缺陷已修（复测确认误开闸门干净回退、PSNR=`inf`），但
> **换支持动态 batch 的 swapper 模型之前不要重跑这套步骤**，闸门保持
> `False`。

1. 准备多人脸视频样本，分别覆盖 bbox 不重叠与重叠两种情况，记录
   `workers=1` 逐脸路径的输出作为 PSNR 基线。
2. 把 `EXPERIMENTAL_FACE_BATCH` 置 `True` 即可启用已就位的接入点；
   `swap_core_batched` 内部仍会在 prepare 之后按 `batch_eligibility()`
   判定，不适用时干净回退逐脸路径，不需要改 `_swap_video_inner`。
3. 比对帧数、PTS、音画、逐帧 PSNR；重叠 bbox 的样本必须走回退分支。
4. 用 `nvidia-smi dmon` 与 `elapsed` / `avg fps` 确认 VRAM 与吞吐收益；
   单人脸视频应自动回退，不得有回归。
5. 任一步不满足即把闸门改回 `False`，无需回滚其它代码。

#### 已实现的安全替代：跳过没有读取方的 `run_recognize`

同为「减少每帧 ORT 调用」，有一条逻辑上可证明等价、改动很小的路径已落地：

- `_swap_video_inner` 原本对每张检测脸都调一次 `run_recognize`（约 405–409 行）。
- 但 headless video 的分支（约 411–427 行）把所有脸都换成同一个
  `force_source_embedding`，只有在指定了 `--target-face`、需要相似度筛选时
  才会读 `face_emb`。图片 / GUI 的 `found_faces` 匹配路径才逐脸读 embedding。
- 因此「headless video + 未指定 `--target-face`」时，每帧 N 次 ArcFace 推理
  的结果全是死数据。跳过它们**不改变任何像素**，只是省掉 N 次 ORT 调用。

判定抽到 stdlib-only 的 `recognize.py::should_skip_recognize()`，由
`tests/test_recognize.py` 做行为测试；`VideoManager` 里 `face_emb=None` 占位，
下游不会读到它。`--force-recognize` 是回滚 / A/B 开关，开启后逐字走旧路径。
`tests/test_cli.py::test_recognize_wiring` 用 AST 守住接线与旧路径未删。
**nsys 间接验证**：3 秒 / 75 帧轨迹里 `:run_recognize` 全程仅 1 次
（1.56s，即启动阶段的源特征提取），逐帧逐脸调用确实被跳过。收益数字
未单独做 A/B，但 nsys 显示每帧 3 张脸，意味着省下约 225 次 ArcFace 推理。

### 7.2 跨帧 batch —— 判定 / 编排 / 推理骨架已落地，主流水线未挂载

把 N 个 worker 的帧拼成一个 batch 一次 ORT Run，需要模型输入支持动态 batch
（`_swapper_batch_unsupported` 已有探测），并重构整个帧流水线。会与 PTS 保序、
`--every-nth`、可变人脸数冲突，**放在最后**，且必须做成独立开关与独立测试。

按「代码先编写进入、不启用」的要求，已落地可离线验证的部分：

- `faceswapper/cross_frame_batch.py`（stdlib-only）：
  `FrameCandidate`、`cross_frame_batch_eligibility()`、`group_frames()`、
  闸门 `EXPERIMENTAL_CROSS_FRAME_BATCH = False` 与
  `resolve_cross_frame_batch()`（闸门 + 判定）。
- **硬条件**：`workers >= 2`、`every_nth == 1`、仅 `inswapper-128`、
  关闭 `StrengthSwitch` / `HighFidelitySwitch`、动态 batch 可用、每帧人脸数
  恒为 1、`pipeline_size` 恒为 128、`index` 连续、**PTS 严格递增**。
  **11-06 补充**：其中「动态 batch 可用」在当前
  `inswapper_128.fp16.onnx`（`static-batch=1`）上恒不满足，因此 7.2 与
  7.1 一样被模型能力阻断，见「第二轮真机数据」的 7.1 受控实验。
  最后两条是 VFR 时序的守门条件：batch 只能切，不能排序。
- `group_frames()` 只做有序切片，保留不足一组的尾部；分组不允许排序。
- `tests/test_cross_frame_batch.py`：7 项行为测试，覆盖上述全部条件、闸门、
  分组保序，以及 AST 检查「`headless.py` / `frame_order.py` /
  `VideoManager.py` 都没有调用被闸门保护的入口」。
- **主流水线完全未挂载**：没有新 CLI flag，也没有改动任何帧路径。

#### 推理侧骨架（已编写，闸门关闭）

一次跨帧 ORT Run 复用 7.1 的 `swap_core(batch_slot=...)` prepare/finish 钩子，
不复制任何 postprocess：

- `cross_frame_batch.py::run_cross_frame_batch()`：纯编排函数，把
  「闸门 → 判定 → prepare 全部帧 → 一次 infer → 按序 finish」抽成可注入
  回调的状态机。闸门关闭或判定不通过时**一次 prepare 都不调用**，直接
  fallback，保证输入帧未被改动。`tests/test_cross_frame_batch.py` 用假回调
  断言调用顺序、闸门/回退不碰帧、prepare 失败不会触发 infer/finish。
- `VideoManager.cross_frame_batch_swap()`：把真实回调绑进来——prepare 用
  `swap_core(mode="prepare")` 收集各帧 128×128 输入，infer 调一次
  `run_swapper_batched`，finish 按组内顺序用 `mode="finish"` 走原有 mask /
  restorer / paste_back。各帧 img 互不相干，因此没有 7.1 的 bbox 重叠问题。
  闸门关闭时由该方法补一次逐脸 fallback，输出与基线一致。
- **主流水线仍未挂载**：`_swap_video_inner`、`headless.py`、
  `frame_order.py` 都不调用它，也没有新 CLI flag。AST 测试守住这条。

**仍未验证**：GPU 侧前向与回填本身、逐帧 PSNR、VFR/音画时序、以及省下
ORT 调用后的实际收益——这些必须在 GPU 环境做（环境现已具备，缺的是执行）。
解冻步骤（截至 2026-11-06 三次核对全部未执行）：

1. 有 GPU 环境后，用多人脸 + 单人险混合片段建立 `workers=1` 的逐帧 PSNR 基线。
2. 在 `frame_order.ordered_swap_frames` 的消费侧，按 `resolve_cross_frame_batch`
   的结果把连续帧分组；不适用或闸门关闭时逐字走原路径。
3. 每组复用 7.1 的 prepare 钩子收集输入 → 一次 `run_swapper_batched` →
   按组内顺序 finish 回填。
4. 比对帧数、PTS、音画、PSNR；VFR 片段必须确认 PTS 未被重排。
5. 任一步不满足即把闸门改回 `False`，无需回滚其它代码。

---

## 8. 阶段 5：GPU 解码 / NVENC 编码（5a 真机不可用，5b 缺可选依赖）

### 8.1 GPU 常驻解码（代码已编写，实验性，默认关闭）

依据：`requirements.txt` 里 `torchcodec` / `pynvvideocodec` 是注释掉的可选项；
`VideoManager.py::_swap_video_inner` 已接受 CUDA tensor 输入并跳过 CPU 上传，
说明 GPU 常驻解码是预留路径。

- `decoder.py`（stdlib-only）：`resolve_decoder(requested, probe)` 与
  `frame_fields(raw)`。`pyav` 分支**完全不调用** probe，默认路径既不 import
  torchcodec 也不碰 CUDA；`gpu` 探测失败抛 `HeadlessError`，不静默回退。
  `tests/test_decoder.py` 覆盖探测成功/失败/异常、未知 backend、以及 dict /
  属性两种 frame 布局（刻意避开 `bool(tensor)` 歧义）。
- `VideoSwapper._gpu_decode_probe()`：检查 `torchcodec` 是否安装、
  `VideoDecoder.__init__` 是否接受 `device=`、`torch.cuda.is_available()`。
- `VideoSwapper._decoded_gpu()`：用 `VideoDecoder(path, device="cuda")` 产出
  CUDA tensor，交给 `_swap_frame`，省掉 `torch.from_numpy(...).to('cuda')`。
  首帧绝对 PTS 由该生成器记录，`resolved_first_pts()` 保证 GPU 与 PyAV 两条
  路径的音频 mux 偏移都能恢复。
- **适配层刻意不猜**：frame 缺 `data`/`frame`、没有 PTS、或 PTS 是小数
  （多半是秒而非 ticks）时一律明确报错，绝不静默用等间隔兜底——那会悄悄
  破坏 VFR 与音画同步。`_normalize_gpu_rgb()` 兼容 CHW/HWC、float 0-1 /
  uint8，非 tensor 直接报错。
- CLI `video --decoder pyav|gpu`，默认 `pyav`；`_encode_video` 按
  `DecoderPlan` 选择帧源，`decoded()` 的 PyAV 路径一行未改。

**真机未验证**：`torchcodec` **未安装**（`importlib.util.find_spec` 返回
`None`），因此 frame 布局与 PTS 单位、VFR 时序、音画同步、省掉上传后的
收益一项都没能核对。GPU 硬件已具备，缺的是这个可选依赖；装上后再按
8.1 的适配层约定核对，不传开关即回原路径。

### 8.2 NVENC 编码开关（已实现，默认关闭）

- 新增 `encoder_probe.py`（stdlib-only）：`resolve_encoder()` 接收注入的
  `codec_factory`（真机传 `av.codec.Codec`），因此「探测 → 选 codec → 映射
  参数」整条链路能脱离 PyAV / GPU 做行为测试。
  `tests/test_encoder_probe.py` 覆盖 libx264 不探测、NVENC 可用/不可用、
  六个 preset 映射、未知 encoder 报错。
- `libx264`（默认）分支**完全不调用** `codec_factory`，也不访问 `av.codec`
  （`_encode_video` 传的是惰性 lambda），默认路径逐字不变。
- 显式 `--encoder nvenc` 时先探测 `h264_nvenc`；不可用直接抛
  `HeadlessError` 并给出修复提示，**不静默回退**（否则用户会以为在用硬件编码）。
- `encoder.py::encoder_worker()` 新增 `codec` / `stream_options` 两个带默认值
  的参数；`headless.py` 的默认编码路径与 `--encode-thread` 路径共用同一个
  `EncoderPlan`，不会出现两条路径编码器不一致。
- 参数映射：`crf` → `cq`（配合 `rc=vbr`），x264 的
  `ultrafast..veryslow` → NVENC 的 `p1..p7`。**语义不等价**：CRF 与 CQ 数值
  范围相近但率控行为不同，p1..p7 也不是 x264 档位的一一对应；旧 FFmpeg
  构建可能只认 `fast/medium/slow`，真机报未知 preset 应回退 libx264。
- NVENC 是独立硬件单元，**不提高 CUDA SM 利用率**，只解除 CPU 编码瓶颈；
  PyAV 目前仍吃 CPU `rgb24` frame，所以并没有省掉那次 D2H。

**真机已验证（不可用）**：`h264_nvenc` 虽在 `ffmpeg -encoders` 中可见，
但 `avcodec_open2("h264_nvenc", {'rc':'vbr','cq':'23','preset':'p1'})`
在本容器抛 `av.error.PermissionError: [Errno 1] Operation not permitted`，
FFmpeg 附带日志称驱动需 570+（本机 580，故更可能是 NVENC 设备权限 /
驱动 API 访问受限）。CLI 明确报错、**没有静默回退**，符合本节设计。
因此 `p1..p7` 是否被接受、画质 / 体积对比、NVENC 单元占用均无从核对，
详见「第二轮真机数据」的 5a 小节。

---

## 9. VRAM 专项（避免走弯路）

| 做法 | 是否真提速 | 说明 |
|---|---|---|
| `Per-Thread` session × N（阶段 1） | 是 | 并发的必要条件，VRAM 约 `N × 3 份 session` |
| 更大 batch（阶段 4） | 是 | kernel 效率提升，VRAM 随 batch 上升 |
| `--backend trt` 加 TensorRT 10 | 是（单帧延迟） | `trt_max_workspace_size=4GiB` 会占 VRAM；需装 `libnvinfer.so.10` |
| `arena_extend_strategy=kNextPowerOfTwo` | 大概率否 | 只是预分配策略；已做成 `--arena-strategy pow2`，默认关 |
| 改 `ThreadsSlider` | 否 | 零读取方的遗留参数 |
| 盲目 `--workers 8` | 否 | GIL、启动开销、OOM 风险 |

---

## 10. 优先级与执行顺序

> **完成度说明（2026-11-06 五次核对，含 7.1 受控实验与 P1 修复）**：下表直接标出
> 各阶段的**实测收益与真机状态**。11 项均已给出结论：8 项有实测数据，
> 4.1 / 4.3 被 `static-batch=1` 模型阻断，5b 缺可选依赖。代码与离线
> 测试的逐项状态见开头「完成情况核对」表。nsys 把瓶颈锁定在逐脸 ORT
> Run 与 stream 同步，但唯一对症的 batch 路线被模型能力卡住，
> **后续优先级已改为「先解决模型能力」，不再按并发或主机侧开销排**。

| 顺序 | 阶段 | 实测收益 | 工作量 | 风险 | 真机状态 |
|---|---|---|---|---|---|
| 0 | 基线 + `avg fps` + `--profile` | 无（但决定后续） | 已完成 | 极低 | **已完成** |
| 1 | `--workers N` | **1.50–1.63×**（30s 基线） | 已完成 | 中（GIL 已命中） | **基本完成** |
| 2 | 编码独立线程 | **负收益**（4.1 → 3.9 fps） | 已完成 | 中 | **已验证，不启用** |
| 3 | 延迟 D2H / arena 策略 | **无收益**（4.1 → 4.0 fps） | 已完成 | 低 | **已验证，不启用** |
| 4.1 | 单帧多脸 batch | 受控开闸曾触发 CUDA 700；已修为干净回退 | 代码已编写、闸门关闭 | 高（模型限制） | **模型阻断**；误开已安全（PSNR=`inf`） |
| 4.2 | 跳过无用的逐脸 recognize | 中（每帧省 N 次 ORT） | 已完成 | 极低 | **间接验证**（nsys） |
| 4.3 | 跨帧 batch | — | 代码已编写、未挂载 | 高 | **模型阻断**（放最后） |
| 5a | NVENC 编码开关 | 环境不支持 | 已完成 | 中 | **已验证不可用** |
| 5b | GPU 常驻解码 | 未测 | 已完成 | 中 | **未做**（缺 `torchcodec`） |
| 6 | detector 输出缓冲复用 | **无收益**（4.1 → 4.0 fps） | 已完成 | 低 | **已验证，PSNR=inf** |
| 7 | 帧路径零拷贝归一化 | **无收益**（4.1 → 3.9 fps） | 已完成 | 低 | **已验证，PSNR=inf** |

建议的**后续顺序**（三次核对 + 7.1 受控实验后重排）：阶段 2 / 3 / 6 / 7
已由 A/B 证伪、5a 环境不可用，**不再投入**；`--workers` 也已到收益递减区。
唯一剩余的前置条件是**换用支持动态 batch 的 swapper 模型**，之后才谈得上
`7.1 解冻 → 4.3 跨帧 batch`；5b 装 torchcodec 收益上限 <1%，排最后。

理由是 nsys 把瓶颈锁定在逐脸 `ort_run_swapper`（225 次 / 11.77s）与
`cudaStreamSynchronize`（占 CUDA API 时间 75.6%），只有减少 ORT 调用
次数才对症；而编码、D2H、显存分配、主机侧 memcpy 合计不足总用时 1%。
但当前 `inswapper_128.fp16.onnx` 的 batch 维锁死为 1，受控开闸实验
已证明会触发 `CUDA failure 700`，所以 7.1 的五步解冻流程在换模型之前
**不得执行**。GPU / VRAM 曲线因同机外部负载无法隔离，不再列为待办。

---

## 11. 每阶段通用验收清单

> **核对状态（2026-11-06 三次核对）**：`elapsed` / `avg fps` / `--profile`
> 三段、`ffprobe` 帧数与时长、与 `workers=1` 的 PSNR 已在 10 秒与 30 秒
> 片段上真机执行（均 `inf`）；阶段 2 / 3 / 6 / 7 的「不传即回滚」与输出
> 一致性也经 A/B + PSNR 核对；nsys host stall 已定位。**仍未核对**：
> GPU 利用率与 VRAM 曲线（同机 4 个外部进程无法隔离，属环境限制）、
> 中断 / 损坏输入 / OOM 的退出码、VFR / 音画偏移细项。离线可覆盖的
> 部分（PTS / VFR 算术、flag 默认值、AST 接线）已有测试，但不能替代
> GPU 实测。

- `elapsed` / `avg fps` / `nvidia-smi` 利用率与 VRAM
- `ffprobe` 帧数、时长、PTS 与音画同步
- 与 `workers=1` 的 PSNR
- 中断、损坏输入、OOM 时的退出码与清理行为
- 不传新参数时输出与基线完全一致（保证可回滚）

---

## 12. 不改代码、现在就能做的措施

> **11-06 五次核对**：以下仍是最有效的现成配置。但真机 A/B 已确认
> **除 `--workers 2` 外，阶段 2 / 3 / 6 / 7 的开关都没有收益**，不要再
> 叠加 `--encode-thread` / `--defer-d2h` / `--reuse-detect-buffers` /
> `--zero-copy-frames`；NVENC 在本环境还不可用。

```bash
python -m faceswapper.cli --models ./models video \
    --source /workspace/data/picture/face10.jpg \
    --target-face /workspace/data/video/target_face.jpg \
    --target /workspace/data/video/target.mp4 \
    --output /workspace/data/output/target_10.mp4 \
    --workers 2 --preset ultrafast --crf 23
```

- `--workers 2` 起步即可：实测 w2 → w4 只从 4.1 涨到 4.4 fps，**继续加
  worker 收益几乎为零**，不必「观察后再加」。观察利用率请用
  `nvidia-smi dmon -s um -d 1`，并先隔离同机其它 GPU 进程，否则读数
  不可归因（见「首批真机数据」结论 5）。
- `--preset ultrafast` 只减少 CPU 编码耗时，**不改变换脸计算量**。
- 关闭 `--restorer` / `--occluder` / `--xseg` / `--face-parser` / `--color-match`
  可减少每帧的 ORT 调用。
- `--detect-input-size` 保持 640；调小会漏检小脸。
- `--every-nth` 是抽样验证，不是加速。`--target-face` 只影响人脸筛选：
  **未指定时**逐脸识别 embedding 已由阶段 4.2 默认跳过（像素不变，省每帧
  N 次 ArcFace 推理）；**指定时**要按相似度筛人，embedding 照常计算。
  `--force-recognize` 可强制回到旧路径。
- `--backend trt` 需先安装 TensorRT 10 运行时（提供 `libnvinfer.so.10`）。

### `--preset` 对文件体积的实测影响

同一片源、同为 `crf=18`，仅改 `--preset`：

| 样本 | preset | 大小 | 视频码率 | H.264 profile |
|---|---|---:|---:|---|
| 早期输出 | `medium`（默认） | 84 MB | 3.41 Mbps | High |
| 后期输出 | `ultrafast` | 233 MB | 9.86 Mbps | Constrained Baseline |

`ultrafast` 关闭 `cabac=1` → `cabac=0`、`bframes=3` → `0`、`8x8dct=1` → `0`、
`subme=7` → `0`、`mbtree=1` → `0`、`aq=1` → `0`，压缩效率下降，CRF 为维持
恒定质量只能多用约 2.9 倍比特。**画质并不更好，只是用体积换编码速度。**
在意体积时务必同时加 `--crf 23`（CRF 每 +6 大致减半码率），或改用 `--preset fast`。
编码参数可直接从输出文件读出：

```bash
strings -a out.mp4 | grep -m1 x264
ffprobe -v error -select_streams v:0 -show_entries stream=profile,bit_rate \
    -of default=noprint_wrappers=1 out.mp4
```

---

## 13. 相关代码索引

| 主题 | 位置 |
|---|---|
| 视频主循环、PTS、编码 | `headless.py::_encode_video` |
| 有序并行窗口 | `frame_order.py::ordered_swap_frames`；`tests/test_frame_order.py` 行为测试 |
| 编码线程与队列保序 | `encoder.py::encoder_worker`、`submit_frames`；`headless.py::_encode_via_thread` |
| 延迟 D2H / finalize | `headless.py::_swap_frame`、`_finalize_frame` |
| arena 策略 | `Models.py::set_arena_strategy`、`_cuda_ep_provider_options` |
| 红线防回归测试 | `tests/test_cli.py::test_sync_red_lines_preserved` |
| recognize 跳过策略 | `recognize.py::should_skip_recognize`；`VideoManager.py::_swap_video_inner`、`tests/test_recognize.py` |
| 7.1 冻结依据 | `VideoManager.py::swap_core` 的输入 warp（约 571）与 paste_back（约 917–925） |
| 7.1 实验实现 | `face_batch.py::batch_eligibility`；`VideoManager.py::swap_core_batched`、`_face_bbox`、`swap_core(batch_slot=...)` |
| 编码器选择与 NVENC 探测 | `encoder_probe.py::resolve_encoder`；`encoder.py::encoder_worker(codec=...)`、`headless.py::_encode_video` |
| GPU 解码探测与帧适配 | `decoder.py::resolve_decoder`、`frame_fields`；`headless.py::_gpu_decode_probe`、`_decoded_gpu` |
| 跨帧 batch 判定 / 分组 / 编排 | `cross_frame_batch.py::cross_frame_batch_eligibility`、`resolve_cross_frame_batch`、`group_frames`、`run_cross_frame_batch`；`VideoManager.py::cross_frame_batch_swap` |
| GPU 基线验收 | `tools/bench_gpu.py`；`faceswapper/bench_plan.py::build_cases`、`parse_metrics`、`parse_psnr`、`summarize`、`build_psnr_command`、`build_gpu_monitor_command`（后两者 11-06 修过参数） |
| 离线测试一键 runner | `faceswapper/test_runner.py`（stdlib-only 的发现 / 分类 / 编排）、`faceswapper/tests/__main__.py`；`tests/test_suite_runner.py` 做真行为测试 |
| 帧解码与逐脸循环 | `headless.py::decoded()`（局部生成器）、`VideoManager.py::_swap_video_inner` |
| session 共享策略 | `Models.py::_get_model_session`、`set_model_session_mode` |
| 后端偏好（cuda 不碰 TRT） | `Models.py::set_backend_preference_for_all`、`headless.py::_SwapperBase.__init__` |
| 同步判断、CUDA EP 选项、session 选项 | `Models.py::_should_drain_syncvec`、`_cuda_ep_provider_options`、`_make_session_options` |
| detector 输出缓冲 shape / 复用（阶段 6） | `detect_buffers.py::output_buffer_specs`、`resolve_reuse_buffers`、`OutputBufferCache`；`Models.py::set_reuse_detect_buffers`、`_get_retinaface_output_buffers`、`_detect_retinaface_inner` |
| 帧零拷贝判定 / 字节预算（阶段 7） | `frame_upload.py::normalize_plan`、`frame_bytes`、`estimate_saved_bytes`；`headless.py::_frame_to_uint8`、`_finalize_frame`；`VideoManager.py::_swap_video_inner`；`tests/test_frame_upload.py` |
| batch 推理 | `Models.py::run_swapper_batched`、`swapper_supports_dynamic_batch()`；`VideoManager.py::swap_core`、`swap_core_batched`、`cross_frame_batch_swap`；`face_batch.py::batch_eligibility`（必填 `batch_backend_available`） |
| CLI 参数 | `cli.py::build_parser`：阶段 0–3（`--workers` / `--profile` / `--encode-thread` / `--defer-d2h` / `--arena-strategy` / `--force-recognize`）、阶段 5（`--encoder` / `--decoder`）、阶段 6–7（`--reuse-detect-buffers` / `--zero-copy-frames`），以及 `--preset` / `--crf` / `--every-nth` / `--backend` |
| 完成计时 / avg fps | `cli.py::main`、`_format_elapsed`、`_format_avg_fps` |
| 分段计时 | `headless.py::_encode_video`、`cli.py::_format_profile` |
| 离线测试（18 个模块，2026-11-06 五次核对 `--strict` 18/18 PASS） | `tests/test_cli.py`、`test_timing.py`、`test_wiring.py`、`test_hyperswap.py`、`test_model_store.py`、`test_install.py`（已改为环境无关断言）、`test_target_match.py`、`test_encoder.py`、`test_encoder_probe.py`、`test_decoder.py`、`test_frame_order.py`、`test_recognize.py`、`test_face_batch.py`（已补动态 batch 判据与接线 AST 测试）、`test_cross_frame_batch.py`、`test_bench_plan.py`（已补 PSNR / dmon 测试）、`test_detect_buffers.py`、`test_frame_upload.py`、`test_suite_runner.py` |
| 命名与 GUI 遗留 | `docs/adr/0001-keep-gui-era-parameter-naming.md`、`CONTEXT.md` |

---

## 14. 阶段 6：detector 输出缓冲复用

> 本节排在「相关代码索引」之后，是为了不改动 2026-11-05 快照里已有的章节号
> 与交叉引用。实现日期 2026-11-06。

### 14.1 依据

`Models._detect_retinaface_inner` 每帧为 9 个 `io_binding` 输出各调一次
`torch.empty`：3 个 stride × score/bbox/kps。shape 只由
`(input_height, input_width, stride)` 决定，而前两个尺寸在一次 video 里恒定
（除非改 `--detect-input-size`）。同一函数里的输入 letterbox 缓冲
（`_get_retinaface_input_buffer`）与 anchor centers
（`_get_retinaface_gpu_anchors`）早就按 worker 缓存了，输出缓冲是这条链上
唯一还每帧重新分配的一环。

阶段 3 的 6.2 只把 arena 策略做成可配置，没有消除分配本身。阶段 6 补这一环：
每帧省 9 次 `torch.empty` 与 9 次缓冲回收，代价是常驻 9 个小 tensor。
**这是分配器开销优化，不是吞吐瓶颈的主解法**；VRAM 数字几乎不变。

### 14.2 实现

| 层 | 改动 |
|---|---|
| `detect_buffers.py` | stdlib-only：`BufferSpec`、`output_buffer_specs()`（算 9 个 shape，顺序与 `bind_output` 的三个切片一致）、`resolve_reuse_buffers()`（闸门）、`OutputBufferCache`（按 `(h, w, strides)` 缓存，`alloc` 注入） |
| `Models.py` | `set_reuse_detect_buffers()`；`_get_retinaface_output_buffers()` 把 `torch.empty` 注入缓存并挂在 `_retinaface_buffers_tls`；`_detect_retinaface_inner` 加一个 `if self._reuse_detect_buffers` 分支 |
| `headless.py` | `SwapOptions.reuse_detect_buffers`；`_SwapperBase.__init__` 调 setter 并传入 `options.backend` |
| `cli.py` | `add_common` 新增 `--reuse-detect-buffers`（`swap` / `video` / `inspect` 共有），两处 Options 构造接线 |
| `tests/` | `test_detect_buffers.py` 7 项真行为测试；`test_cli.py` 加 flag / 默认值 / compose / AST 闸门与 fallback 检查 |

默认 `False`。关闭时 `_detect_retinaface_inner` 的 else 分支逐字保留原来每帧
9 次 `torch.empty` 的逻辑（只多一层缩进），
`test_detect_buffer_reuse_wiring` 用 AST 钉住「闸门常量为 False、复用分支唯一、
else 里仍有至少 3 次 `torch.empty`」。

`resolve_reuse_buffers()` 只接受显式开启 + `backend == "cuda"`；TRT-EP 的
`io_binding` 输出行为未经真机核对，不猜、不静默套用（与 `decoder.py` 探测
失败不静默回退是同一条原则）。

### 14.3 为什么按 worker 分开是安全的

每组缓冲挂在 `threading.local` 的 `_retinaface_buffers_tls` 上，与已有的输入
缓冲同级。不同 worker 的 CUDA stream 并发推进，共享同一组缓冲会让 ORT 的写入
互相踩；同 worker 内则是串行的：`rf_postprocess` 在 worker 自己的 stream 上
读完这一帧，下一帧的 ORT 写入才排到它后面。缓存 key 含 `(h, w, strides)`，
改 `--detect-input-size` 或换 stride 组合会整组重建，不会拿旧 shape 去绑定。

一个已知的边界：`delete_models()` 不清这些缓存——`threading.local` 的属主是
各 worker 线程，主线程够不到别人的 slot。这与既有的输入 letterbox 缓冲完全
一致，9 个缓冲合计约 1 MB，CLI 单次运行随进程退出即释放；长驻进程调用
`delete_models()` 时它们仍常驻，属于既有行为，没有为阶段 6 新增特殊的清理
语义。

### 14.4 真机验收清单（第 1–2 步已验证逐像素一致，3–6 步未执行）

1. `workers=1` 且不带开关，跑一段多人脸片段，存逐帧 PSNR 基线。
2. 加 `--reuse-detect-buffers` 重跑，比对帧数、PTS、音画与逐帧 PSNR；
   **必须逐像素一致**，否则说明 ORT 没有完整覆写缓冲，立即把闸门改回 False。
   **已完成**：10 秒 / 250 帧 `workers=2` 下 PSNR=`inf`、250 帧与音画
   正常，说明 ORT 完整覆写复用缓冲；吞吐 4.0 fps vs 基线 4.1，无收益。
3. `--workers 2/4` 下重复第 2 步，确认 per-thread 隔离没有串味。
4. `--detect-input-size 320/480/640` 各跑一次，确认 key 变化时 shape 正确。
5. 用 `elapsed` / `avg fps` / `nvidia-smi dmon` 记录收益；预期很小，若没有
   可测量提升就保持默认关闭，不必为了「用了新开关」而改默认值。
6. `--backend trt` 下确认 `resolve_reuse_buffers` 自动关闭（离线已测，真机
   再确认一次没有走复用分支）。

**回滚**：不传 `--reuse-detect-buffers` 即回到原路径；即使代码层改默认值，
`resolve_reuse_buffers` 也只放行显式开启 + CUDA EP。

---

## 15. 阶段 7：帧路径零拷贝归一化

> 与第 14 节同理，本节续在「相关代码索引」之后，不改动 2026-11-05 快照的
> 章节号与交叉引用。实现日期 2026-11-06。

### 15.1 依据

帧路径上共有 5 处 `astype(uint8)`：`VideoManager._swap_video_inner` 的 CPU
numpy H2D 分支，以及 `headless.py` 的 `source_embedding()`、
`_detect_face_slots()`、`ImageSwapper.swap()`、`_finalize_frame()`。
`numpy.astype` 默认 `copy=True`，**即使输入已是 uint8 C contiguous 也会整帧
复制**。PyAV 的 `to_ndarray(format="rgb24")` 与 D2H 结果通常已经满足这两个
条件，于是每帧白付一次全帧 memcpy。

| 量 | 1080p RGB uint8 |
|---|---|
| 单帧字节 | `1920 × 1080 × 3 = 6,220,800 B`（约 5.9 MiB） |
| H2D + D2H 两侧 | 约 12.4 MB/帧 |
| 60 秒 × 30 fps | 约 22 GB 纯主机内存带宽 |

这是主机侧带宽与 CPU 时间开销，与 GPU SM 利用率无关，VRAM 数字完全不变。
因此**不要把它当成 GPU 利用率瓶颈的主解法**；只有 nsys 显示 H2D / D2H 确实
排得上号时，才值得考虑改默认值。

### 15.2 实现

| 层 | 改动 |
|---|---|
| `frame_upload.py` | stdlib-only：`FramePlan`、`frame_bytes()`、`normalize_plan()`（闸门 + dtype + layout 判定）、`estimate_saved_bytes()`、`format_saved_bytes()`；不 import numpy |
| `headless.py` | `_frame_to_uint8()` helper、`SwapOptions.zero_copy_frames`、4 处帧路径改调 helper；`_SwapperBase.__init__` 把闸门传给 `self.vm` |
| `VideoManager.py` | `self.zero_copy_frames = False`；`_swap_video_inner` 的 CPU numpy H2D 分支按 plan 复用或转换 |
| `cli.py` | `add_common` 新增 `--zero-copy-frames`（`swap` / `video` / `inspect` 共有），两处 Options 构造接线 |
| `tests/` | `test_frame_upload.py` 6 项真行为测试；`test_cli.py` 加 flag / compose / defaults / AST 闸门与两侧 fallback 检查 |

`normalize_plan()` 的四种结果：

| 条件 | `needs_copy` | `bytes_saved` | `reason` |
|---|---|---|---|
| 闸门关闭 | True | 0 | `gate-closed` |
| dtype 不是 uint8 | True | 0 | `dtype-...-needs-conversion` |
| 非 C contiguous | True | 0 | `not-c-contiguous` |
| uint8 + C contiguous | False | `frame_bytes(h, w)` | `already-uint8-contiguous` |

只有最后一种省字节；其余仍走 `astype` 或
`np.ascontiguousarray(..., dtype=np.uint8)`，数值等价。闸门关闭时的 `astype`
表达式逐字保留，`test_zero_copy_frames_wiring` 用 AST 同时钉住 `SwapOptions` /
`VideoManager` 两个默认值、H2D 与 helper 两条 fallback，以及
`self.vm.zero_copy_frames` 接线。

### 15.3 别名与生命周期

跳过复制返回的是与输入共享内存的数组，安全性依赖三条：调用方不原地写、
只交给同步的 `torch.from_numpy(...).to("cuda")` / `VideoFrame.from_ndarray`、
PyAV 每帧 `to_ndarray` 都产出新数组（无跨帧复用）。`--workers > 1` 下每帧数组
各自独立，没有跨 worker 共享；复用只发生在本帧内。`--decoder gpu` 的
torchcodec 路径直接给 CUDA tensor，根本不走这段 CPU numpy H2D 分支，因此
不受影响。

### 15.4 真机验收清单（第 1–2 步已验证逐像素一致，3–6 步未执行）

1. `workers=1` 且不带开关，跑一段 1080p 视频，存逐帧 PSNR 基线。
2. 加 `--zero-copy-frames` 重跑，比对帧数、PTS、音画与逐帧 PSNR；**必须
   逐像素一致**，否则说明某条路径的 dtype / layout 判定与实际内存不符。
   **已完成**：10 秒 / 250 帧 `workers=2` 下 PSNR=`inf`、帧数与音画正常，
   dtype / layout 判定与实际内存一致；吞吐 3.9 fps vs 基线 4.1，无收益。
3. 分别在 `--decoder pyav` 与 `gpu`、`--encoder libx264` 与 `nvenc` 下重复，
   确认没有哪一侧的帧其实是 float / 非 contiguous 而被误判为可复用。
4. `swap` 图片路径同样跑一次（source / target / 输出写盘），确认
   `ImageSwapper.swap()` 的写出结果一致。
5. `--workers 2/4` 下重复第 2 步，确认没有跨帧别名。
6. 用 `elapsed` / `avg fps` / nsys 的 H2D / D2H 段记录收益；若不可测量，
   保持默认关闭，不必为了「用了新开关」而改默认值。

**回滚**：不传 `--zero-copy-frames` 即回到原路径。与阶段 6 不同，这里没有
ORT 输出覆写之类的第三方行为假设，两侧 `astype` fallback 由 AST 测试守住，
回滚成本为零。
