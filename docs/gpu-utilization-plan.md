# 提升 GPU 与 VRAM 利用率的实施方案

日期：2026-11-05（最近核对：2026-11-06）

## 状态

- 阶段 0 已实现：`elapsed:`、`avg fps`、`--profile` 分段计时，均有离线测试覆盖。
- 阶段 1 的 `--workers N` 已实现；保序调度器抽到 stdlib-only 的
  `frame_order.py`，由 `tests/test_frame_order.py` 用假 `swap_fn` 做真行为
  测试（乱序完成仍按解码顺序输出），**尚未做真实 GPU 视频验证**。
- 阶段 2 的 `--encode-thread` 已实现（默认关闭），队列保序、sentinel flush、
  异常传播与背压均有离线测试覆盖。
- 阶段 3 的 6.1 / 6.2 已实现（`--defer-d2h`、`--arena-strategy`，均默认关闭），
  红线同步由 AST 测试守住；**nsys 与真机 A/B 未做**。
- 阶段 4 的 7.1 代码已编写，接入点受闸门保护、默认短路
  （`EXPERIMENTAL_FACE_BATCH=False`）；需 GPU 环境做多人脸 PSNR 比对后才能
  解冻，见 7.1 的五条阻断条件与解冻步骤。另实现了一条等价且更安全的优化：
  跳过没有读取方的逐脸 `run_recognize`，`--force-recognize` 可回滚；这是全部
  改造中**唯一默认生效**的行为变化（不改变像素）。
- 阶段 5 的编码侧（`--encoder libx264|nvenc`）与解码侧
  （`--decoder pyav|gpu`）代码均已落地，两个默认值都不变；两者都需要真机
  核对第三方 API/参数。阶段 4 的 7.2 判定 / 分组 / 前向与回填编排均已落地，
  闸门关闭、主流水线未挂载，推理结果仍需真机逐帧 PSNR。
- 阶段 6 的 `--reuse-detect-buffers` 已落地（默认关闭、`trt` 后端自动关闭），
  shape 计算 / 闸门 / per-worker 缓存均有离线行为测试，默认帧路径逐字保留；
  ORT 是否每帧完整覆写复用缓冲、逐帧 PSNR 与实际收益仍需真机核对，见第 14 节。
- 阶段 7 的 `--zero-copy-frames` 已落地（默认关闭），frame dtype / layout 判定
  与字节预算抽到 stdlib-only 的 `frame_upload.py`，H2D 与 D2H 两侧的 `astype`
  fallback 由 AST 测试钉住；逐帧 PSNR、各解码路径实际 dtype / layout 与真实
  memcpy 带宽收益仍需真机核对，见第 15 节。
- 当前环境没有 GPU 栈，以上均为代码层实现，**未做任何真机性能验证**；
  3.4 的基线矩阵仍是后续所有阶段的前置条件。

### 完成情况核对（2026-11-06）

对照代码库逐项复查的结果：**代码层 11 个阶段 / 子项全部收口，真机验证 0 项
完成**。因此本文出现的「待验证步骤」「解冻步骤」「真机验收清单」截至目前
都还是未执行项，不是已完成项。

| 阶段 | 代码 | 离线测试 | 默认行为 | 真机验证 |
|---|---|---|---|---|
| 0 基线 / `avg fps` / `--profile` | 已完成 | PASS | 默认生效 | **未做**（3.4 矩阵待 `--run`） |
| 1 `--workers N` | 已完成 | PASS | `workers=1`（串行） | **未做** |
| 2 `--encode-thread` | 已完成 | PASS | 关闭 | **未做** |
| 3 `--defer-d2h` / `--arena-strategy` | 已完成（保守版） | PASS | 均关闭 | **未做**（含 nsys） |
| 4.1 单帧多脸 batch | 已编写、未挂载主路径 | PASS + AST 闸门 | `EXPERIMENTAL_FACE_BATCH=False` | **未做**（解冻前置） |
| 4.2 跳过逐脸 recognize | 已完成 | PASS | **默认生效**，`--force-recognize` 回滚 | **未做** |
| 4.3 跨帧 batch | 已编写、未挂载主路径 | PASS + AST 未挂载 | `EXPERIMENTAL_CROSS_FRAME_BATCH=False` | **未做** |
| 5a NVENC 编码 | 已完成 | PASS | 关闭（默认 `libx264`） | **未做** |
| 5b GPU 常驻解码 | 已完成 | PASS | 关闭（默认 `pyav`） | **未做** |
| 6 detector 输出缓冲复用 | 已完成 | PASS | 关闭，`trt` 自动关 | **未做**（第 14 节清单） |
| 7 帧路径零拷贝 | 已完成 | PASS | 关闭 | **未做**（第 15 节清单） |

核对依据：`faceswapper/tests/` 下 17 个离线模块，16 个 PASS；
`test_target_match` 因本机未装 numpy 报 `ModuleNotFoundError`（环境缺失，
非代码缺陷）；四个实验 / 优化闸门均为 `False`；`compileall` 与
`git diff --check` 通过。

三点需要特别留意：

1. **唯一默认改变行为的项是 4.2**。其余新开关全部默认关闭或闸门为 False，
   不传开关即回到基线路径；这也是本文所有回滚说明的前提。
2. **7.1 / 7.2 是「代码已编写但冻结」**，不是「已实现可用」。判定、编排、
   推理骨架齐备且离线测试通过，但主流水线没有挂载、没有 CLI flag，启用前
   必须先做逐帧 PSNR。
3. **真机侧仍是整张白纸**：`elapsed` / `avg fps` / GPU 与 VRAM 曲线、
   `ffprobe` 帧数与 PTS / VFR / 音画同步、与 `workers=1` 的逐帧 PSNR、
   nsys host stall、NVENC 与 torchcodec 的第三方 API 行为、以及 7.1 / 7.2
   解冻后的收益数字，一项都还没有采集。前置动作只有一条：
   `python tools/bench_gpu.py ... --run`。

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
| 平均处理速度 | `elapsed:` 行附带的 `(avg N fps)`；分母是编码循环实际帧数 | 随 worker 数单调上升 |
| 分段耗时 | `video --profile` 打印的 `decode/swap/encode` 三段 | 定位瓶颈，不与总用时相加 |
| GPU 利用率 | `nvidia-smi dmon -s u -d 1` | 峰值 >70%，不再周期性归零 |
| VRAM 峰值 | `nvidia-smi --query-gpu=memory.used --format=csv -l 1` | 随并发度线性上升，无 OOM |
| 输出正确性 | `ffprobe`、`elapsed`、前若干帧 PSNR | 与 `workers=1` 一致 |

**硬约束**：不得破坏 PTS / VFR 时序、音画同步、逐帧像素结果。`headless.py::_encode_video`
的注释与 `tests/test_timing.py` 就是这个契约的守门人，任何改动都要能过它们。

---

## 2. 现状与瓶颈

| # | 瓶颈 | 位置 |
|---|---|---|
| B1 | 解码（PyAV，CPU）、GPU 推理、libx264 编码全在**同一个线程逐帧串行** | `headless.py::_encode_video` |
| B2 | `Shared` 模式下每次 ORT 调用前做一次阻塞式 `syncvec.cpu()`；每帧结束还有 `result.detach().cpu().numpy()` | `Models.py::_should_drain_syncvec`、`headless.py::_swap_frame` |
| B3 | ORT session 单份 `Shared`；CUDA EP 用 `arena_extend_strategy=kSameAsRequested` 按需分配 VRAM | `Models.py::__init__`、`Models.py::_cuda_ep_provider_options` |
| B4 | 换脸按脸逐个 ORT Run，没有跨帧 batch；生产路径上 `run_swapper_batched` 只在 polyphase 内部使用（阶段 4 的冻结骨架另有 3 处调用，闸门关闭、不进帧路径） | `VideoManager.py::_swap_video_inner`、`Models.py::run_swapper_batched` |
| B5 | `ThreadsSlider` 是零读取方的遗留参数；`_model_session_mode` 原本没有 setter | `parameters.py`、`Models.py` |
| B6 | 可选 GPU 解码依赖被注释掉，但 `VideoManager.py::_swap_video_inner` 已接受 CUDA tensor 输入 | `requirements.txt`、`VideoManager.py` |
| B7 | 逐脸 `run_recognize` 的结果在 headless 且无 `--target-face` 时完全没有读取方 | `VideoManager.py::_swap_video_inner`（已由 7.1 的替代优化跳过，`--force-recognize` 可回滚） |

结论：任意时刻只有 1 帧、1 个 ORT Run、1 次编码在推进，GPU 与 CPU 阶段无法重叠。
这也是"GPU 和 VRAM 都很低"的根本原因，而不是参数没调满。

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
nvidia-smi dmon -s u,m -d 1 > gpu-baseline.log &
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

**通过标准**：拿到可复现的数字，且 `workers=1` 的输出与改造前完全一致（回归基线）。

> **核对状态（2026-11-06）**：脚本与 `bench_plan.py` 已落地、离线测试 PASS，
> 但 `--run` 从未在 GPU 环境执行，上面的「通过标准」尚未达成。本矩阵仍是
> 后续所有阶段的前置条件，不能因为代码齐了就跳过。

---

## 4. 阶段 1：`--workers N`（已实现，待真机验证）

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

### 4.2 待验证步骤（截至 2026-11-06 全部未执行）

1. 用 3.4 的矩阵跑 `workers=1/2/4`，记录 `elapsed`、GPU 利用率、VRAM。
2. `ffprobe -count_frames` 比对帧数与时长，检查音画偏移是否与 `workers=1` 一致。
3. 观察启动开销：每个 worker 首次会懒建 3 份 session 并做一次 cuDNN EXHAUSTIVE
   benchmark，短视频可能得不偿失。
4. **重点看 GIL**：`_swap_frame` 里仍有大量 Python 层调度。若 N 增大而 `avg fps`
   不涨甚至下降，说明瓶颈不在 GPU kernel，应转阶段 2 / 3，而不是继续加 N。
5. 已知限制：`workers>1` 时不要打开 `OrientAutoSwitch`（跨帧共享状态会竞态，默认关闭）；
   当前只作用于 `video`，`swap` 仍串行。

**回滚**：不传 `--workers` 即走原路径，无需回滚代码。

---

## 5. 阶段 2：编码流水线化（已实现，待真机验证）

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
**真机比对未做**（`workers=1` vs `workers=4 + --encode-thread` 的帧数、PTS、
音画、PSNR）。

### 5.3 风险与对策

| 风险 | 对策 |
|---|---|
| PyAV 对象跨线程 | 所有 `av.open` / `add_stream` / `encode` / `mux` / `close` 都在编码线程内 |
| 异常时死锁 | sentinel + `stop_event` + `get(timeout=1.0)` + `try/finally` |
| 内存膨胀 | `Queue(maxsize=2*workers)` 有界 |
| 进度回调时序 | 移到编码线程按序触发，语义与串行一致 |

### 5.4 预期（未验证）

GPU 利用率的周期性归零消失；`avg fps` 提升幅度取决于原编码耗时占比。配合
`--preset ultrafast` 时收益会更小，因为编码本来就不贵。

**保留意见**：`--workers N` 的 2N 有序窗口在阶段 1 已让「主线程编码队首帧时
后续帧并行推进」，编码与推理本就部分重叠；窗口满时主线程阻塞在
`f.result()`，独立编码线程的增量只剩「提交新帧不再被 encode 阻塞」。另外
PyAV 的 `encode()` 是否在 Cython 层释放 GIL 未验证——若不释放，编码线程
反而会卡住主线程的 Python 循环，存在负收益可能。因此该开关默认关闭，
应先用 `--profile` 看 `encode=` 占比、再决定是否开启，不要默认启用。

---

## 6. 阶段 3：减少每帧同步与显存分配开销（已实现保守版，待真机验证）

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

> **仍未做**：步骤 1 的 `nsys profile -t cuda,nvtx` host stall 确认、`--profile`
> 的 A/B、`workers=1/2/4` 的输出 PSNR 比对。开启 `--defer-d2h` 时，D2H 耗时
> 不计入 `--profile` 三段，三段之和会更小于总用时。开关默认关闭。

### 6.2 显存分配（做成可配置，默认不变）

- `Models.py::set_arena_strategy()` 在任何 session 懒加载前设置策略，只接受
  ORT 认识的两个值，拼错直接报错而不是静默回退。
- `_cuda_ep_provider_options()` 与 restorer 的内联 `cuda_options` 都改读
  `self._arena_strategy`，不再硬编码；默认仍是 `kSameAsRequested`。
- CLI `--arena-strategy same|pow2`（`add_common`，swap / video / inspect 共有），
  `_SwapperBase.__init__` 负责把它映射成 ORT 的值。
- `_make_session_options()` 的 `intra_op_num_threads` 逻辑无需改动：现状已是
  `Per-Thread=1`、`Shared=max(2, min(8, cores//2))`，CPU 侧不会超订。

`pow2` 只减少重复 `cudaMalloc`、让 VRAM 数字变大，**不是提速手段**；A/B 仍未做。

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

这五条决定了 7.1 **不能直接挂进主路径**：硬接会产出「看起来省了 ORT
调用、但多人脸/重叠人脸像素不对」的结果。按「代码先编写进入、不启用」的
要求，实现已落地但保持关闭。

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

#### 解冻步骤（需要 GPU 环境，截至 2026-11-06 未执行）

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
**真机收益未测**；单人脸视频省 1 次识别，多人脸省 N 次。

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
ORT 调用后的实际收益——这些必须在 GPU 环境做。解冻步骤（截至 2026-11-06
全部未执行）：

1. 有 GPU 环境后，用多人脸 + 单人险混合片段建立 `workers=1` 的逐帧 PSNR 基线。
2. 在 `frame_order.ordered_swap_frames` 的消费侧，按 `resolve_cross_frame_batch`
   的结果把连续帧分组；不适用或闸门关闭时逐字走原路径。
3. 每组复用 7.1 的 prepare 钩子收集输入 → 一次 `run_swapper_batched` →
   按组内顺序 finish 回填。
4. 比对帧数、PTS、音画、PSNR；VFR 片段必须确认 PTS 未被重排。
5. 任一步不满足即把闸门改回 `False`，无需回滚其它代码。

---

## 8. 阶段 5：GPU 解码 / NVENC 编码（两侧代码均已实现，待真机验证）

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

**真机未验证**：torchcodec 各版本的 frame 布局与 PTS 单位、VFR 时序、音画
同步、以及省掉上传后的实际收益。这些必须在 GPU 环境核对；不传开关即回原路径。

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

**真机验证未做**：NVENC 可用性、`p1..p7` 是否被当前 FFmpeg 接受、画质/体积
与 libx264 的对比、以及 `nvidia-smi` 里 NVENC 单元的占用。

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

> **完成度说明（2026-11-06 核对）**：下表「需真实 GPU 验证」列标的是
> **必要性**，不是完成状态；该列 11 项截至目前**全部未做**。代码与离线测试的
> 逐项状态见开头「完成情况核对」表。

| 顺序 | 阶段 | 预期收益 | 工作量 | 风险 | 需真实 GPU 验证 |
|---|---|---|---|---|---|
| 0 | 基线 + `avg fps` + `--profile` | 无（但决定后续） | 代码已完成 | 极低 | 是（3.4 矩阵待跑） |
| 1 | `--workers N`（已实现） | 中–高 | 已完成 | 中（GIL 可能限制） | **是，尚未做** |
| 2 | 编码独立线程 | 待实测（可能低于预期） | 代码已完成 | 中（线程 / 异常） | 是 |
| 3 | 延迟 D2H / arena 策略（保守版） | 未测（可能很小） | 代码已完成 | 低（未删同步） | 是 + nsys，未做 |
| 4.1 | 单帧多脸 batch | 未测 | 代码已完成、闸门关闭 | 高（启用后会改像素） | 是，解冻前必须 |
| 4.2 | 跳过无用的逐脸 recognize | 中（每帧省 N 次 ORT） | 代码已完成 | 极低 | 是 |
| 4.3 | 跨帧 batch（判定/编排/推理骨架） | — | 代码已完成、未挂载 | 高（需逐帧 PSNR） | 是 |
| 5a | NVENC 编码开关 | 中（只解 CPU 编码瓶颈） | 代码已完成 | 中（画质 / 驱动） | 是 |
| 5b | GPU 常驻解码（torchcodec） | 未测 | 代码已完成、默认关闭 | 中（frame/PTS 单位） | 是 |
| 6 | detector 输出缓冲复用 | 未测（省每帧 9 次分配） | 代码已完成、默认关闭 | 低（默认路径逐字保留） | 是 |
| 7 | 帧路径零拷贝归一化 | 未测（1080p 省约 12.4 MB/帧 memcpy） | 代码已完成、默认关闭 | 低（两侧 `astype` fallback 逐字保留） | 是 |

建议的**真机验证顺序**：`0 → 1 → 2 →（按 nsys 结果决定）3 → 阶段 4 的 7.1
单帧多脸 batch → 5`。0–3 的代码已落地，但仍要按这个顺序逐个实测，不要
因为代码齐了就跳过基线。跨帧 batch 放最后，除非前四步都测过仍不够。
阶段 6 不改像素、回滚成本为零，可作为 3 之后的低风险 A/B 插队做；它只省
分配器开销，不要当成吞吐瓶颈的主解法。阶段 7 同理，且只动主机侧 memcpy，
可与阶段 6 合成一组 A/B；若 nsys 显示瓶颈不在 H2D / D2H，收益可能接近零。

---

## 11. 每阶段通用验收清单

> **核对状态（2026-11-06）**：以下 5 条**均未在真机执行**。离线可覆盖的部分
> （PTS / VFR 算术、flag 默认值、AST 接线）已有测试，但不能替代 GPU 实测。

- `elapsed` / `avg fps` / `nvidia-smi` 利用率与 VRAM
- `ffprobe` 帧数、时长、PTS 与音画同步
- 与 `workers=1` 的 PSNR
- 中断、损坏输入、OOM 时的退出码与清理行为
- 不传新参数时输出与基线完全一致（保证可回滚）

---

## 12. 不改代码、现在就能做的措施

```bash
python -m faceswapper.cli --models ./models video \
    --source /workspace/data/picture/face10.jpg \
    --target-face /workspace/data/video/target_face.jpg \
    --target /workspace/data/video/target.mp4 \
    --output /workspace/data/output/target_10.mp4 \
    --workers 2 --preset ultrafast --crf 23
```

- `--workers 2` 起步，用 `nvidia-smi dmon -s u -d 1` 观察后再加。
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
| GPU 基线验收 | `tools/bench_gpu.py`；`faceswapper/bench_plan.py::build_cases`、`parse_metrics`、`summarize` |
| 帧解码与逐脸循环 | `headless.py::decoded()`（局部生成器）、`VideoManager.py::_swap_video_inner` |
| session 共享策略 | `Models.py::_get_model_session`、`set_model_session_mode` |
| 后端偏好（cuda 不碰 TRT） | `Models.py::set_backend_preference_for_all`、`headless.py::_SwapperBase.__init__` |
| 同步判断、CUDA EP 选项、session 选项 | `Models.py::_should_drain_syncvec`、`_cuda_ep_provider_options`、`_make_session_options` |
| detector 输出缓冲 shape / 复用（阶段 6） | `detect_buffers.py::output_buffer_specs`、`resolve_reuse_buffers`、`OutputBufferCache`；`Models.py::set_reuse_detect_buffers`、`_get_retinaface_output_buffers`、`_detect_retinaface_inner` |
| 帧零拷贝判定 / 字节预算（阶段 7） | `frame_upload.py::normalize_plan`、`frame_bytes`、`estimate_saved_bytes`；`headless.py::_frame_to_uint8`、`_finalize_frame`；`VideoManager.py::_swap_video_inner`；`tests/test_frame_upload.py` |
| batch 推理 | `Models.py::run_swapper_batched`、`VideoManager.py::swap_core` |
| CLI 参数 | `cli.py::build_parser`：阶段 0–3（`--workers` / `--profile` / `--encode-thread` / `--defer-d2h` / `--arena-strategy` / `--force-recognize`）、阶段 5（`--encoder` / `--decoder`）、阶段 6–7（`--reuse-detect-buffers` / `--zero-copy-frames`），以及 `--preset` / `--crf` / `--every-nth` / `--backend` |
| 完成计时 / avg fps | `cli.py::main`、`_format_elapsed`、`_format_avg_fps` |
| 分段计时 | `headless.py::_encode_video`、`cli.py::_format_profile` |
| 离线测试（17 个模块，2026-11-06 核对） | `tests/test_cli.py`、`test_timing.py`、`test_wiring.py`、`test_hyperswap.py`、`test_model_store.py`、`test_install.py`、`test_target_match.py`（本机缺 numpy）、`test_encoder.py`、`test_encoder_probe.py`、`test_decoder.py`、`test_frame_order.py`、`test_recognize.py`、`test_face_batch.py`、`test_cross_frame_batch.py`、`test_bench_plan.py`、`test_detect_buffers.py`、`test_frame_upload.py` |
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

### 14.4 真机验收清单（截至 2026-11-06 未执行）

1. `workers=1` 且不带开关，跑一段多人脸片段，存逐帧 PSNR 基线。
2. 加 `--reuse-detect-buffers` 重跑，比对帧数、PTS、音画与逐帧 PSNR；
   **必须逐像素一致**，否则说明 ORT 没有完整覆写缓冲，立即把闸门改回 False。
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

### 15.4 真机验收清单（截至 2026-11-06 未执行）

1. `workers=1` 且不带开关，跑一段 1080p 视频，存逐帧 PSNR 基线。
2. 加 `--zero-copy-frames` 重跑，比对帧数、PTS、音画与逐帧 PSNR；**必须
   逐像素一致**，否则说明某条路径的 dtype / layout 判定与实际内存不符。
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
