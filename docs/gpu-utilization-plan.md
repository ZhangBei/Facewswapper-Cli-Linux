# 提升 GPU 与 VRAM 利用率的实施方案

日期：2026-11-05

## 状态

- 阶段 0 的完成计时已实现（`elapsed:`），`avg fps` 与 `--profile` 待补。
- 阶段 1 的 `--workers N` 已实现并通过离线测试，**尚未做真实 GPU 视频验证**。
- 阶段 2–5 为待实施方案。

适用范围：headless CLI 的 `video` / `swap` 路径。文中代码位置以 2026-11-05
的代码快照为准，后续改动后请以函数名而非行号定位。术语沿用
[CONTEXT.md](../CONTEXT.md) 的约定：中文行文，领域术语保留英文原词。

---

## 1. 目标与判定标准

**目标**是提高吞吐（帧/秒）与 GPU SM 利用率。**VRAM 占用上升是并发的副产物，
不是优化目标本身**；单纯把显存池调大不算提速。

| 指标 | 采集方式 | 建议达标线 |
|---|---|---|
| 总用时 | `cli.py::main()` 打印的 `elapsed: HH:MM:SS.s` | 相对基线下降 |
| 平均处理速度 | 待补 `avg fps = frames / elapsed` | 随 worker 数单调上升 |
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
| B4 | 换脸按脸逐个 ORT Run，没有跨帧 batch；`run_swapper_batched` 只在 polyphase 内部使用 | `VideoManager.py::_swap_video_inner`、`Models.py::run_swapper_batched` |
| B5 | `ThreadsSlider` 是零读取方的遗留参数；`_model_session_mode` 原本没有 setter | `parameters.py`、`Models.py` |
| B6 | 可选 GPU 解码依赖被注释掉，但 `VideoManager.py::_swap_video_inner` 已接受 CUDA tensor 输入 | `requirements.txt`、`VideoManager.py` |

结论：任意时刻只有 1 帧、1 个 ORT Run、1 次编码在推进，GPU 与 CPU 阶段无法重叠。
这也是"GPU 和 VRAM 都很低"的根本原因，而不是参数没调满。

---

## 3. 阶段 0：建立基线（先做，不可跳过）

### 3.1 已完成

`cli.py::main()` 用 `time.monotonic()` 在命令入口计时，成功结束后打印
`elapsed: HH:MM:SS.s`；仅 `swap` / `video` 打印，失败、`Ctrl+C`、异常退出均不打印。
格式化函数在 `cli.py::_format_elapsed()`，测试在 `tests/test_cli.py::test_format_elapsed()`。

### 3.2 待补：平均处理速度

1. 在 `cli.py::main()` 里把 `report["frames"]` 传出来，或在 `_cmd_video` 内单独再计一次时。
2. 输出改为 `elapsed: 00:02:34.6 (avg 30.6 fps)`。
3. 在 `tests/test_cli.py` 加一个纯函数测试，覆盖 `frames / elapsed` 的除零与格式化。

### 3.3 待补：分段计时开关 `--profile`

1. 在 `headless.py::_encode_video` 的 `decoded()` 生成器、`_parallel_swap_frames`、
   编码循环三处各累计 `time.monotonic()` 耗时。
2. 结束时打印 `decode=Xs swap=Ys encode=Zs`，用来直接判定瓶颈在 GPU 推理还是 CPU 编码。
3. 需要更细粒度时用项目已有的 `_nvtx.py` 配合 `nsys profile -t cuda,nvtx` 看 stream 空隙。

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

**通过标准**：拿到可复现的数字，且 `workers=1` 的输出与改造前完全一致（回归基线）。

---

## 4. 阶段 1：`--workers N`（已实现，待真机验证）

### 4.1 实现清单

| 层 | 改动 |
|---|---|
| `cli.py` | `video` 新增 `--workers`（默认 1，内部 clamp 到 1–8），传入 `VideoOptions` |
| `headless.py` | `VideoOptions.workers`；`workers>1` 时调用 `Models.set_model_session_mode("Per-Thread")` |
| `Models.py` | 恢复 `set_model_session_mode()`，激活已有的 thread-local session 分发 |
| `headless.py` | `_encode_video` 拆出有序生成器 `decoded()`；新增 `_parallel_swap_frames()` |
| `tests/test_cli.py` | `test_video_workers_flag()`、`test_parallel_path_preserves_order()` |

机制：N 个 worker 线程各自懒加载一份 ORT session 与一条 CUDA stream（仓库里
`_sessions_tls`、`VideoManager.py::_get_worker_stream`、
`Models.py::_should_drain_syncvec` 的 stream 判断早已为此准备）；解码与编码留在主线程；
`_parallel_swap_frames` 用 `ThreadPoolExecutor` 加最多 `2N` 帧的有序窗口，**只有队首
future 完成才交给编码**，因此 PTS、VFR 时序、音画对齐与串行路径一致。`--every-nth`
跳过的帧不提交、原样透传。

### 4.2 待验证步骤

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

## 5. 阶段 2：编码流水线化（推荐下一步）

### 5.1 目标

把 `out_stream.encode()` 与 `output.mux()` 移出主线程，让 CPU 编码与 GPU 推理解耦。
GPU 利用率曲线上的周期性归零主要来自这里。VRAM 不变，可与阶段 1 叠加。

### 5.2 实现步骤

1. **新增编码线程辅助函数**（只改 `headless.py`，不动 `VideoManager` / `Models`）：

```python
def _encoder_worker(silent_path, width, height, fps, in_tb, pix_fmt,
                    crf, preset, queue, error_holder, on_progress, total):
    # PyAV 对象全部在本线程创建、使用、销毁，不跨线程
    output = av.open(silent_path, mode="w", format="mp4", ...)
    out_stream = output.add_stream("libx264", rate=Fraction(fps)...)
    ...
    while True:
        item = queue.get()
        if item is None:                       # sentinel
            for packet in out_stream.encode(None):
                output.mux(packet)
            break
        index, rgb, pts = item
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), "rgb24")
        frame.pts = ...; frame.time_base = ...
        for packet in out_stream.encode(frame):
            output.mux(packet)
        on_progress(index, total)              # 按序回调
    output.close()
```

2. **改造 `_encode_video` 主循环**
   - 保留 `decoded()` 生成器与 PTS 算法（阶段 1 已验证），一行都不改。
   - 新建 `queue.Queue(maxsize=2 * workers)`，启动一个非 daemon 编码线程。
   - 主循环把 `(index, rgb, pts)` 放入队列；`queue.put` 在队列满时自然阻塞，
     替代阶段 1 的窗口限流。
   - 循环结束投递 sentinel 并 `join()`；`first_pts` 由主线程算好直接返回，不依赖编码线程。
   - `finally` 里保证 sentinel 已投递、线程已 join，避免异常时挂死。

3. **异常传播**
   - 编码线程用 `try/except` 把异常写进 `error_holder`，`finally` 里 `output.close()`。
   - 主线程 `join()` 后若 `error_holder` 非空则 `raise`，保证退出码仍是 1 / 2。
   - `queue.get()` 加 `timeout=1.0` 并轮询 `stop_event`，防止主线程异常退出时编码线程永久阻塞。

4. **参数**：新增 `--encode-thread`（store_true，默认关）或并入 `--workers`
   （`workers>1` 自动启用）。建议默认关，与阶段 1 一样做成可开关。

5. **验证**
   - `tests/test_timing.py` 的 PTS 归一化 / VFR / offset 算术保持不变。
   - 新增队列保序单测：投 7 个 `(index, rgb, pts)` 加 sentinel，用假 `av` 桩断言编码顺序为 0..6。
   - 真机比对 `workers=1` 与 `workers=4 + --encode-thread` 的帧数、PTS、音画、PSNR。

### 5.3 风险与对策

| 风险 | 对策 |
|---|---|
| PyAV 对象跨线程 | 所有 `av.open` / `add_stream` / `encode` / `mux` / `close` 都在编码线程内 |
| 异常时死锁 | sentinel + `stop_event` + `get(timeout=1.0)` + `try/finally` |
| 内存膨胀 | `Queue(maxsize=2*workers)` 有界 |
| 进度回调时序 | 移到编码线程按序触发，语义与串行一致 |

### 5.4 预期

GPU 利用率的周期性归零消失；`avg fps` 提升幅度取决于原编码耗时占比。配合
`--preset ultrafast` 时收益会更小，因为编码本来就不贵。

---

## 6. 阶段 3：减少每帧同步与显存分配开销

### 6.1 同步（收益中，风险高）

- 现状：`Models.py::_should_drain_syncvec()` 在 `Shared` 下恒 `True`；
  `Per-Thread` 加非默认 stream 时返回 `False`（阶段 1 启用 worker 后已自动跳过）。
- 剩余同步：`headless.py::_swap_frame` 每帧一次 `result.detach().cpu().numpy()` 强制 D2H。
- 步骤：
  1. 先用 `nsys profile -t cuda,nvtx` 确认 worker stream 与默认 stream 之间是否还有
     host stall，不要凭猜测删同步。
  2. 让 `_parallel_swap_frames` 的 worker 返回 GPU tensor，把 `.cpu().numpy()` 推迟到
     编码线程；编码线程读取前用 `wait_stream(worker_stream)` 或 CUDA event 保证顺序。
  3. 若采用阶段 2 的独立编码线程，需把 worker 的 stream / event 一并随
     `(index, rgb, pts)` 传过去。

> **红线**：不能直接删 `syncvec.cpu()`。ORT 与 torch 在不同 stream 时，去掉它会读到
> 未写完的缓冲、产出坏帧。必须先用 `user_compute_stream` 把两者绑到同一条 stream，
> 或用 CUDA event 显式排序。

### 6.2 显存分配（收益小，不要误当主方案）

- `Models.py::_cuda_ep_provider_options()` 里 `arena_extend_strategy` 从
  `kSameAsRequested` 改成 `kNextPowerOfTwo`，可减少重复 `cudaMalloc`，VRAM 略增。
- 必须做 A/B（`elapsed` 对比），很可能测不出差异；它只是让"VRAM 占用"这个数字变大。
- `Models.py::_make_session_options()` 在 `Per-Thread` 下设 `intra_op_num_threads=1`、
  `Shared` 下设 `max(2, min(8, cores // 2))`，多 worker 时确认 CPU 侧没有超订。

---

## 7. 阶段 4：batch 推理（收益最大、风险最高）

### 7.1 单帧多脸 batch

`VideoManager.py::_swap_video_inner()` 的 `for face_kps, face_emb in ret:` 逐脸调
`swap_core`，且 `img` 被逐次原地修改（后一张脸基于前一张结果），**存在数据依赖**，
不能简单并行。

1. 分析 `swap_core` 的输出是否只作用于该脸的 bbox 区域；若是，可先把各脸 crop 收集成 batch。
2. 调用 `Models.py::run_swapper_batched()`（已有动态 batch 与 `static-batch=1` 兜底）一次推理。
3. 按原顺序回填到 `img`，保持叠加语义。
4. 用多人脸视频做 PSNR 比对；单人脸视频无收益，应自动回退逐脸路径。

预期：多人脸场景 VRAM 与 GPU 利用率明显上升。

### 7.2 跨帧 batch

把 N 个 worker 的帧拼成一个 batch 一次 ORT Run，需要模型输入支持动态 batch
（`_swapper_batch_unsupported` 已有探测），并重构整个帧流水线。会与 PTS 保序、
`--every-nth`、可变人脸数冲突，**放在最后**，且必须做成独立开关与独立测试。

---

## 8. 阶段 5：GPU 解码 / NVENC 编码（可选，独立性最好）

依据：`requirements.txt` 里 `torchcodec` / `pynvvideocodec` 是注释掉的可选项；
`VideoManager.py::_swap_video_inner` 已接受 CUDA tensor 输入并跳过 CPU 上传，
说明 GPU 常驻解码是预留路径。

1. 安装 `torchcodec>=0.4`，用 `VideoDecoder(..., device="cuda")` 解码出 GPU tensor。
2. 让 `_swap_frame` / `decoded()` 直接产出 CUDA tensor，省掉
   `torch.from_numpy(...).to('cuda')`。
3. 编码侧先探测 `av.codec.Codec("h264_nvenc")` 是否可用；可用则加
   `--encoder nvenc`，把 `crf` / `preset` 映射成 NVENC 的 `cq` / `preset`
   （语义不同，README 必须说明）。
4. NVENC 是独立硬件单元，**不提高 CUDA SM 利用率**，只解除 CPU 编码瓶颈；
   若 PyAV 不能直接吃 GPU frame，仍需一次 D2H。
5. 做成 `--encoder libx264|nvenc`，默认不变。

---

## 9. VRAM 专项（避免走弯路）

| 做法 | 是否真提速 | 说明 |
|---|---|---|
| `Per-Thread` session × N（阶段 1） | 是 | 并发的必要条件，VRAM 约 `N × 3 份 session` |
| 更大 batch（阶段 4） | 是 | kernel 效率提升，VRAM 随 batch 上升 |
| `--backend trt` 加 TensorRT 10 | 是（单帧延迟） | `trt_max_workspace_size=4GiB` 会占 VRAM；需装 `libnvinfer.so.10` |
| `arena_extend_strategy=kNextPowerOfTwo` | 大概率否 | 只是预分配策略 |
| 改 `ThreadsSlider` | 否 | 零读取方的遗留参数 |
| 盲目 `--workers 8` | 否 | GIL、启动开销、OOM 风险 |

---

## 10. 优先级与执行顺序

| 顺序 | 阶段 | 预期收益 | 工作量 | 风险 | 需真实 GPU 验证 |
|---|---|---|---|---|---|
| 0 | 基线 + `avg fps` + `--profile` | 无（但决定后续） | 小 | 极低 | 是 |
| 1 | `--workers N`（已实现） | 中–高 | 已完成 | 中（GIL 可能限制） | **是，尚未做** |
| 2 | 编码独立线程 | 中–高 | 中 | 中（线程 / 异常） | 是 |
| 3 | 同步与 stream 调优 | 中 | 中 | **高**（易出坏帧） | 是 + nsys |
| 4 | batch 推理 | 高 | 大 | 高 | 是 |
| 5 | GPU 解码 / NVENC | 中 | 中 | 中（画质 / 驱动） | 是 |

建议顺序：`0 → 1 的真机验证 → 2 →（按 nsys 结果决定）3 → 阶段 4 的 7.1
单帧多脸 batch → 5`。跨帧 batch 放最后，除非前四步都测过仍不够。

---

## 11. 每阶段通用验收清单

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
- `--every-nth` 是抽样验证，不是加速；`--target-face` 只影响人脸筛选，
  每帧的识别 embedding 仍会计算。
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
| 有序并行窗口 | `headless.py::_parallel_swap_frames` |
| 帧解码与逐脸循环 | `headless.py::decoded()`（局部生成器）、`VideoManager.py::_swap_video_inner` |
| session 共享策略 | `Models.py::_get_model_session`、`set_model_session_mode` |
| 后端偏好（cuda 不碰 TRT） | `Models.py::set_backend_preference_for_all`、`headless.py::_SwapperBase.__init__` |
| 同步判断、CUDA EP 选项、session 选项 | `Models.py::_should_drain_syncvec`、`_cuda_ep_provider_options`、`_make_session_options` |
| batch 推理 | `Models.py::run_swapper_batched`、`VideoManager.py::swap_core` |
| CLI 参数 | `cli.py::build_parser`（`--workers`、`--preset`、`--crf`、`--every-nth`、`--backend`） |
| 完成计时 | `cli.py::main`、`_format_elapsed` |
| 离线测试 | `tests/test_cli.py`、`tests/test_timing.py`、`tests/test_hyperswap.py`、`tests/test_model_store.py` |
| 命名与 GUI 遗留 | `docs/adr/0001-keep-gui-era-parameter-naming.md`、`CONTEXT.md` |
