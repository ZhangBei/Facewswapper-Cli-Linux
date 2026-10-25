# Faceswapper 领域词汇表

本项目的注释与文档用中文撰写，但**领域术语一律保留英文原词**，不造中文译名。
这样做的理由：这些词在上游文档（insightface / FaceFusion / ONNX Runtime /
TensorRT）里就是这些拼写，译成中文后反而检索不到，也会让中英混排的注释读起来
更碎。

## 约定

1. 注释、docstring、`help=` 字符串用中文行文。
2. 下表列出的术语在中文句子里**逐字保留英文**，不加引号、不翻译、不改大小写。
3. 标识符、文件名、CLI 参数名、代码片段、数学表达式、URL、许可证名一律原样
   保留——它们不是术语，是代码。
4. 拿不准某个词算不算领域术语时，保留英文。中文注释里嵌一个英文技术词，好过
   一个似是而非的译名。

## 必须保留英文的术语

| 术语 | 含义 |
|---|---|
| `swapper` | 换脸生成器模型（`inswapper` / `hyperswap`） |
| `pass` | 一次前向推理；各 pass 之间共享 0-1 输入输出契约 |
| `polyphase` / `tiling` | 用 128 模型拼出 256/512 的拼合方式 |
| `emap` | inswapper 用的 latent 投影 |
| `latent` | 条件向量 |
| `embedding` | 512 维身份向量（ArcFace 输出） |
| `swap_size` / `pipeline_size` | 喂给模型的方形裁剪尺寸 / 裁剪后管线尺寸 |
| `detector` / `recognizer` | 人脸检测器（RetinaFace / SCRFD）/ 识别器（ArcFace） |
| `restorer` / `occluder` | 人脸修复器 / 遮挡遮罩 |
| `session` / `engine` | ONNX Runtime session / TensorRT engine |
| `worker` / `per-thread` | 工作线程 / 每线程一份会话的模式 |
| `marker` / `snapshot` | 标记点 / 参数快照。`snapshot` 指一次运行生效的参数集合（GUI 时代概念，标识符不可改）；`marker` 已无对应代码 |
| `slot` | 一帧里检测到的、属于同一个人的脸条目（含 `Embedding` / `SourceFaceAssignments` / `AssignedEmbedding` / `Scale`） |
| `provider` / `spec` | 模型下载源 / `ModelSpec` |
| `time_base` / `PTS` | 流的时间基 / 呈现时间戳 |
| `VRAM` / `scratch` | 显存 / 暂存缓冲 |
| `strength` / `paste-back` | 强度循环 / 贴回原图 |
| `mask` | 遮罩 |
| `GUI` | 已移除的 Qt 界面。它留下的命名（`*TextSel` / `*Slider` / `*Switch`）与扁平参数 dict 仍是 `saved_parameters.json` 的磁盘契约，不可改 |
| `preload` / `warmup` / `profile` / `cache` | 预热 / 预热 / TRT profile / 引擎缓存 |
| `syncvec` / `io_binding` / `stream` | torch↔ORT 同步向量 / IO 绑定 / CUDA stream |
| `batch` / `dim` / `FP16` / `FP32` | 保留英文 |
| `ONNX` / `TRT` / `ORT` / `CUDA` / `nsys` / `PyAV` | 保留英文 |
| `ResearchRAIL` / `CRC32` / `SHA-256` | 许可证与校验算法名 |

## 维护注意事项

- `Models.py` 里 `trt_dump_ep_context_model` 那段 12 行说明在约 626 / 783 /
  911 / 1528 行附近重复了 4 次（11/12 行相同）。改动其中一处需同步其余三处。
- GUI 时代命名为何保留、GUI 专用代码删除了哪些，见
  [docs/adr/0001-keep-gui-era-parameter-naming.md](docs/adr/0001-keep-gui-era-parameter-naming.md)。
