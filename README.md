# Faceswapper 无界面换脸 CLI

基于 insightface `inswapper` 模型的命令行换脸工具，面向 **Ubuntu 24.04 +
NVIDIA GPU** 的云端/服务器环境。没有图形界面，不需要桌面环境，通过 SSH
即可运行。

```text
源人脸图片 + 目标图片/视频 -> 换脸结果
```

当前支持：

- 图片换脸：单张、批量、多人脸或只处理最大人脸
- 视频换脸：替换所有人，或只替换参考图指定的目标人物
- 多张源脸合并为一个身份
- 换脸模型可选 inswapper_128（polyphase 拼接到 256/512）或 hyperswap-256
- 按需下载模型、多下载源、完整性校验
- 可选人脸修复、遮罩、色彩匹配等参数

## 目录

1. [环境准备](#1-环境准备)
2. [安装步骤](#2-安装步骤)
3. [使用方法](#3-使用方法)
4. [常见问题](#4-常见问题)
5. [免责声明](#5-免责声明)

---

## 1. 环境准备

### 1.1 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 24.04；其他 Linux 理论可用 |
| Python | 3.11 或更高 |
| GPU | NVIDIA GPU；4 GB 显存可尝试低配参数，推荐 8 GB 以上 |
| NVIDIA 驱动 | cu128 需 570+，cu126 需 560+，cu124 需 550+ |
| FFmpeg | 6.1 或更高，包含 `libx264` 和 `ffprobe` |
| 网络工具 | `curl`，用于模型下载 |
| 磁盘 | 基础模型约 450 MB，另需缓存和输出空间 |

先确认 GPU 和驱动：

```bash
nvidia-smi
```

能够正常打印 GPU 型号和驱动版本即可继续。

> **当前推理代码必须使用 NVIDIA GPU。** ONNX Runtime、PyTorch 张量、
> `io_binding` 和视频处理路径固定使用 CUDA。纯 CPU 需要先完成设备抽象改造，
> 目前没有可靠的 `--device cpu` 参数。

### 1.2 安装系统工具

```bash
sudo apt update
sudo apt install -y curl
```

`curl` 是模型下载器断点续传所需的唯一系统工具，FFmpeg 由 conda 提供（见下）。

### 1.3 安装 Miniconda

已有 conda 可跳过这一步：

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"
```

### 1.4 创建 conda 环境

```bash
cd /path/to/Faceswapper-Cli
conda env create -f environment.yml
conda activate faceswapper
python -m pip install --upgrade pip
```

`environment.yml` 只安装**原生依赖**，Python 包统一留给 pip：

| 依赖 | 来源 | 用途 |
|---|---|---|
| `python=3.11` | conda | 运行时不拆分 |
| `ffmpeg>=6.1` | conda | 视频 mux 走系统 ffmpeg；自检需要 `libx264` |
| `cuda-toolkit=12.8` | conda | CUDA 运行时库 |
| `libgl` / `libglib` | conda | headless OpenCV 仍链接它们 |
| `curl` | conda | 模型下载的断点续传 |
| `pip` | conda | Python 包统一由 pip 安装 |

> **驱动版本不匹配怎么办？**
> `environment.yml` 默认 `cuda-toolkit=12.8`，需要驱动 570+。驱动较旧时先改
> 这一行，再创建环境：
>
> ```bash
> sed -i 's/cuda-toolkit=12.8/cuda-toolkit=12.6/' environment.yml
> ```
>
> 或改用 `12.4`。CUDA 小版本要和后面 `tools/install.py` 的通道一致
> （12.8→`cu128`，12.6→`cu126`，12.4→`cu124`）。

验证 FFmpeg 可用且带 `libx264`：

```bash
ffmpeg -version
ffmpeg -hide_banner -encoders 2>&1 | grep -o 'enable-libx264'
ffprobe -version
```

### 1.5 替代方案：venv + apt

不想用 conda、且有 root 权限时，可以用系统包加 venv：

```bash
sudo apt update
sudo apt install -y python3.11-venv ffmpeg curl libgl1 libglib2.0-0

cd /path/to/Faceswapper-Cli
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

两种方案的 Python 依赖完全相同，都通过 pip 安装；模型下载和换脸命令也
完全一致。

---

## 2. 安装步骤

### 2.1 安装 Python 依赖

**推荐：自动检测环境**

```bash
python tools/install.py default
```

它会读 `nvidia-smi` 拿到驱动版本，据此选 CUDA 通道，然后**只装缺的
部分**：

| 驱动 | 选中的通道 |
|---|---|
| 570+ | `cu128` |
| 560-569 | `cu126` |
| 550-559 | `cu124` |

torch 始终从 **pip 自己的源**安装（即 `PIP_INDEX_URL` 或 PyPI），和模型
下载一样不写死任何地址。需要指定 CUDA 专用 wheel 通道时显式传
`--index-url`：

```bash
python tools/install.py default \
    --index-url https://download.pytorch.org/whl/cu126
```

已经装好的东西不会被动：torch 能用 CUDA 就跳过安装，onnxruntime 已有
CUDAExecutionProvider 就不卸载（卸载只发生在真要装之前），
`requirements.txt` 里已有的包不会重装。环境已满足时直接退出并提示
去跑 `check_env.py`。

找不到 `nvidia-smi` 或驱动低于 550 时会明确报错并告诉你怎么办，不会
猜一个通道装错东西。

**也可以显式指定通道**（行为与之前完全一致，适合你比 `nvidia-smi` 更清楚
该用哪个源的情况——比如系统已装 CUDA toolkit、或要用镜像）：

```bash
python tools/install.py cu128
python tools/install.py cu126
python tools/install.py cu124
```

显式指定时 torch 走 pip 自己的源，除非加 `--index-url`。

安装器按顺序做的事：

1. 卸载 CPU、DirectML、OpenVINO 等其他 ONNX Runtime 变体
2. 安装 PyTorch 和 torchvision
3. 安装 `requirements.txt`
4. 完成 headless OpenCV、ONNX Runtime GPU、PyAV 等依赖安装

常用选项：

```bash
# 只看会执行什么命令，不实际安装
python tools/install.py default --dry-run

# torch 已单独装好，只装其余依赖
python tools/install.py default --skip-torch

# 忽略检测结果，全部重装
python tools/install.py default --force-reinstall
```

### 2.2 依赖从哪个源安装

torch 默认走 **pip 自己的源**（即 `PIP_INDEX_URL` 或 PyPI），项目不写死任何
地址。需要指定 CUDA 专用 wheel 通道时显式传 `--index-url`：

```bash
python tools/install.py cu128 --index-url https://download.pytorch.org/whl/cu128
```

> **为什么 Python 包统一走 pip？**
> conda 和 pip 各自提供 `onnxruntime`，混装是"没有 CUDAExecutionProvider"
> 最常见的原因。所以 `environment.yml` 只装原生依赖，Python 包全部由 pip
> 安装，保证环境里只有一个 ONNX Runtime 构建。

`requirements.txt` 中的关键依赖：

| 包 | 作用 |
|---|---|
| `onnxruntime-gpu` | 模型推理（CUDA EP / TensorRT EP） |
| `onnx` | 读取 ONNX 图（提取 emap 投影矩阵） |
| `opencv-python-headless` | 图像读写、颜色转换 |
| `torch` / `torchvision` | 逐帧 warp、mask 混合等张量运算 |
| `av` | PyAV：视频解封装与解码 |
| `scikit-image` | 相似变换估计 |
| `pillow` / `numpy` | 图像与数组基础 |

> PyAV 需要 `>=13.0,<19.0`：14 及以上版本要求 `add_stream(rate=...)` 传
> `Fraction` 而不是 float，代码已按此实现。

### 2.3 模型下载

模型文件不能通过 pip 安装。项目带自动下载器，不需要手工复制文件。

**下载基础模型：**

```bash
python -m faceswapper.cli --models ./models download-models
```

基础模型共 3 个，约 450 MB：

| 本地文件 | 用途 | 大小 |
|---|---|---:|
| `det_10g.onnx` | RetinaFace 人脸检测 | 16.1 MiB |
| `w600k_r50.onnx` | ArcFace 人脸识别 | 166.3 MiB |
| `inswapper_128.fp16.onnx` | 128 分辨率换脸 | 264.8 MiB |

下载完成后校验完整性，失败会尝试下一个下载源，不会留下未验证的文件。

#### 选择下载源

**默认下载源是 `huggingface`（Hugging Face 官方仓库）**，只有真的需要下载
模型时才会用到它：本地模型齐全时，`swap` / `video` / `inspect` 直接离线运行，
不联网、也不要求任何配置。优先级从高到低：

1. 命令行 `--model-provider`（全局）或 `download-models --provider`
2. 环境变量 `HF_ENDPOINT`（与 `huggingface_hub` 使用同一个变量，配一次镜像全局生效）
3. 默认 `huggingface`

改用国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m faceswapper.cli --models ./models download-models
```

显式按命令指定时会覆盖上面的默认源，支持三种具名来源：

- `huggingface`：Hugging Face 官方仓库
- `hf-mirror`：Hugging Face 国内镜像
- `github`：FaceFusion GitHub Release

按指定顺序尝试，前者失败自动用下一个：

```bash
python -m faceswapper.cli --models ./models download-models \
    --provider hf-mirror \
    --provider github
```

`--provider` 还接受完整 URL，指向任意自建镜像（按 Hugging Face 布局拼接）：

```bash
python -m faceswapper.cli --models ./models download-models \
    --provider https://your-mirror.example.com
```

例如上面这条会请求
`https://your-mirror.example.com/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx`。

#### 可选模型

下载器管理的全部模型：

| `--model` | 本地文件 | 必需 | 上游发布 |
|---|---|---|---|
| `base` | 3 个基础模型（省略 `--model` 时的默认） | — | — |
| `retinaface` | `det_10g.onnx` | 是 | `models-3.0.0` |
| `arcface` | `w600k_r50.onnx` | 是 | `models-3.0.0` |
| `inswapper` | `inswapper_128.fp16.onnx` | 是 | `models-3.0.0` |
| `scrfd` | `scrfd_2.5g_bnkps.onnx` | 否 | `models-3.0.0` |
| `hyperswap` | `hyperswap_1a_256.onnx` | 否 | `models-3.3.0` |
| `gfpgan` | `GFPGANv1.4.onnx` | 否 | `models-3.0.0` |
| `gpen256` | `GPEN-BFR-256.onnx` | 否 | `models-3.0.0` |
| `gpen512` | `GPEN-BFR-512.onnx` | 否 | `models-3.0.0` |
| `codeformer` | `codeformer_fp16.onnx` | 否 | `models-3.0.0` |
| `xseg` | `occluder.onnx` | 否 | `models-3.1.0` |
| `faceparser` | `faceparser_resnet34.onnx` | 否 | `models-3.0.0` |
| `dflxseg` | `dfl_xseg.onnx` | 否 | `models-3.0.0` |
| `resnet50` | `res50.onnx` | 否 | 无（仅登记，需自备） |
| `all` | 以上全部 | — | — |

下载单个可选模型：

```bash
python -m faceswapper.cli --models ./models download-models --model hyperswap
python -m faceswapper.cli --models ./models download-models --model gfpgan
```

> **三个 `--model*` 参数不要混淆，且必须写全名。**
> `--models DIR`（全局，模型目录）和 `--model-provider`（全局，下载源）写在
> 子命令**前面**；`--model`（`download-models` / `check-models` 专属，选择要
> 下载或校验的模型）写在子命令**后面**。CLI 已关闭 argparse 的长选项缩写，
> `--mod`、`--model-p` 这类简写一律不接受，也不会再把 `--model` 误判成
> `--models` / `--model-provider` 的歧义前缀。

**只登记不下载**的文件（上游没有对应发布，需要自行提供）：

- `res50.onnx` — ResNet50 检测器（对应 `--model resnet50`）
- `inswapper_256_phase1.onnx` — 256 原生 inswapper
- `inswapper_512_level2.onnx` — 512 原生 inswapper

这三个 `check-models` 会报告是否存在，缺失只提示不报错；基础换脸用不到。

#### 按需自动下载

换脸命令会在推理前，按本次运行的**参数快照**算出真正需要的模型集合，缺失或
损坏时自动下载。因此可以跳过手动下载直接跑换脸。集合规则：

| 条件 | 额外下载 |
|---|---|
| 始终 | 3 个基础模型 |
| `--detect-type SCRDF` | `scrfd_2.5g_bnkps.onnx` |
| `--swapper-model hyperswap-256` | `hyperswap_1a_256.onnx` |
| `--restorer [模型]` | 对应的修复模型 |
| `--occluder` | `occluder.onnx` |
| `--xseg` | `dfl_xseg.onnx` |
| `--face-parser` | `faceparser_resnet34.onnx` |

也就是说，不打开开关就只下载基础模型，不会为了用不到的功能拉几百 MB。
已存在且校验通过的文件不会重复下载。

禁用自动下载：

```bash
python -m faceswapper.cli --no-auto-download --models ./models swap \
    --source face.jpg --target target.jpg --output out.jpg
```

#### 检查模型

```bash
# 完整校验（计算 SHA-256 / CRC32）
python -m faceswapper.cli --models ./models check-models

# 快速检查（只比大小和缓存）
python -m faceswapper.cli --models ./models check-models --quick

# 只看列表不校验
python -m faceswapper.cli --models ./models download-models --list
```

输出示例：

```text
valid    det_10g.onnx                16.1 MiB  RetinaFace detector
valid    w600k_r50.onnx             166.3 MiB  ArcFace recognition
valid    inswapper_128.fp16.onnx     264.8 MiB  Inswapper 128 face swapper
```

校验结果缓存在 `models/.faceswapper-models.json`，文件内容、大小或修改时间
变化时才重新计算。

强制重新下载：

```bash
python -m faceswapper.cli --models ./models download-models --force
python -m faceswapper.cli --models ./models download-models --model inswapper --force
```

> **校验方式**
> 大部分模型按 SHA-256 校验。`hyperswap_1a_256.onnx` 上游只发布 CRC32，因此
> 按 CRC32 校验——这是 FaceFusion 自己对新版模型使用的校验方式。

> **HyperSwap 许可**
> `hyperswap_1a_256.onnx` 采用 ResearchRAIL 许可，**仅限非商业用途**。商用请
> 继续使用 inswapper_128。

### 2.4 环境检测

```bash
python tools/check_env.py --models ./models
```

按顺序检查：

1. Python 版本
2. NVIDIA 驱动 / `nvidia-smi`
3. `ffmpeg`、`ffprobe`、`curl`
4. PyTorch 与 CUDA 是否可用
5. ONNX Runtime 是否包含 `CUDAExecutionProvider`
6. Python 运行时依赖是否齐全
7. 模型文件及校验值
8. CLI 模块能否导入

全部通过时输出 `all checks passed`。可选模型缺失只警告，不影响退出码。
每一项失败都会打印对应的修复命令。

### 2.5 离线测试套件

不需要 GPU、模型，也不需要 torch / ORT / OpenCV 的一批行为测试，一条命令
跑完，适合 clone 之后先做一次自检：

```bash
python -m faceswapper.tests
```

每个 `test_*.py` 都是自带 `main()` 的独立脚本，runner 用子进程逐个执行，
一个模块失败不会中断整轮，也不会在模块间共享 `sys.modules` 状态。输出按
模块给出状态与耗时，末尾汇总：

| 状态 | 判定 | 退出码 |
|---|---|---|
| `PASS` | `rc=0` 且打印 `all tests passed` | `0` |
| `SKIP` | 缺 `numpy` / `torch` 等白名单里的运行时依赖 | `0` |
| `FAIL` | 断言失败、超时、缺非可选依赖，或 `rc=0` 却没有完成标记 | `1` |

只有白名单里的运行时依赖缺失才判 `SKIP`，import 拼写错误一律算 `FAIL`，
不会把代码错误藏成环境问题。GPU 机器上做正式验收时加 `--strict`，让
`SKIP` 也返回非零：

```bash
python -m faceswapper.tests --list                # 只列模块，不执行
python -m faceswapper.tests --module test_cli    # 只跑指定模块，可重复
python -m faceswapper.tests --strict --timeout 300
```

`python -m faceswapper.tests.test_cli` 这类单模块命令仍然可用，runner 只是
把它们收拢成一条命令。这批测试只覆盖离线逻辑，**不能替代**
[docs/gpu-utilization-plan.md](docs/gpu-utilization-plan.md) 里的真机验收
（3.4 基线矩阵、逐帧 PSNR、nsys 等）。

---

## 3. 使用方法

所有命令都要带 `--models ./models` 指定模型目录。换脸前先准备好：

- 源人脸图片：`face.jpg`，清晰、正面、光线均匀，只取最大的一张脸
- 目标：一张图片、一个图片文件夹，或一个视频文件

### 3.1 快速开始

**图片换脸** —— 替换目标图中检测到的所有人脸：

```bash
python -m faceswapper.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg
```

**视频换脸** —— 替换视频中检测到的所有人脸，音频一并保留：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg \
    --target input.mp4 \
    --output output.mp4
```

这两条就是完整可用的最小命令。首次运行会自动下载缺失的基础模型，默认从
Hugging Face 官方仓库获取（来源见 [2.3](#23-模型下载)）。下面按场景给出
带参数的版本。

> 视频想提速，只需加 `--workers 2`（实测约 1.5×）。其余实验开关在真机上
> 均无收益，数据、原因与不要叠加的清单见 3.3 节的
> 「提速办法（2026-11-06 真机实测）」。

### 3.2 图片换脸

#### 只替换最大的一张脸

目标图里有多人、只想换最突出的那个时：

```bash
python -m faceswapper.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg \
    --face-policy largest
```

`--face-policy` 取 `all`（默认，替换全部）或 `largest`。

#### 低显存 / 最快配置

4 GB 显存或想先快速验证链路时：

```bash
python -m faceswapper.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg \
    --face-policy largest \
    --swapper-model inswapper-128
```

加速因素：

- `--swapper-model inswapper-128`：换脸模型跑 128×128，比 256/512 快得多
- `--face-policy largest`：少一次换脸
- 输出 `.jpg` 而不是 `.png`：省一次无损编码
- 只传一张源脸：跳过 embedding 合并
- 不启用修复、遮罩、色彩匹配等可选功能

#### 多张源脸合并为一个身份

```bash
python -m faceswapper.cli --models ./models swap \
    --source face1.jpg \
    --source face2.jpg \
    --target target.jpg \
    --output out.png \
    --merge Sph
```

合并方式支持 `Mean`、`Median`、`Sph`、`Geo`、`Qual`，默认 `Mean`。
`--source` 也可指向一个文件夹，里面的每张图各取最大人脸。

#### 批量处理整个文件夹

```bash
python -m faceswapper.cli --models ./models swap \
    --source ./faces \
    --target-dir ./targets \
    --output-dir ./output
```

`--target-dir` 与 `--output-dir` 成对使用，输出文件命名为
`<原文件名>_swapped.<扩展名>`，已存在的文件会被覆盖。批量模式下
`--output` 不适用。

#### 使用 HyperSwap 换脸模型

```bash
python -m faceswapper.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.png \
    --swapper-model hyperswap-256
```

首次使用会自动下载模型，也可以先手动下载：

```bash
python -m faceswapper.cli --models ./models download-models --model hyperswap
```

与 inswapper 的区别：

| | inswapper-128 | hyperswap-256 |
|---|---|---|
| 前向方式 | 128×128 输出，NxN polyphase 拼到 256/512 | 原生 256×256 单次前向 |
| 身份 latent | L2 归一化 → emap 投影 → 归一化 | 只做 L2 归一化，跳过 emap 投影 |
| 输入值域 | 0-1 | -1 ~ 1（在 pass 内部转换） |
| 许可 | 随 insightface | **ResearchRAIL，仅非商业** |

两者的身份 latent 都受 `scale` 参数缩放，差别只在有没有 emap 投影那一步。

HyperSwap 只在 `--swapper-model hyperswap-256` 下生效，默认值
`inswapper-128` 行为不变。

#### 换脸模型取值怎么读

`--swapper-model` 同时决定**用哪个模型**和**输出多大**，四个取值：

| 取值 | 实际加载的模型 | 输出 |
|---|---|---|
| `inswapper-128`（默认） | `inswapper_128.fp16.onnx` | 128×128 |
| `inswapper-256` | `inswapper_128.fp16.onnx` | 256×256，NxN polyphase 拼成 |
| `inswapper-512` | `inswapper_128.fp16.onnx` | 512×512，NxN polyphase 拼成 |
| `hyperswap-256` | `hyperswap_1a_256.onnx` | 原生 256×256 单次前向 |

注意 `inswapper-256` / `inswapper-512` 里的数字是**输出分辨率**，不是模型
本身——三者跑的都是同一个 128 模型，靠 polyphase 拼出更大的输出。想要
原生单次前向的大模型，用 `hyperswap-256`。

#### 查看目标图里检测到了哪些脸

```bash
python -m faceswapper.cli --models ./models inspect --target target.jpg
```

输出每张脸的编号和 `scale`（两眼间距，反映该脸在画面里有多大），便于确认
`--face-policy largest` 会选中哪一张、以及 `--detect-score` 该取什么值。

### 3.3 视频换脸

#### 只替换指定的那个人

需要三份输入：

| 输入 | 参数 | 含义 |
|---|---|---|
| 换脸源图 | `--source` | 最终要换成谁 |
| 人物参考图 | `--target-face` | 视频里要替换的是谁 |
| 目标视频 | `--target` | 待处理视频 |

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4
```

处理逻辑：

1. 从 `--source` 提取源人物特征
2. 从 `--target-face` 提取目标人物参考特征
3. 逐帧检测所有脸并计算身份相似度
4. 只替换与参考人物相似度达到 `--target-threshold` 的脸

其他人脸保持不变。可以传多张目标人物参考图提高匹配稳定性：

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_front.jpg \
    --target-face person_side.jpg \
    --target input.mp4 \
    --output output.mp4
```

#### 用多张源图合成一个身份

`--source` 可以直接指向文件夹，也可以重复传多次。所有源图各取最大人脸、
提取 embedding，再按 `--merge` 合并成**一个**身份用于整段视频——与 3.2 节
「多张源脸合并为一个身份」是同一套逻辑，不需要额外的 `--source-folder`
之类参数：

```bash
python -m faceswapper.cli --models ./models video \
    --source /path/to/face10/ \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --merge Qual
```

合并方式 `Mean` / `Median` / `Sph` / `Geo` / `Qual`，默认 `Mean`；源图质量
参差时 `Qual` 会让清晰照片占更大权重。目录只扫描一层，支持
`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`。**每张图都必须能检测到人脸**，
否则会直接报 `no face detected in source image: ...` 而不是跳过。源图建议
都是同一个人、不同角度与光线；混入其他人会把身份平均歪。多张源图只增加
启动阶段的特征提取耗时，逐帧推理仍只用那一个合并后的 embedding。

#### 匹配阈值

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target-face person.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --target-threshold 60
```

- 范围 `0-100`，默认 `55`
- 调高更严格，减少换错人的风险
- 调低更容易匹配，但可能换到长相相似的人

#### 低显存 / 快速验证配置

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --swapper-model inswapper-128 \
    --preset ultrafast \
    --crf 23
```

- `--preset`：x264 速度/质量档位，`ultrafast` ~ `veryslow`，默认 `medium`
- `--crf`：H.264 质量，数值越低质量越高、文件越大，默认 `18`

进一步压榨速度：

- 不需要音频时加 `--no-audio`，省一次音频 mux
- 目标人物只给一张最清晰的参考图
- 4K 或大码率视频先降到 720p
- 验证阶段用 `--every-nth 20` 抽样处理

降到 720p：

```bash
ffmpeg -i input.mp4 -vf scale=-2:720 \
    -c:v libx264 -preset ultrafast -crf 23 \
    -c:a copy input_720p.mp4
```

`--every-nth 20` 只处理第 20 的倍数帧、其余原样透传，用于快速验证链路，
**不是完整的逐帧换脸结果**：

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target test.mp4 \
    --output test_out.mp4 \
    --swapper-model inswapper-128 \
    --preset ultrafast \
    --every-nth 20
```

#### 提速办法（2026-11-06 真机实测）

**结论先行**：唯一被真机验证有效的提速手段是 `--workers 2`。在 Tesla T4 /
720p / 25 fps 的 30 秒 / 750 帧样本上约 **1.5×**；其余可选开关全部持平或
负收益，NVENC、GPU 常驻解码、batch 推理在当前环境不可用。

推荐命令 —— 只加这一个开关，不要叠加下面列出的实验开关：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --workers 2 \
    --preset ultrafast --crf 23
```

**实测数据**（`tools/bench_gpu.py --run --workers 1,2,4`，30 秒正式基线，
报告在 `bench-out/baseline-30s/bench_report.json`）：

| workers | elapsed | avg fps | 加速比 | 对 w1 逐帧 PSNR |
|---:|---:|---:|---:|---|
| 1 | 226.9s | 3.3 | 1.00 | — |
| 2 | 151.0s | 5.0 | **1.50** | `inf` |
| 4 | 139.1s | 5.4 | **1.63** | `inf` |

`inf` 表示逐像素完全一致，并发没有改变输出。w2 → w4 只多 0.4 fps，
收益已递减，**不建议加到 4 以上**。

##### 提速原因

`nsys profile -t cuda,nvtx` 的轨迹显示，耗时集中在逐脸 swapper 推理与
同步等待，而不是主机侧数据搬运：

| 观测 | 数值 | 含义 |
|---|---|---|
| `cudaStreamSynchronize` | 13.41s / 2618 次，占 CUDA API 时间 **75.6%** | 主机在等 GPU 完成 |
| `ort_run_swapper`（NVTX） | 11.77s / 225 次（每帧 3 张脸 × 75 帧） | 逐脸 ORT Run 是主体 |
| `syncvec_drain` | 0.323s | 原先以为的主瓶颈，实测可忽略 |
| H2D + D2H memcpy | 0.243s | 相对 36.1s 总用时不足 1% |
| `cudaMalloc` + `cudaFree` | 0.074s | 分配器开销可忽略 |

`--workers N` 让 N 个 worker 各持一份 ORT session 和一条 CUDA stream，
把「主机等某一次 ORT Run」的时间与另一帧的推理重叠起来；解码与编码仍在
主线程，输出严格按解码顺序（`frame_order.py` 的 `2N` 有序窗口），因此
PTS、VFR 时序、音画对齐与逐帧像素都不变。

反过来，编码、D2H、memcpy、显存分配这些环节的收益上限本来就不到 1%，
针对它们的开关在 A/B 里全部测不出收益：

| 开关 | 10 秒 / 250 帧 A/B（基线 60.9s / 4.1 fps） | 逐帧 PSNR |
|---|---|---|
| `--encode-thread` | 64.8s / 3.9 fps（**负收益**） | `inf` |
| `--defer-d2h --arena-strategy pow2` | 63.0s / 4.0 fps | `inf` |
| `--reuse-detect-buffers` | 62.6s / 4.0 fps | `inf` |
| `--zero-copy-frames` | 63.3s / 3.9 fps | `inf` |
| 阶段 6 + 7 组合 | 62.8s / 4.0 fps | `inf` |

##### 不要这样提速

- 不要叠加上面那批实验开关：全部无收益，`--encode-thread` 还会变慢。
- 不要把显存占用当提速指标：VRAM 随 worker 数上升只是各持一份 session
  的副产物；`--arena-strategy pow2` 只让数字变大，不提速。本机同机还有
  4 个外部 GPU 进程，`nvidia-smi` 读数无法归因于本工具。
- `--every-nth` 是抽样验证、不是加速；降分辨率、`--preset ultrafast`、
  关掉可选功能是减少总耗时，不改变换脸计算量。
- NVENC（`--encoder nvenc`）在本容器权限不足会明确报错；GPU 常驻解码
  （`--decoder gpu`）因 torchcodec 与 torch ABI 不匹配不可用；单帧多脸
  batch 虽有数值等价的动态 batch 模型可用，但输出 PSNR 仅 45.69 dB，
  闸门保持关闭。

完整的分阶段数据、nsys 轨迹与验收顺序见
[docs/gpu-utilization-plan.md](docs/gpu-utilization-plan.md)。

#### 流水线结构与同步开销（含实测修正）

> **2026-11-06 真机实测修正**：下面这段是改造前的结构分析。`nsys` 后来
> 显示主机主要是在**等 GPU**（`cudaStreamSynchronize` 占 CUDA API 时间
> 75.6%），并非 GPU 空闲；且本机同机有 4 个外部 GPU 进程，`nvidia-smi`
> 读数无法归因于本工具。因此「GPU / 显存占用很低」这一旧判断在这台机器上
> 未得到证实。实测有效的提速办法与原因见上一节。

改造前（`workers=1`）的流水线结构：

- 解码（PyAV，CPU）、GPU 推理、libx264 编码都在**同一个线程逐帧串行**
  （`headless.py` 的 `_encode_video`），任意时刻只有一帧在推进；
- 每次 ORT 调用前有 `syncvec.cpu()` 的 CPU↔GPU 同步，每帧结束还有一次
  `result.cpu().numpy()` 回拷；
- 换脸按脸逐次执行，没有跨帧 batch；ORT session 是单份 `Shared`，CUDA EP
  用 `kSameAsRequested` 按需分配显存。

所以单纯调大显存池不会提速，`saved_parameters.json` 里的 `ThreadsSlider`
是没有读取方的遗留参数，改它无效。前面「低显存 / 最快配置」里列的
`--preset ultrafast`、降分辨率、关掉可选功能等仍然有效，但它们是减少总
耗时，不会让 GPU 吃满。`--every-nth` 是抽样验证、不是加速。`--target-face`
只影响人脸筛选：未指定时逐脸识别 embedding 默认已跳过（见阶段 4，像素不变，
省每帧 N 次 ArcFace 推理）；指定时要按相似度筛人，embedding 照常计算。

真正有效的是增加并发，即 `--workers N`（默认 `1`，走串行路径）：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --workers 2
```

`N` 个 worker 各持一份 ORT session 和一条 CUDA stream，解码/编码留在主线程，
换脸推理在线程池里并行；输出仍严格按解码顺序，PTS、VFR 时序和音画对齐与
串行路径一致。显存约为 `N × 3 份模型 session`。**实测 w2 → w4 只多
0.4 fps，取 2 即可**；本机 `nvidia-smi` 读数受同机其它进程污染，不能用来
判断本工具的利用率。

| 方案 | 真机效果 | 状态 |
|---|---|---|
| `--workers N` 多 worker 并行帧 | **约 1.5×（w2）**，唯一有效 | 见「提速办法」 |
| 编码移到独立线程 | **负收益** | 保持默认关闭 |
| 跨帧 batch 推理 | 上限最高 | 代码已编写、闸门关闭；像素等价未通过 |

`--workers > 1` 会启用 Per-Thread session，此时不要打开 `OrientAutoSwitch`
（自动朝向）：它维护跨帧共享状态，多线程下会竞态。默认参数是关闭的，
正常使用不受影响。

观察利用率请用 `nvidia-smi dmon -s um -d 1`（`-s u,m` 是非法写法）；
但要得到可归因的数字，需先隔离同机其它 GPU 进程。

#### 编码独立线程 `--encode-thread`

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --workers 4 --encode-thread --profile
```

主线程按解码顺序把帧投进有界队列（`maxsize=2*workers`），编码线程按 FIFO
取用并负责创建、使用和关闭全部 PyAV 对象；因此 PTS、VFR 时序、音画对齐
和逐帧输出顺序与串行路径一致，进度回调也仍按序触发。编码线程出错会写回
异常并置停止标志，主线程立即中断，不会静默产出半个视频。

可与 `--workers` 叠加，但**默认关闭，不传该参数即走原来的主线程编码路径**，
不需要回滚代码。`--workers > 1` 的 2N 有序窗口已经让编码与推理部分重叠；
`--profile` 里 `encode` 仅占总用时约 5–8%，可争取的空间本来就很有限。
PyAV 的 libx264 编码是否在 C 层释放 GIL 未单独验证，但实测的负收益与
「不释放 GIL 反而卡住主线程」这一推测一致。
离线测试覆盖了队列保序、flush、异常传播与背压。**真机 A/B 已测为负收益**
（64.8s / 3.9 fps，基线 60.9s / 4.1 fps；PSNR=`inf`），不建议开启。

#### 阶段 3：延迟 D2H 与显存策略

两个默认关闭的开关，**都不删除任何 CUDA 同步**：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --workers 4 --defer-d2h --arena-strategy pow2
```

- `--defer-d2h`：把逐帧 `.cpu().numpy()` 从 worker 线程挪到帧的消费侧，
  worker 不再被 D2H 阻塞。仅 `--workers > 1` 生效，串行路径自动忽略。
  `VideoManager.swap_video` 里的 `default_stream().wait_stream(worker_stream)`
  仍保证拷贝读到已写完的缓冲，`Models` 里的 `syncvec.cpu()` 与
  `user_compute_stream` 保持原样。
- `--arena-strategy pow2`：CUDA EP 的 `arena_extend_strategy` 从
  `kSameAsRequested` 换成 `kNextPowerOfTwo`。它只减少重复 `cudaMalloc`、
  让 VRAM 占用数字变大，**不是提速手段**（所有命令均可用）。

与实施方案的差别：D2H 推迟到**帧的消费侧（主线程）**而非编码线程，顺序继续
靠已有的 `wait_stream`，没有新增 CUDA event、也没有把 stream 对象跨线程传递，
因此 `encoder.py` 仍是 stdlib-only。代价是收益可能低于预期：D2H 仍在主线程
串行，跨 worker 的默认 stream 隐式屏障也还在。开启 `--defer-d2h` 时，D2H
耗时不计入 `--profile` 的三段。后续 `nsys` 已确认 host stall 主体是
`cudaStreamSynchronize`（占 CUDA API 时间 75.6%），而 D2H memcpy 仅
0.062s；真机 A/B 也显示该开关无收益（63.0s / 4.0 fps，PSNR=`inf`）。
不传这两个开关即回到原行为。

#### 阶段 4：跳过无用的逐脸识别

单帧多脸 batch（方案 7.1）的代码已按「先编写、不启用」落地：唯一接入点在
`_swap_video_inner` 的 headless 分支里，被 `EXPERIMENTAL_FACE_BATCH=False`
闸门包住，默认短路到原逐脸循环，输出与基线逐字一致；指定 `--target-face`
时也不会走 batch（相似度筛选语义不同）。之所以不能直接启用：paste-back 虽只作用于该脸
bbox，但 `run_swapper_batched` 的 N 是 polyphase phase 数而非脸数、
`pipeline_size` 逐脸不同、`swap_core` 是 438 行单体函数，且输入 warp 会从
已被前一张脸写回的 `img` 采样——预批量准备会改变重叠人脸的像素。因此
`face_batch.py` 强制要求 bbox 两两不重叠、只用 `inswapper-128`、关闭多 pass
开关，任一不满足就回退逐脸路径。**真机受控实验补充的第六条硬阻断**：当前
`inswapper_128.fp16.onnx` 是 `static-batch=1`，当时把闸门临时置 `True`
会让 N>1 的 batch 被 ONNX 以 `Got: 3 Expected: 1` 拒绝，异常回退后触发
`CUDA failure 700` 崩溃。现已给判定补上必填的 `batch_backend_available`
并修正 session 探测顺序，复测确认误开闸门会干净回退（PSNR=`inf`、无
CUDA 错误）。动态 batch 模型也已通过本地图改写构造并验证数值等价
（逐脸路径 PSNR=`inf`），但用它开闸后 batch 对逐脸只有 **PSNR 45.69 dB**。
因此**两个实验闸门仍必须保持 `False`**：剩余阻断是 7.1 prepare/finish
的像素等价性，不是模型能力。六条完整依据、受控实验与解冻步骤见
[docs/gpu-utilization-plan.md 7.1](docs/gpu-utilization-plan.md)。

改实现了一条逻辑等价、改动更小的优化：headless video 未指定 `--target-face`
时，所有检测到的人脸都换成同一个 `--source` 身份，逐帧 `run_recognize` 算出的
embedding **没有任何读取方**。现在默认跳过这些调用，每帧少 N 次 ArcFace 推理
（N = 检测到的脸数），像素输出完全不变。

```bash
# 退回旧路径，便于 A/B 或排查
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 --force-recognize
```

指定 `--target-face` 时会自动照常计算 embedding（要按相似度筛人）；图片换脸走
`found_faces` 匹配路径，同样不受影响。判定在 `recognize.py`，离线行为测试覆盖
四种组合。`nsys` 已确认逐脸调用被跳过（`:run_recognize` 全程仅 1 次，
即源特征提取），但收益数字未单独做 A/B。

跨帧 batch（方案 7.2）的**判定、有序分组、前向/回填编排与闸门**均已写进
代码，但 `EXPERIMENTAL_CROSS_FRAME_BATCH=False` 且主流水线完全没有调用它。
判据很硬：`workers >= 2`、不跳帧、仅 `inswapper-128`、每帧恰好一张脸、
`pipeline_size` 恒为 128、`index` 连续、PTS 严格递增——最后两条保证 VFR
时序不被 batch 重排。编排层「prepare 全部帧 → 一次 ORT Run → 按序 finish」
抽成注入回调的纯函数并有离线行为测试；`VideoManager.cross_frame_batch_swap`
复用 7.1 的 prepare/finish 钩子，不复制 postprocess，闸门关闭时补一次逐脸
fallback 保证输出等价。GPU 侧前向、逐帧 PSNR 与 VFR/音画时序仍需真机核对，
五步解冻流程见 [docs/gpu-utilization-plan.md 7.2](docs/gpu-utilization-plan.md)。

#### 阶段 5：NVENC 编码开关

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 --encoder nvenc
```

默认 `--encoder libx264`，行为与之前逐字一致。选择 `nvenc` 时会先探测当前
PyAV/FFmpeg 是否提供 `h264_nvenc`；不可用直接报错并给出修复提示，**不会静默
回退**。`--encode-thread` 与默认编码路径共用同一个编码器选择结果。

参数**不是等价替换**：x264 的 `--crf` 映射成 NVENC 的 `cq`（配合 `rc=vbr`），
`--preset` 的六档映射成 `p1..p7`。CRF 与 CQ 的率控行为不同，`p1..p7` 也不是
x264 档位的一一对应；较旧的 FFmpeg 可能不认 `p1..p7`，遇到未知 preset 请回退
`--encoder libx264`。NVENC 是独立硬件单元，**不提高 CUDA SM 利用率**，只解除
CPU 编码瓶颈；PyAV 目前仍吃 CPU `rgb24` frame，那次 D2H 并没有省掉。

#### GPU 常驻解码 `--decoder gpu`（实验性）

`--decoder pyav` 是默认，完全不 import torchcodec、也不碰 CUDA。显式
`--decoder gpu` 时会先探测 `torchcodec>=0.4`、`VideoDecoder` 是否接受
`device=`，以及 `torch.cuda.is_available()`；任一不满足直接报错并提示回退，
**不静默回退**。通过后用 `VideoDecoder(video_path, device="cuda")` 产出
CUDA tensor——`_swap_video_inner` 本来就能吃 CUDA tensor，于是省掉
`torch.from_numpy(...).to('cuda')` 这次上传。

torchcodec 各版本的 frame 字段与 PTS 单位不统一，适配层刻意**不猜**：取不到
`data` / `frame`、拿不到 PTS、或 PTS 是小数（多半是秒）时一律明确报错，避免
静默产出坏帧或错时序。首帧绝对 PTS 由 GPU 路径自己记录，音频 mux 的偏移照常
恢复。判定与字段提取在 `decoder.py`，有离线行为测试。

真机已核对（本环境不可用）：装上 `torchcodec 0.17.0` 后 import 即报
`libtorchcodec_image.so: undefined symbol: torch_from_blob`，当前
`torch 2.10.0+cu128` 与 torchcodec ABI 不匹配；`--decoder gpu` 会明确
报 `GPU decode unavailable` 并提示用默认路径，不会静默回退。验证后已
卸载该可选包。因此 frame 布局、PTS 单位、VFR 与音画同步仍未能核对。
不传该开关即回到原 PyAV 路径。

#### 阶段 6：复用 detector 输出缓冲（每帧少 9 次显存分配）

`Models._detect_retinaface_inner` 每帧为 9 个 `io_binding` 输出各调一次
`torch.empty`（3 个 stride × score/bbox/kps）。这些缓冲的 shape 只由
`(input_height, input_width, stride)` 决定，而前两个尺寸在一次 video 里恒定
（除非改 `--detect-input-size`）。输入缓冲和 anchor centers 早就按 worker
缓存了，输出缓冲是这条链上唯一还每帧重新分配的一环。

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --reuse-detect-buffers
```

形状计算、闸门判定与「按 key 命中缓存、未命中才 alloc」抽在 stdlib-only 的
`detect_buffers.py`，`torch.empty` 由 `Models.py` 注入，因此缓存命中/失效、
跨 worker 隔离都能离线做行为测试。每组缓冲挂在已有的
`_retinaface_buffers_tls` 上——各 worker 的 stream 并发推进，共享会让 ORT 的
写入互相踩；同一 worker 内则是安全的，`rf_postprocess` 读完这一帧后下一帧的
写入才排到它后面。

默认**关闭**，不传开关时 `_detect_retinaface_inner` 逐字走原来每帧 9 次
`torch.empty` 的路径（AST 测试钉住这条 fallback）。`--backend trt` 下判定逻辑
会自动关闭复用：TRT-EP 的 `io_binding` 输出行为未经真机核对，不猜。

**真机已验证**：10 秒 / 250 帧 `workers=2` 下逐帧 PSNR=`inf`（说明 ORT
每帧完整覆写复用缓冲）、帧数与音画正常；但吞吐无可测量收益（62.6s /
4.0 fps vs 基线 60.9s / 4.1 fps）。默认保持关闭，不传开关即回滚。

#### 阶段 7：帧路径零拷贝归一化（少一次整帧 memcpy）

帧路径上有多处 `frame.astype("uint8")`：numpy 的 `astype` 默认 `copy=True`，
哪怕输入已经是 `uint8` 且 C contiguous，也会**无条件**复制一整帧。PyAV 的
`to_ndarray(format="rgb24")`（H2D 侧）与 `_finalize_frame` 的 D2H 结果恰好常
常已经满足这两个条件。1080p RGB 单侧约 6.2 MB，H2D + D2H 两侧合计约
12.4 MB/帧，60 秒 30fps 就是约 22 GB 纯内存带宽——不碰 GPU 计算，也不改一个
像素。

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --zero-copy-frames
```

判定抽在 stdlib-only 的 `frame_upload.py`：`normalize_plan()` 只吃 `dtype` /
`c_contiguous` / `height` / `width` / 闸门，返回 `FramePlan`（是否要复制、省下
的字节数、原因）。生产代码读取 `rgb.dtype.name` 与
`rgb.flags.c_contiguous` 后按 plan 决定直接复用原数组，或用
`np.ascontiguousarray(..., dtype=np.uint8)` 一次完成转换 + 整理。模块本身不
import numpy，因此闸门关闭、命中、dtype 不符、布局不连续、字节预算都能离线
做真行为测试。

共接了 5 个调用点：`VideoManager._swap_video_inner` 的 CPU numpy H2D 分支，
以及 `headless.py` 的 `source_embedding()`、`_detect_face_slots()`、
`ImageSwapper.swap()`、`_finalize_frame()`。跳过复制时返回的是与输入共享内存
的数组，调用方只把它交给同步的 `torch.from_numpy(...).to("cuda")` /
`VideoFrame.from_ndarray`，不会原地写它；PyAV 每帧 `to_ndarray` 都产出新数组，
不存在跨帧复用。

默认**关闭**：闸门关闭时所有 `astype` 表达式逐字保留（AST 测试钉住 H2D 分支
与 `_frame_to_uint8` 两条 fallback）。开启后也不改像素——dtype 或布局不满足时
仍照旧复制，只是不再对已经合格的帧做无用功。

**真机已验证**：10 秒 / 250 帧 `workers=2` 下逐帧 PSNR=`inf`（dtype /
layout 判定与实际内存一致）、帧数与音画正常；但吞吐无可测量收益（63.3s /
3.9 fps vs 基线 60.9s / 4.1 fps）。`nsys` 显示 H2D + D2H memcpy 合计仅
0.243s / 36s，收益上限 <1%。默认保持关闭，不传开关即回滚。

#### GPU 基线验收脚本

方案 3.4 的 `workers=1/2/4` 矩阵已封装成 `tools/bench_gpu.py`，**默认
dry-run**（只打印命令，不碰 GPU、不写文件），加 `--run` 才真正执行：

```bash
python tools/bench_gpu.py --source face.jpg --target in.mp4
python tools/bench_gpu.py --source face.jpg --target in.mp4 --run \
    --workers 1,2,4 --monitor-gpu --check-frames --psnr
```

脚本会采集 `elapsed` / `avg fps` / `--profile` 三段耗时、GPU 利用率与显存
曲线、`ffprobe` 帧数，并以 `workers=1` 为回归基线算加速比和逐帧 PSNR，
结果写到 `bench-out/bench_report.json`。`--extra-flag` 可透传任意开关做 A/B
（每个 token 单独给一次，例如 `--extra-flag=--encode-thread`）。命令构造与
输出解析在 `faceswapper/bench_plan.py`，有离线行为测试，因此在没有 GPU 的
机器上也能安全地预览与自检。报告现在会把逐帧 PSNR 的 `average` 数值写成
`psnr_average` 字段；`--monitor-gpu` 用 `nvidia-smi dmon -s um` 采样。

脚本已在 Tesla T4 上跑通 10 秒矩阵（两轮）、30 秒 / 750 帧正式基线，
以及阶段 2 / 3 / 6 / 7 的 A/B；w2 / w4 对 `workers=1` 的 PSNR 均为
`inf`（逐像素一致）。正式基线 `workers=1/2/4` 为 `3.3 / 5.0 / 5.4 fps`
（加速比 `1.0 / 1.50 / 1.63`）。**真机结论**：除 `--workers 2` 外，
编码独立线程、延迟 D2H、detector 缓冲复用、帧零拷贝均无可测量收益，
NVENC 在本环境不可用，不建议开启这些开关；`nsys` 已把瓶颈定位到逐脸
ORT Run 与 `cudaStreamSynchronize`。后续又验证了 P0（动态 batch 模型可
本地构造且数值等价，但 7.1 batch 路径 PSNR 仅 45.69 dB）与 P2
（`torchcodec 0.17` 与 `torch 2.10` ABI 不匹配，GPU 解码不可用）。
完整数字、结论与剩余待办见
[docs/gpu-utilization-plan.md](docs/gpu-utilization-plan.md) 的
「第二 / 第三轮真机数据」两节。

完整的分阶段实施方案（现状瓶颈、详细实现步骤、验收清单、风险与回滚）见
[docs/gpu-utilization-plan.md](docs/gpu-utilization-plan.md)。

#### 完成后的总用时

换脸命令成功结束后会打印整条命令的总用时（从参数校验、模型准备、源特征
提取到编码与音频 mux），格式为 `HH:MM:SS.s`。视频命令还会附带平均处理速度：

```text
frame 4731/4731 (100.0%)
done: 4731 frame(s) at 24.000 fps, 720x1280, audio=yes -> output.mp4
elapsed: 00:02:34.6 (avg 30.6 fps)
```

`avg fps` 用的是编码循环实际消费的帧数，不是容器声明的 `stream.frames`
（后者可能为 0，或只是按 duration 估算的值），因此不受元数据缺失影响。
图片换脸 `swap` 没有帧数概念，只打印 `elapsed:`；失败或 `Ctrl+C` 中断时
不打印，避免把半个任务的时间误报成一次完整运行。

#### 分段计时 `--profile`

`video` 加 `--profile` 会在结束时额外打印三段累计耗时：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 --profile
```

```text
done: 4731 frame(s) at 24.000 fps, 720x1280, audio=yes -> output.mp4
profile: decode=12.40s swap=131.20s encode=18.30s
elapsed: 00:02:34.6 (avg 30.6 fps)
```

`decode` 是取帧、`to_ndarray` 与 PTS 计算，`swap` 是逐帧换脸，`encode` 是
编码与 mux。`--workers > 1` 时三段在不同线程上重叠，之和会大于总用时——
它用来判断瓶颈落在哪一段，不要相加。默认关闭；关闭时计时逻辑只多一次
布尔判断，不改变推理与编码行为。

### 3.4 参数总览

全局参数必须写在子命令**前面**：

```bash
python -m faceswapper.cli \
    --models /data/models \
    --model-provider hf-mirror \
    video --help
```

| 全局参数 | 说明 |
|---|---|
| `--models DIR` | 模型目录，默认 `./models` |
| `--model-provider` | 自动下载源（URL 或 `github`/`huggingface`/`hf-mirror`），可重复；默认 `huggingface`，设置 `$HF_ENDPOINT` 时以它为准，仅在实际发生模型下载时生效 |
| `--no-auto-download` | 禁用模型自动下载 |

> **所有选项都必须写完整名称**，CLI 不接受长选项缩写：`--mod`、`--model-p`
> 这类简写会被判为未识别参数。这样做是为了让子命令的 `--model` 不被
> argparse 误判成 `--models` / `--model-provider` 的歧义前缀（见
> [2.3 模型下载](#23-模型下载)）。选项的位置规则同上：全局参数在子命令前，
> 子命令专属参数在子命令后。

`swap` / `video` / `inspect` 共同参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--params FILE` | 加载 GUI 时代保存的 `saved_parameters.json` | 无 |
| `--detect-type` | `Retinaface` 或 `SCRDF` | `Retinaface` |
| `--detect-input-size` | `320/416/480/640` | `640` |
| `--detect-score` | 检测阈值 `1-100` | `50` |
| `--merge` | 多源脸合并方式 | `Mean` |
| `--swapper-model` | `inswapper-128`/`inswapper-256`/`inswapper-512`/`hyperswap-256` | `inswapper-128` |
| `--color-match` | 启用 LAB 色彩匹配 | 关闭 |
| `--backend` | `cuda`（纯 CUDA EP，不尝试 TensorRT）或 `trt`（需 TensorRT 10 运行时） | `cuda` |
| `--arena-strategy` | `same`（按需分配）或 `pow2`（减少重复 `cudaMalloc`，只是让 VRAM 数字变大，不是提速手段） | `same` |
| `--reuse-detect-buffers` | 复用 detector 的 9 个 `io_binding` 输出缓冲，省掉每帧 9 次显存分配；`--backend trt` 下自动关闭 | 关闭 |
| `--zero-copy-frames` | 帧已是 `uint8` C contiguous 时跳过 `astype` 的整帧复制（H2D 与 D2H 两侧），不改变像素 | 关闭 |

> **`--detect-input-size` 不要随便调小。** 它只影响检测器输入分辨率，
> 不影响换脸质量——调小是纯速度交易：`480` ≈ 1.8× 快，`416` ≈ 2.4×，
> `320` ≈ 4× 快但源帧里小于约 36 px 的人脸开始漏检。视频路径上漏检的
> 表现不是崩溃，而是**该换的脸没被换掉**（逐帧相似度掉到
> `--target-threshold` 以下，或压根没检测到脸）。低显存配置请优先调
> `--swapper-model`，检测分辨率保持 `640` 最稳。

功能开关（打开后才会下载对应模型，见 [2.3](#23-模型下载)）：

| 参数 | 说明 | 下载的模型 |
|---|---|---|
| `--restorer [模型]` | 人脸修复，`GFPGAN`/`CF`/`GPEN256`/`GPEN512`，省略值为 `GFPGAN` | 对应的修复模型 |
| `--occluder` | 遮挡遮罩 | `occluder.onnx` |
| `--xseg` | DeepFaceLab XSeg 遮挡遮罩 | `dfl_xseg.onnx` |
| `--face-parser` | 背景遮罩 | `faceparser_resnet34.onnx` |

```bash
python -m faceswapper.cli --models ./models swap \
    --source face.jpg --target target.jpg --output out.png \
    --restorer GPEN256 --occluder
```

`swap` 专有参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--source` | 源脸图片，可重复或指向文件夹 | 必填 |
| `--target` | 目标图片，可重复或指向文件夹 | 必填 |
| `--target-dir` | 目标图片文件夹（批量模式） | 无 |
| `--output` | 输出图片路径（单个目标） | 无 |
| `--output-dir` | 输出文件夹（批量模式） | 无 |
| `--face-policy` | `all` 或 `largest` | `all` |

`video` 专有参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--source` | 源脸图片，可重复或指向文件夹 | 必填 |
| `--target-face` | 视频中要替换的人物参考图，可重复。省略则替换所有脸 | 无 |
| `--target-threshold` | 人物匹配相似度阈值 `0-100` | `55` |
| `--target` | 输入视频文件 | 必填 |
| `--output` | 输出视频文件 | 必填 |
| `--crf` | H.264 质量，越低越好 | `18` |
| `--preset` | x264 档位 | `medium` |
| `--encoder` | `libx264`（默认）或 `nvenc`（需探测 `h264_nvenc` 可用） | `libx264` |
| `--decoder` | `pyav`（默认，CPU 逐帧）或 `gpu`（torchcodec 常驻 CUDA，实验性） | `pyav` |
| `--no-audio` | 不复制源音频轨 | 关闭 |
| `--every-nth` | 只处理第 N 帧，其余透传 | `1` |
| `--workers` | 并行处理帧的 worker 数，`>1` 时各 worker 独占一份模型 session | `1` |
| `--profile` | 打印 decode / swap / encode 分段累计耗时 | 关闭 |
| `--encode-thread` | 把编码移到独立线程，与换脸推理解耦 | 关闭 |
| `--defer-d2h` | 把逐帧 GPU→CPU 拷贝从 worker 推迟到帧的消费侧（仅 `--workers > 1` 生效） | 关闭 |
| `--force-recognize` | 保留每帧逐脸 ArcFace 推理的旧路径（回滚 / A/B 用） | 关闭 |

查看全部已知参数及默认值：

```bash
python -m faceswapper.cli params
```

> 部分参数只在 `--params saved_parameters.json` 里可配，没有对应命令行开关，
> 例如修复结果与原图的混合方式（`RestorerDetTypeTextSel`：
> Original/Blend/Reference）。命令行开关的值优先于参数文件。

### 3.5 视频时序

视频处理用 PyAV 逐帧解码，重新编码时**保留每帧原始 PTS**，可变帧率视频的
帧间距不会被强制改写。音频用系统 ffmpeg 从源视频复制，并对两条流分别恢复
起始偏移。因此在源视频本身音画同步正常的情况下，输出会保持原有音画关系。

---

## 4. 常见问题

### 4.1 安装问题

#### `AttributeError: 'float' object has no attribute 'numerator'`

PyAV 14 及以上版本不再接受 float 作为 `add_stream(rate=...)` 的参数，只接受
`Fraction`。确认 PyAV 版本在 `requirements.txt` 声明的范围内：

```bash
python -c "import av; print(av.__version__)"
```

版本低于 13.0 或高于 19.0 时重新安装：

```bash
python -m pip install 'av>=13.0,<19.0'
```

#### `torch.cuda.is_available()` 返回 False

按顺序检查：

1. `nvidia-smi` 是否正常，是否看得到 GPU
2. 驱动版本是否匹配所选的 `cu128`/`cu126`/`cu124`
3. 是否装成了 CPU 版 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`torch.version.cuda` 为 `None` 说明是 CPU wheel，重新选择通道：

```bash
python tools/install.py cu126
```

#### ONNX Runtime 没有 `CUDAExecutionProvider`

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出必须包含 `CUDAExecutionProvider`。缺失说明装成了 CPU 版，或被 conda 版
`onnxruntime` 覆盖。先清理再重装：

```bash
conda list onnxruntime
pip show onnxruntime-gpu
conda remove --force onnxruntime
python tools/install.py cu128
```

#### 启动时报 `libnvinfer.so.10` / TensorRT `EP Error`

日志形如 `Failed to load library ...libonnxruntime_providers_tensorrt.so
with error: libnvinfer.so.10: cannot open shared object file`，随后是
`EP Error ... RegisterTensorRTPluginsAsCustomOps` 和
`Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider']`。

含义：ONNX Runtime 尝试加载 TensorRT EP，但环境里没有 TensorRT 10 的运行时
库，于是自动回退到 CUDA EP。**推理结果不受影响**——只要后面打印
`[Models] ...: CUDAExecutionProvider`，就说明实际在 GPU 上跑。

- 默认 `--backend cuda` 不应再出现这段日志。早期版本只给部分模型设了 CUDA
  偏好，`hyperswap` 仍会去初始化 TRT EP；当前版本已让 cuda 模式覆盖全部
  模型，若仍看到说明运行的是旧代码。
- 确实想用 TensorRT 加速：先安装 TensorRT 10 运行时（提供
  `libnvinfer.so.10`，例如 `pip install tensorrt`，并确保库在
  `LD_LIBRARY_PATH` 中），再传 `--backend trt`；没装好仍会回退到 CUDA EP。

#### 更换模型下载源

默认源是 `huggingface`（Hugging Face 官方仓库），只有本次运行需要的模型
**缺失或损坏**、确实要联网下载时才会用到。先确认到底缺哪个文件：

```bash
python -m faceswapper.cli --models ./models check-models --quick
```

改用镜像或自建源时，设置环境变量，或按命令显式指定（注意全局参数要写在
子命令前面）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或：全局自动下载（swap / video / inspect）
python -m faceswapper.cli --models ./models --model-provider hf-mirror swap \
    --source face.jpg --target target.jpg --output out.jpg
# 或：download-models 子命令
python -m faceswapper.cli --models ./models download-models --provider github
```

> 旧版本没有内置默认源，会报 `no model download source given`；早期版本
> 即使模型齐全也会无条件校验下载源。两种情况更新
> `faceswapper/model_store.py` 后即可解决。

#### Hugging Face 下载很慢或中断

默认源就是 Hugging Face 官方仓库。国内访问慢或中断时设置镜像，对所有
huggingface 工具同时生效：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

中断后直接重跑同一条命令，已下载的 `.part` 文件支持续传，完成后会做完整性
校验。也可以按顺序指定多个来源，失败自动切换：

```bash
python -m faceswapper.cli --models ./models download-models \
    --provider hf-mirror \
    --provider github
```

#### 模型下载后仍报校验失败

文件可能被截断或损坏。强制重新下载：

```bash
python -m faceswapper.cli --models ./models download-models --force
```

只重下某一个：

```bash
python -m faceswapper.cli --models ./models download-models \
    --model inswapper --force
```

### 4.2 使用问题

#### `ambiguous option: --model could match --models, --model-provider`

旧版本 CLI 的参数解析问题：顶层同时有 `--models` 和 `--model-provider`，
argparse 默认允许长选项缩写，会先把子命令里的 `--model` 当成这两个选项的
歧义前缀，在参数交给 `download-models` / `check-models` 之前就报错退出。
当前版本已在顶层 parser 关闭缩写（`allow_abbrev=False`），`--model` 会原样
传给子解析器，文中所有带 `--model` 的示例均可直接使用。

如果仍看到这条报错，说明运行的不是更新后的代码（例如环境里装着旧版
`faceswapper` 包，或 `python -m` 没有从当前项目目录加载）。更新
`faceswapper/cli.py` 后，用只列状态、不下载的命令验证参数解析：

```bash
python -m faceswapper.cli --models ./models download-models --model hyperswap --list
```

在旧版本上也可以先省略 `--model`，直接跑换脸命令，让
`--swapper-model hyperswap-256` 等开关触发按需自动下载（见 2.3 节
「按需自动下载」）。注意 `--model=hyperswap` 同样绕不过这个歧义。

#### 没有检测到人脸

依次尝试：

- 换一张清晰、正面、光线均匀的源脸
- 降低 `--detect-score`（默认 50）
- 图片保持 `--detect-input-size 640`，不要降到 320
- 目标人物脸部太小先提高视频分辨率
- 用 `inspect` 子命令确认到底检测到了什么：

```bash
python -m faceswapper.cli --models ./models inspect --target target.jpg
```

#### 换错了人

视频里用 `--target-face` 指定要替换的人物，并调高阈值：

```bash
python -m faceswapper.cli --models ./models video \
    --source new_face.jpg \
    --target-face person.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --target-threshold 70
```

图片里人很多时用 `--face-policy largest` 只处理最突出的那张。

#### 显存不足（CUDA out of memory）

依次尝试：

1. `--swapper-model inswapper-128`
2. 关闭 `--restorer`、`--occluder`、`--xseg`、`--face-parser`、`--color-match`
3. 视频降到 720p 再处理
4. 确认没有多个进程同时占用同一张 GPU（`nvidia-smi`）
5. 以上都无效才降 `--detect-input-size`（`480` → `320`）。这确实省
   显存，但会漏检小人脸，属于拿结果换显存，见上面的警告

#### 换脸结果模糊

- `--swapper-model inswapper-256` 或 `inswapper-512`：换脸输出分辨率更高，显存和时间也更多
- 换 `hyperswap-256`：原生 256×256 单次前向，不经 polyphase 拼接
- 开 `--restorer` 做一次人脸修复
- `--crf` 降到 16 左右减少编码损失

#### 音画不同步

先确认源文件本身是否同步。源视频已经错位时先强制转成恒定帧率并重采样音频：

```bash
ffmpeg -i input.mp4 -fps_mode cfr -r 30 \
    -af aresample=async=1 input_cfr.mp4
```

再用 `input_cfr.mp4` 作为 `--target`。若输出仍不同步而源文件正常，请把
`ffprobe -show_streams` 的输出和源视频片段一起反馈。

#### conda 环境损坏

模型文件保存在 `models/` 下，重建环境不会删除，不需要重新下载：

```bash
conda deactivate
conda env remove -n faceswapper
sed -i 's/cuda-toolkit=12.8/cuda-toolkit=12.6/' environment.yml
conda env create -f environment.yml
conda activate faceswapper
python tools/install.py cu126
```

### 4.3 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 未预期错误，会打印 traceback |
| `2` | 用户输入、模型或环境错误 |

---

## 5. 免责声明

本项目仅用于合法、负责任和合乎伦理的用途。使用他人肖像前必须获得明确同意。

不得制作或传播伤害、诽谤、欺骗、骚扰他人的内容。使用者需要自行遵守所在地
关于隐私、肖像、深度合成、版权和数据处理的法律法规。

`hyperswap_1a_256.onnx` 采用 ResearchRAIL 许可，**仅限非商业用途**；
其余模型请遵守各自的上游许可，模型许可证与代码许可证不同。

使用本项目即表示使用者理解并承担相应责任。项目作者不对误用、滥用或使用
结果造成的后果负责。

