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
- 整帧旋转（90/180/270°）的视频自动扶正，**不需要额外参数**（见 3.1.1）
- 多张源脸合并为一个身份
- 换脸模型可选 inswapper_128（polyphase 拼接到 256/512）或 hyperswap-256
- 按需下载模型、多下载源、完整性校验
- 可选人脸修复、遮罩、色彩匹配等参数

## 目录

1. [环境准备](#1-环境准备)
2. [安装步骤](#2-安装步骤)
3. [使用方法](#3-使用方法)
4. [常见问题](#4-常见问题)
5. [侧脸与旋转](#5-侧脸与旋转)
6. [免责声明](#6-免责声明)

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
| `hyperswap1b` | `hyperswap_1b_256.onnx` | 否 | `models-3.3.0` |
| `hyperswap1c` | `hyperswap_1c_256.onnx` | 否 | `models-3.3.0` |
| `arcface100` | `glintr100.onnx` | 否 | 无 GitHub release；HuggingFace / 镜像 / URL（社区转出，sha256） |
| `landmark68` | `2dfan4.onnx` | 否 | `models-3.0.0` |
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
| `--swapper-model hyperswap-1b-256` | `hyperswap_1b_256.onnx` |
| `--swapper-model hyperswap-1c-256` | `hyperswap_1c_256.onnx` |
| `--restorer [模型]` | 对应的修复模型 |
| `--occluder` | `occluder.onnx` |
| `--xseg` | `dfl_xseg.onnx` |
| `--face-parser` | `faceparser_resnet34.onnx` |
| `--recognizer glintr100` | `glintr100.onnx` |
| `--align 68-point` | `2dfan4.onnx` |

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
> 均无收益，A/B 数据见 3.3 节的「提速办法（2026-11-06 真机实测）」。
> 改造后的完整命令写法见下一节 3.1.1。

### 3.1.1 改造后命令速查（2026-11-07）

> **一句话**：整帧被转过 90/180/270 度的视频，现在**不用加任何参数**就能
> 正常识别和换脸；正常视频的命令一个字都不用改。

| 场景 | 在最小命令后追加 | 说明 |
|---|---|---|
| 普通视频 / 图片 | 无 | 与改造前完全一致 |
| 手机竖拍、横竖颠倒、剪辑转过 90° | **无** | 默认自动扶正，见 5.2 |
| 目标脸在画面内大角度歪头（roll）导致漏检 | `--probe-angles 45` | 8 向探测，只旋转检测副本；默认路径不变 |
| 多人镜头里目标人物歪头漏检、别人正脸还在 | `--target-face person.jpg --probe-angles 45 --probe-on-target-miss` | 受控补探测 + 跨角度合并去重 |
| 关掉自动扶正（回滚 / A-B） | `--no-auto-orient` | 回到改造前行为 |
| 侧脸漏检 | `--detect-input-size 960 --detect-retry` | 见 5.1 |
| 一正一侧混合帧（正脸检出、侧脸漏掉） | `--detect-retry --detect-retry-quality` | 低置信度 / kps 几何差时重检，只追加不替换 |
| 侧脸换上去但贴歪（L3 几何） | `--align 68-point --align-for-swap` | 仅 video + `--target-face`；正脸帧保持原 kps |
| 侧脸匹配不上 | `--recognize-mirror --pose-compensation 15` | 见 5.1 |
| 提速 | `--workers 2` | 唯一实测有效，可与自动扶正叠加 |

**旋转视频的完整命令**（推荐写法，与最小命令只差文件名）：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target rotated90.mp4 --output out.mp4 \
    --workers 2
```

**不需要再写**：用 `--params saved_parameters.json` 去开 `OrientAutoSwitch`、
手工指定旋转角度，或任何「朝向」相关参数——这些都是改造前的做法。

旋转与侧脸同时存在时，直接叠加开关即可：

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target-face person.jpg \
    --target rotated90.mp4 --output out.mp4 \
    --workers 2 --detect-input-size 960 --recognize-mirror \
    --pose-compensation 15
```

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
| `hyperswap-1b-256` | `hyperswap_1b_256.onnx` | 同上，1b 轮训练权重 |
| `hyperswap-1c-256` | `hyperswap_1c_256.onnx` | 同上，1c 轮训练权重 |

注意 `inswapper-256` / `inswapper-512` 里的数字是**输出分辨率**，不是模型
本身——三者跑的都是同一个 128 模型，靠 polyphase 拼出更大的输出。想要
原生单次前向的大模型，用 `hyperswap-256`。三个 hyperswap 变体架构、输入
规格完全一致，只有训练轮次不同，身份相似度与伪影表现略有差别。

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

**结论**：唯一实测有效的提速手段是 `--workers 2`（Tesla T4 / 720p / 30 秒
750 帧样本约 **1.5×**）；其余实验开关全部持平或负收益，不建议叠加。

```bash
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target input.mp4 --output output.mp4 \
    --workers 2 --preset ultrafast --crf 23
```

| workers | elapsed | avg fps | 加速比 | 对 w1 逐帧 PSNR |
|---:|---:|---:|---:|---|
| 1 | 226.9s | 3.3 | 1.00 | — |
| 2 | 151.0s | 5.0 | **1.50** | `inf` |
| 4 | 139.1s | 5.4 | **1.63** | `inf` |

`inf` = 逐像素一致，并发没改变输出；w2 → w4 只多 0.4 fps，不必超过 2。
`nsys` 把瓶颈定位到逐脸 ORT Run 与 `cudaStreamSynchronize`（占 CUDA API
时间 75.6%），H2D + D2H memcpy 仅 0.243s / 36.1s——针对主机数据搬运的
优化上限不到 1%。

| 实验开关 | 作用 | A/B（10 秒 250 帧，基线 60.9s / 4.1 fps） | 状态 |
|---|---|---|---|
| `--encode-thread` | 编码移到独立线程 | 64.8s / 3.9 fps（**负收益**） | 默认关，不建议开 |
| `--defer-d2h` | D2H 挪到帧消费侧 | 63.0s / 4.0 fps（与 `--arena-strategy pow2` 同开） | 默认关，仅 `--workers > 1` 生效 |
| `--arena-strategy pow2` | 减少重复 `cudaMalloc` | 同上 | 只让 VRAM 数字变大，不提速 |
| `--reuse-detect-buffers` | 复用 detector 9 个输出缓冲 | 62.6s / 4.0 fps | 默认关，`--backend trt` 下自动关 |
| `--zero-copy-frames` | 跳过整帧 `astype` 复制 | 63.3s / 3.9 fps | 默认关 |
| `--encoder nvenc` | NVENC 硬件编码 | 本容器权限不足，明确报错 | 默认 `--encoder libx264` |
| `--decoder gpu` | GPU 常驻解码 | torchcodec 0.17 与 torch 2.10 ABI 不匹配，明确报错 | 默认 `--decoder pyav` |
| 7.1 / 7.2 batch | 单帧多脸 / 跨帧 batch | 动态 batch 模型数值等价，但 7.1 输出仅 PSNR 45.69 dB | 两个闸门保持 `False` |

上表 A/B 的逐帧 PSNR 均为 `inf`（像素一致），只是吞吐无收益。各开关的实现、
闸门与离线测试见 [docs/gpu-utilization-plan.md](docs/gpu-utilization-plan.md)。

`--force-recognize` 是唯一默认生效的性能变化：未指定 `--target-face` 时
跳过无读取方的逐帧 ArcFace 推理（像素不变），传该 flag 可退回旧路径。
`--every-nth` 是抽样验证、不是加速；`--target-face` 只影响人脸筛选。

> `--workers > 1` 时不要通过 `--params` 打开 GUI 时代的 `OrientAutoSwitch`
> （跨帧状态会竞态）。5.2 节默认开启的整帧朝向兜底**不在此列**，它逐帧
> 无状态，可与 `--workers 2` 一起用。

#### GPU 基线验收脚本

`workers=1/2/4` 矩阵封装成 `tools/bench_gpu.py`，**默认 dry-run**（只打印
命令，不碰 GPU），加 `--run` 才执行：

```bash
python tools/bench_gpu.py --source face.jpg --target in.mp4
python tools/bench_gpu.py --source face.jpg --target in.mp4 --run \
    --workers 1,2,4 --monitor-gpu --check-frames --psnr
```

采集 `elapsed` / `avg fps` / `--profile` 三段、GPU 利用率与显存、`ffprobe`
帧数，以 `workers=1` 为基线算加速比与逐帧 PSNR，报告写
`bench-out/bench_report.json`。`--extra-flag` 可透传任意开关做 A/B（每个
token 单独给一次，例如 `--extra-flag=--recognize-mirror`）。命令构造与解析在
`faceswapper/bench_plan.py`，无 GPU 也能离线自检。

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
| `--detect-input-size` | `320/416/480/640/800/960`；侧脸/小脸召回差时加大（960² ≈ 2.25× 640² 检测耗时） | `640` |
| `--detect-score` | 检测阈值 `1-100` | `50` |
| `--merge` | 多源脸合并方式 | `Mean` |
| `--swapper-model` | `inswapper-128`/`inswapper-256`/`inswapper-512`/`hyperswap-256`/`hyperswap-1b-256`/`hyperswap-1c-256` | `inswapper-128` |
| `--color-match` | 启用 LAB 色彩匹配 | 关闭 |
| `--backend` | `cuda`（纯 CUDA EP，不尝试 TensorRT）或 `trt`（需 TensorRT 10 运行时） | `cuda` |
| `--arena-strategy` | `same`（按需分配）或 `pow2`（减少重复 `cudaMalloc`，只是让 VRAM 数字变大，不是提速手段） | `same` |
| `--reuse-detect-buffers` | 复用 detector 的 9 个 `io_binding` 输出缓冲，省掉每帧 9 次显存分配；`--backend trt` 下自动关闭 | 关闭 |
| `--zero-copy-frames` | 帧已是 `uint8` C contiguous 时跳过 `astype` 的整帧复制（H2D 与 D2H 两侧），不改变像素 | 关闭 |
| `--recognize-mirror` | 识别时把裁剪图水平镜像再各跑一次 ArcFace、取平均，压低侧脸 embedding 方差；识别推理 x2 | 关闭 |
| `--recognize-dim2` | 识别用 2×2 网格平均（4 个裁剪各跑一次 ArcFace 取平均）；识别推理 x4 | 关闭 |
| `--detect-retry` | 0 检出帧自动重试一次：检测输入升到 960、分数阈值 +20；只对漏检帧追加推理，`OrientAutoSwitch` 开启时自动让位 | 关闭 |
| `--detect-retry-quality [SPEC]` | 把 `--detect-retry` 放宽到「检出了但不可靠」的一正一侧混合帧。`SPEC` = `MIN_CONF:MIN_KPS_QUALITY`，省略值用默认 `0.60:0.50`（**两个阈值都需真机标定**）。判据是本帧最高 confidence 过低或 5 点构型质量过低（旋转不变，不受歪头污染）；重检结果按 5 点 RMS 去重后**只追加不替换**。默认不传 = 旧行为逐像素不变；需配合 `--detect-retry` | 不传（关闭） |
| `--no-auto-orient` | 关闭**默认开启**的整帧朝向兜底（0 检出帧试 90/180/270 度、转正后换脸，正常帧零开销），用于回滚 / A-B。见 5.2 | 关闭（即兜底开启） |
| `--probe-angles STEP` | 朝向探测的角度步长：`45` = 8 向、`22.5` = 16 向、`90` = 4 向（须整除 360）。用于**脸在画面内大角度 roll 导致 detector 漏检**：显式传入后只旋转检测副本、不旋转用于换脸的原帧，5 点 kps 逆映射回原帧坐标，输出尺寸不变。默认不传 = 旧的整帧旋转路径，逐像素行为不变。只在 0 检出帧触发，8 向最多多跑 7 次检测 | 不传（旧路径） |
| `--probe-on-target-miss` | 指定 `--target-face` 且本帧已有脸、却没有任何一张匹配上目标人物时，补做多向探测并跨角度合并去重（5 点 RMS 去重、0° 检出优先保留，只追加不替换）。未给 `--probe-angles` 时补探测按 8 向；仅对帧内 ≤3 张脸的帧触发，0 检出帧仍由原兜底负责。**目标人物不在画面里的帧也会触发**，成本可能显著上升，建议先 A-B | 关闭 |
| `--recognizer` | B3：识别权重。`w600k_r50`（默认）或 `glintr100`（ResNet100 ArcFace，侧脸/难样本相似度更稳）；识别推理 x4，需下载 `glintr100.onnx` | `w600k_r50` |
| `--align` | B4：识别对齐。`5-point`（默认，检测 5 点直接算相似变换）或 `68-point`（2dfan4 检 68 点后取两眼/鼻/嘴角再变换，侧脸裁剪更准）；识别前多一次推理，需下载 `2dfan4.onnx` | `5-point` |
| `--align-for-swap [MIN_SIDENESS]` | 把识别阶段已算出的 68 点派生 5 点**复用到 `swap_core` 的替换几何**（`SimilarityTransform` 与 `paste_back` 一起跟随），修 L3「侧脸换上去但贴歪」。可接侧度阈值（默认 `0.35`，需真机标定）：用旋转不变的 `yaw_sideness` 判定，只有大 yaw 帧启用，正脸帧保持原 kps 逐像素不变；另加 68 点几何质量门，遮挡导致 68 点不自洽时自动回退。**强制要求 `--align 68-point` 与 `--target-face`**（否则明确报错），仅 video 路径；不传 = 旧行为，无额外推理 | 不传（关闭） |

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
| `--pose-compensation` | 姿态自适应匹配阈值：全侧脸时判定线比 `--target-threshold` 低多少分（按每张检测脸的关键点正脸度线性插值）；`0` = 关闭 | `0` |
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

## 5. 侧脸与旋转

> 2026-11-07 落地。除 5.2 的整帧朝向兜底**默认开启**外，本章开关全部默认
> 关闭，不传即逐像素一致（PSNR=`inf`）。**真机收益数据仍待跑**，下表是开关
> 与命令，不是已验证的收益承诺。

先分清三种「转」：脸在画面里**歪着**（roll）不用管，5 点相似变换天然吸收；
脸**侧过去**（yaw）用 5.1；**整个画面**被转了 90 度（手机竖拍 / 横竖颠倒）
用 5.2，默认自动处理。

### 5.1 侧脸：症状 → 命令

| 症状 | 追加开关 | 代价 |
|---|---|---|
| 该换的帧没换（漏检） | `--detect-input-size 960`，再加 `--detect-retry` | 960² ≈ 2.25× 640² 检测耗时；retry 只对 0 检出帧多一次检测 |
| 指定 `--target-face` 时匹配不上 / 换错人 | `--recognize-mirror --pose-compensation 15` 起步 | 识别 ×2；正脸阈值不变，越侧越松 |
| 还不够 | 再加 `--recognize-dim2`、`--recognizer glintr100`、`--align 68-point` | 识别 ×4 / ×4 / +1 次 landmark；后两者按需下载模型 |
| 源图本身是侧脸 | `--merge Qual`（多源时）+ 上面的识别开关 | — |

开关正交、可组合。**不要降 `--detect-score`**：降阈值换来的误检会把背景
当脸换掉。视频未指定 `--target-face` 时逐帧识别整体被跳过，识别类开关此时
没有额外成本。

```bash
# 侧脸 + 指定人：识别与匹配全开
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target-face person.jpg \
    --target input.mp4 --output output.mp4 \
    --recognize-mirror --recognize-dim2 \
    --recognizer glintr100 --align 68-point --pose-compensation 15
```

### 5.2 整帧旋转：默认自动扶正

画面整体转过 90/180/270 度时 RetinaFace 对横躺脸召回下降，表现为整段「该换
的脸没换掉」。现在**默认处理，无需任何参数**：

```bash
# 旋转视频：命令与最小命令完全一样
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target rotated90.mp4 --output out.mp4

# 回滚 / A-B：关掉兜底
python -m faceswapper.cli --models ./models video \
    --source face.jpg --target rotated90.mp4 --output out.mp4 \
    --no-auto-orient
```

机制：第一遍检出脸就结束（正常帧零额外推理、零旋转）；只有 0 检出的帧才试
90/180/270，取检出脸数最多的角度转正，检测、识别、换脸在旋转域跑完再整帧
逆旋回来。只用 90 倍数 → 帧尺寸不变、插值不影响画质；不跨帧存状态 → 与
`--workers 2` 不冲突、输出不依赖调度顺序。GUI 时代的 `OrientAutoSwitch`
若通过 `--params` 显式打开，兜底会让位，不会叠加两次旋转。

代价与风险：漏检帧最多多 3 次检测 + 1 次重检 + 2 次整帧旋转，空镜头多的
素材会逐帧重复这笔开销；判定只看检出脸数，极端误检画面可能选错朝向，此时
用 `--no-auto-orient` 关掉或抬高 `--detect-score`。

**验收（待真机）**：`tools/bench_gpu.py --run --psnr` 配合 `--extra-flag`
做「默认 vs `--no-auto-orient`」A/B，比对命中帧数、`ffprobe` 输出宽高，以及
正着放素材的 PSNR=`inf`。GPU 侧旋转几何尚未跑过。

---

## 6. 免责声明

本项目仅用于合法、负责任和合乎伦理的用途。使用他人肖像前必须获得明确同意。

不得制作或传播伤害、诽谤、欺骗、骚扰他人的内容。使用者需要自行遵守所在地
关于隐私、肖像、深度合成、版权和数据处理的法律法规。

`hyperswap_1a_256.onnx` 采用 ResearchRAIL 许可，**仅限非商业用途**；
其余模型请遵守各自的上游许可，模型许可证与代码许可证不同。

使用本项目即表示使用者理解并承担相应责任。项目作者不对误用、滥用或使用
结果造成的后果负责。

