# Rope 无界面换脸 CLI

基于 insightface `inswapper` 模型的命令行换脸工具，面向 **Ubuntu +
NVIDIA GPU** 的云端/服务器环境。没有图形界面，不需要桌面环境，通过 SSH
即可运行。

```text
源人脸图片 + 目标图片/视频 -> 换脸结果
```

当前支持：

- 图片换脸：单张、批量、多人脸或只处理最大人脸
- 视频换脸：替换所有人，或只替换参考图指定的目标人物
- 多张源脸合并为一个身份
- 自动下载缺失模型、多下载源、SHA-256 完整性校验
- 可选人脸修复、遮罩、色彩匹配等参数

## 目录

0. [快速开始](#0-快速开始)
1. [环境要求](#1-环境要求)
2. [安装](#2-安装)
3. [准备模型文件](#3-准备模型文件)
4. [环境自检](#4-环境自检)
5. [图片换脸](#5-图片换脸)
6. [视频换脸](#6-视频换脸)
7. [常用参数](#7-常用参数)
8. [常见问题](#8-常见问题)
9. [项目结构](#9-项目结构)
10. [免责声明](#10-免责声明)

---

## 0. 快速开始

在一台新的 Ubuntu 24.04 + NVIDIA GPU 云主机上，按顺序执行：

```bash
# 1. 安装 Miniconda（已有 conda 可跳过）
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"

# 2. 创建 conda 环境（含 CUDA、FFmpeg、libGL）
cd /path/to/Rope-Rope-Bronze
conda env create -f environment.yml
conda activate rope
python -m pip install --upgrade pip

# 3. 安装 CUDA Python 运行时（驱动较旧时改用 cu126 或 cu124）
python tools/install.py cu128

# 4. 下载并校验模型（可省略，首次换脸会自动下载）
python -m rope.cli --models ./models download-models

# 5. 环境自检
python tools/check_env.py --models ./models

# 6. 图片换脸
python -m rope.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg
```

每一步的详细说明见后续章节。

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 24.04；其他 Linux 理论可用 |
| Python | 3.11 或更高 |
| GPU | NVIDIA GPU；4 GB 显存可尝试低配参数，推荐 8 GB 以上 |
| NVIDIA 驱动 | 支持所选 CUDA wheel；cu128 建议驱动 570+ |
| FFmpeg | 6.x，包含 `libx264` 和 `ffprobe` |
| 网络工具 | `curl`，用于模型下载 |
| 磁盘 | 基础模型约 450 MB，另需缓存和输出空间 |

> **当前推理代码必须使用 NVIDIA GPU。** ONNX Runtime、PyTorch 张量、
> `io_binding` 和视频处理路径固定使用 CUDA。纯 CPU 需要先完成设备抽象改造，
> 目前没有可靠的 `--device cpu` 参数。

先确认 GPU：

```bash
nvidia-smi
```

能够正常打印 GPU 型号和驱动版本即可继续。

---

## 2. 安装

### 2.0 两种安装方式

本项目支持两种环境方案，任选其一：

| 方案 | 适用场景 | 章节 |
|---|---|---|
| Conda（推荐） | 云主机没有 root，或想统一管理 CUDA、FFmpeg | [2.1](#21-方式一conda推荐) |
| venv + apt | 有 root 权限，希望轻量安装 | [2.4](#24-方式二venv--apt) |

两种方式的 Python 依赖完全相同，都通过 `pip` 安装；模型下载和换脸命令也
完全一致。

---


### 2.1 方式一：Conda（推荐）

```bash
sudo apt update
sudo apt install -y python3.11-venv ffmpeg curl
```

验证：

```bash
ffmpeg -version
ffprobe -version
curl --version
```

安装 Miniconda（如尚未安装）：

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"
```

创建环境。`environment.yml` 会安装 Python 3.11、FFmpeg、CUDA 运行时和
OpenCV 所需的 `libGL`：

```bash
cd /path/to/Rope-Rope-Bronze
conda env create -f environment.yml
conda activate rope
python -m pip install --upgrade pip
```

> **驱动版本不匹配怎么办？**
> `environment.yml` 默认使用 `cuda-toolkit=12.8`，需要驱动 570+。驱动较旧时
> 先修改该行，再创建环境：
>
> ```bash
> sed -i 's/cuda-toolkit=12.8/cuda-toolkit=12.6/' environment.yml
> ```
>
> 或改用 `12.4`。CUDA 小版本必须与后面 `tools/install.py` 使用的
> `cu126` / `cu124` 通道一致。

安装 Python 依赖：

```bash
python tools/install.py cu128
```

驱动较旧时选择对应通道：

```bash
python tools/install.py cu126
python tools/install.py cu124
```

安装器会：

1. 先卸载 CPU、DirectML、OpenVINO 等其他 ONNX Runtime 变体
2. 安装对应 CUDA 版本的 PyTorch
3. 安装 `requirements.txt`
4. 安装 headless OpenCV、ONNX Runtime GPU、PyAV 等依赖

只查看将要执行的命令，不实际安装：

```bash
python tools/install.py cu128 --dry-run
```

已单独安装 PyTorch 时：

```bash
python tools/install.py cu128 --skip-torch
```

> **为什么 Python 包仍然走 pip？**
> conda 和 pip 各自提供 `onnxruntime`，混装是 "没有 CUDAExecutionProvider"
> 最常见的原因。因此 `environment.yml` 只安装原生依赖（CUDA、FFmpeg、
> `libGL`），Python 包统一由 pip 安装，保证只有一个 ONNX Runtime 构建。

### 2.4 方式二：venv + apt

不想使用 conda 时，用系统包加 venv：

```bash
sudo apt update
sudo apt install -y python3.11-venv ffmpeg curl libgl1 libglib2.0-0

cd /path/to/Rope-Rope-Bronze
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python tools/install.py cu128
```

这种方式需要 root 或 sudo 权限安装系统包。

---

## 3. 准备模型文件

ONNX 模型不能通过 `pip install` 安装。项目现在提供自动下载器，不需要手工
复制文件。

### 3.1 自动下载基础模型

```bash
python -m rope.cli --models ./models download-models
```

默认下载并校验以下 3 个模型：

| 本地文件 | 用途 | 大小 |
|---|---|---:|
| `det_10g.onnx` | RetinaFace 人脸检测 | 约 16 MiB |
| `w600k_r50.onnx` | ArcFace 人脸识别 | 约 166 MiB |
| `inswapper_128.fp16.onnx` | 128 分辨率换脸 | 约 265 MiB |

下载时使用 SHA-256 验证。校验失败会尝试下一个下载源，最终不会留下未验证的
模型文件。

### 3.2 选择下载源

支持三个来源：

- `huggingface`：Hugging Face 官方仓库
- `hf-mirror`：Hugging Face 国内镜像
- `github`：FaceFusion GitHub Release

按指定顺序下载：

```bash
python -m rope.cli --models ./models download-models \
    --provider hf-mirror \
    --provider github
```

显式指定只使用一个来源：

```bash
python -m rope.cli --models ./models download-models \
    --provider github
```

### 3.3 下载可选模型

GFPGAN 人脸修复：

```bash
python -m rope.cli --models ./models download-models \
    --model gfpgan
```

SCRFD 备用人脸检测器：

```bash
python -m rope.cli --models ./models download-models \
    --model scrfd
```

也可以同时指定：

```bash
python -m rope.cli --models ./models download-models \
    --model gfpgan \
    --model scrfd \
    --provider huggingface
```

当前自动下载器管理以下模型：

| `--model` 参数 | 本地文件 | 是否必需 |
|---|---|---|
| `retinaface` | `det_10g.onnx` | 是 |
| `arcface` | `w600k_r50.onnx` | 是 |
| `inswapper` | `inswapper_128.fp16.onnx` | 是 |
| `gfpgan` | `GFPGANv1.4.onnx` | 否 |
| `scrfd` | `scrfd_2.5g_bnkps.onnx` | 否 |

以下高级模型暂未纳入自动下载器，需要自行提供：

- `inswapper_256_phase1.onnx`
- `inswapper_512_level2.onnx`
- GPEN、CodeFormer、XSeg、FaceParser 等

基础图片和视频换脸不需要这些文件。

### 3.4 检查模型

完整计算并校验 SHA-256：

```bash
python -m rope.cli --models ./models check-models
```

使用文件大小和本地缓存快速检查：

```bash
python -m rope.cli --models ./models check-models --quick
```

输出：

```text
valid    det_10g.onnx                16.1 MiB  RetinaFace detector
valid    w600k_r50.onnx             166.3 MiB  ArcFace recognition
valid    inswapper_128.fp16.onnx     264.8 MiB  Inswapper 128 face swapper
```

校验结果缓存在 `models/.rope-models.json`。文件内容、大小或修改时间变化时会
重新计算 SHA-256。

### 3.5 首次换脸自动下载

图片和视频命令会在推理前自动检查 3 个基础模型。模型缺失或损坏时自动下载，
因此也可以跳过第 3 章，直接运行换脸命令。

禁用自动下载：

```bash
python -m rope.cli --no-auto-download --models ./models swap \
    --source face.jpg --target target.jpg --output out.jpg
```

无网络、模型缺失或损坏时，命令会明确失败，不会进入推理后才报晦涩错误。

### 3.6 模型来源

模型托管在 FaceFusion 模型资产仓库：

- Hugging Face：`facefusion/models-3.0.0`
- GitHub：`facefusion/facefusion-assets`

SHA-256 和文件大小已经写入项目模型清单。模型许可证与代码许可证不同，
请同时遵守对应模型的上游许可。

---

## 4. 环境自检

```bash
python tools/check_env.py --models ./models
```

自检内容：

- Python 版本
- NVIDIA 驱动
- `ffmpeg`、`ffprobe`、`curl`
- PyTorch CUDA 是否可用
- ONNX Runtime 是否包含 `CUDAExecutionProvider`
- Python 运行时依赖
- 必需模型及 SHA-256
- CLI 模块能否导入

成功输出：

```text
all checks passed
```

---

## 5. 图片换脸

### 5.1 最基础命令

```bash
python -m rope.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.png
```

默认替换目标图中检测到的所有人脸。

只替换最大的一张脸：

```bash
python -m rope.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg \
    --face-policy largest
```

### 5.2 低显存配置

```bash
python -m rope.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg \
    --face-policy largest \
    --swapper-size 128 \
    --detect-input-size 320
```

### 5.3 最快配置

```bash
python -m rope.cli --models ./models swap \
    --source face.jpg \
    --target target.jpg \
    --output out.jpg \
    --face-policy largest \
    --swapper-size 128 \
    --detect-input-size 320
```

加速因素：

- `--swapper-size 128`
- `--detect-input-size 320`
- `--face-policy largest`
- 输出 `.jpg` 而不是 `.png`
- 只传一张源脸
- 不启用修复、遮罩、色彩匹配等可选功能

### 5.4 多张源脸合并

```bash
python -m rope.cli --models ./models swap \
    --source face1.jpg \
    --source face2.jpg \
    --target target.jpg \
    --output out.png \
    --merge Sph
```

合并方式支持 `Mean`、`Median`、`Sph`、`Geo`、`Qual`。

### 5.5 批量处理

```bash
python -m rope.cli --models ./models swap \
    --source ./faces \
    --target-dir ./targets \
    --output-dir ./output
```

输出文件命名为 `<原文件名>_swapped.<扩展名>`。

### 5.6 检测目标图片中的脸

```bash
python -m rope.cli --models ./models inspect --target target.jpg
```

---

## 6. 视频换脸

### 6.1 只替换指定人物

需要三份输入：

| 输入 | 参数 | 含义 |
|---|---|---|
| 换脸源图 | `--source` | 最终要换成谁 |
| 人物参考图 | `--target-face` | 视频里要替换的是谁 |
| 目标视频 | `--target` | 待处理视频 |

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4
```

处理逻辑：

1. 从 `--source` 提取源人物特征
2. 从 `--target-face` 提取目标人物参考特征
3. 逐帧检测所有脸并计算身份相似度
4. 只替换与参考人物相似度达到阈值的脸

其他人脸保持不变。

可以传多张目标人物参考图：

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_front.jpg \
    --target-face person_side.jpg \
    --target input.mp4 \
    --output output.mp4
```

### 6.2 匹配阈值

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target-face person.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --target-threshold 60
```

- 范围为 `0-100`
- 默认值为 `55`
- 调高更严格，减少换错人风险
- 调低更容易匹配，但可能换到相似人物

### 6.3 替换视频中所有人脸

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target input.mp4 \
    --output output.mp4
```

### 6.4 低显存配置

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --swapper-size 128 \
    --detect-input-size 320 \
    --preset ultrafast \
    --crf 23
```

### 6.5 最快配置

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target-face person_in_video.jpg \
    --target input.mp4 \
    --output output.mp4 \
    --swapper-size 128 \
    --detect-input-size 320 \
    --preset ultrafast \
    --crf 23
```

额外优化：

- 不需要音频时增加 `--no-audio`
- 目标人物只需要一张最清晰的参考图
- 4K 或大码率视频先降到 720p
- 首次验证可以用 `--every-nth 20`

降到 720p：

```bash
ffmpeg -i input.mp4 -vf scale=-2:720 \
    -c:v libx264 -preset ultrafast -crf 23 \
    -c:a copy input_720p.mp4
```

`--every-nth 20` 只用于验证链路，并不是完整的逐帧换脸结果：

```bash
python -m rope.cli --models ./models video \
    --source new_face.jpg \
    --target test.mp4 \
    --output test_out.mp4 \
    --swapper-size 128 \
    --detect-input-size 320 \
    --preset ultrafast \
    --every-nth 20
```

### 6.6 视频时序处理

视频处理使用 PyAV 逐帧解码，并保留每帧原始 PTS。可变帧率视频的帧间距不会被
强制改写。音频使用 FFmpeg 从源视频复制，并恢复视频流和音频流各自的起始偏移。

因此在源视频同步正常的情况下，输出会保持原有音画关系。

---

## 7. 常用参数

全局参数必须写在子命令前面：

```bash
python -m rope.cli \
    --models /data/models \
    --model-provider hf-mirror \
    video --help
```

| 全局参数 | 说明 |
|---|---|
| `--models DIR` | 模型目录，默认 `./models` |
| `--model-provider` | 自动下载源，可重复 |
| `--no-auto-download` | 禁用模型自动下载 |

图片和视频共同参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--params FILE` | 加载旧参数文件 | 无 |
| `--detect-type` | `Retinaface` 或 `SCRDF` | `Retinaface` |
| `--detect-input-size` | `320/416/480/640` | `640` |
| `--detect-score` | 检测阈值 `1-100` | `50` |
| `--merge` | 多源脸合并方式 | `Mean` |
| `--swapper-size` | `128/256/512` | `128` |
| `--color-match` | 启用 LAB 色彩匹配 | 关闭 |
| `--backend` | `cuda` 或 `trt` | `cuda` |

查看所有参数默认值：

```bash
python -m rope.cli params
```

---

## 8. 常见问题

### 8.1 首次运行会自动下载模型吗

会。缺少或损坏基础模型时，`swap` 和 `video` 会自动下载。若希望离线运行或
完全控制模型来源，增加 `--no-auto-download`。

### 8.2 模型下载到一半中断

再次运行下载命令即可。已存在的 `.part` 文件支持继续传输，完整下载后会进行
SHA-256 校验。

### 8.3 Hugging Face 下载较慢

使用镜像：

```bash
python -m rope.cli --models ./models download-models \
    --provider hf-mirror \
    --provider github
```

### 8.4 `torch.cuda.is_available()` 返回 False

检查：

1. `nvidia-smi` 是否正常
2. 驱动版本是否匹配所选 `cu128/cu126/cu124`
3. 是否安装了 CPU 版 PyTorch

重新选择 CUDA 通道：

```bash
python tools/install.py cu126
```

### 8.5 ONNX Runtime 没有 CUDA Provider

重新运行 CUDA 安装器。安装器会先卸载其他 ONNX Runtime 变体：

```bash
python tools/install.py cu128
```

确认：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出必须包含 `CUDAExecutionProvider`。

### 8.6 没有检测到人脸

尝试：

- 使用清晰、正面、光线均匀的源脸
- 降低 `--detect-score`
- 图片保持 `--detect-input-size 640`
- 目标人物脸部过小时先提高视频分辨率

### 8.7 显存不足

依次尝试：

1. `--swapper-size 128`
2. `--detect-input-size 320`
3. 关闭修复、遮罩和色彩匹配
4. 将视频降到 720p
5. 避免多个进程同时使用同一 GPU

### 8.8 音画不同步

先确认源文件本身是否同步。如果源视频已经错位，可先强制转为恒定帧率：

```bash
ffmpeg -i input.mp4 -fps_mode cfr -r 30 \
    -af aresample=async=1 input_cfr.mp4
```

再用 `input_cfr.mp4` 作为 `--target`。

### 8.9 如何强制重新下载模型

```bash
python -m rope.cli --models ./models download-models --force
```

只强制某个模型：

```bash
python -m rope.cli --models ./models download-models \
    --model inswapper \
    --force
```

### 8.10 conda 环境缺少 CUDA Provider

如果环境里同时存在 conda 和 pip 安装的 `onnxruntime`，会出现这个错误。
检查并清理：

```bash
conda list onnxruntime
pip show onnxruntime-gpu
```

只保留 pip 版本：

```bash
conda remove --force onnxruntime
python tools/install.py cu128
```

验证：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出必须包含 `CUDAExecutionProvider`。

### 8.11 重建 conda 环境

环境损坏或需要更换 CUDA 版本时：

```bash
conda deactivate
conda env remove -n rope
sed -i 's/cuda-toolkit=12.8/cuda-toolkit=12.6/' environment.yml
conda env create -f environment.yml
conda activate rope
python tools/install.py cu126
```

模型文件保存在 `models/`，不会被删除，不需要重新下载。

---|---|
| `0` | 成功 |
| `1` | 未预期错误，会打印 traceback |
| `2` | 用户输入、模型或环境错误 |

---

### 8.12 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 未预期错误，会打印 traceback |
| `2` | 用户输入、模型或环境错误 |

---

## 9. 项目结构

```text
rope/
  cli.py             CLI 入口
  model_store.py     模型清单、下载和 SHA-256 校验
  headless.py        图片和视频换脸管线
  parameters.py      参数定义
  parameters_io.py   参数文件读写
  _default_data.py   参数默认值
  VideoManager.py    单帧换脸核心
  Models.py          ONNX Runtime 模型会话
  EmbeddingMerge.py  多源脸合并
  tests/             无需 GPU 的测试
tools/
  install.py         CUDA Python 依赖安装器
  check_env.py       云端环境自检
requirements.txt     headless Python 依赖
environment.yml      Conda 原生依赖环境
```

### 运行测试

这些测试不需要 GPU、模型或 OpenCV：

```bash
python -m pytest -q
```

也可以直接运行：

```bash
python -m rope.tests.test_cli
python -m rope.tests.test_model_store
python -m rope.tests.test_target_match
python -m rope.tests.test_timing
```

---

## 10. 免责声明

本项目仅用于合法、负责任和合乎伦理的用途。使用他人肖像前必须获得明确同意。

不得制作或传播伤害、诽谤、欺骗、骚扰他人的内容。使用者需要自行遵守所在地
关于隐私、肖像、深度合成、版权和数据处理的法律法规。

使用本项目即表示使用者理解并承担相应责任。项目作者不对误用、滥用或使用结果
造成的后果负责。

---

## 相关链接

- [FaceFusion](https://github.com/facefusion/facefusion)
- [FaceFusion 模型资产](https://github.com/facefusion/facefusion-assets)
- [Hugging Face 模型仓库](https://huggingface.co/facefusion/models-3.0.0)
- [InsightFace](https://github.com/deepinsight/insightface)
