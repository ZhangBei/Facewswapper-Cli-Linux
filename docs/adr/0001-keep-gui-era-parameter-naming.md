# 1. 保留 GUI 时代的参数命名，不重命名

日期：2026-10-23

## 状态

已接受。

## 背景

本项目是 Qt 桌面应用的后裔。GUI 已被整体移除（移除发生在版本控制建立
之前，git 历史里没有对应删除记录），但它留下了三样东西：

1. **扁平参数 dict 的 key 命名**——`SwapperTypeTextSel`、`ThresholdSlider`、
   `RestorerSwitch` 等 68 个 key，后缀 `TextSel` / `Slider` / `Switch` /
   `Button` 是 widget 类型。
2. **`saved_parameters.json` 的磁盘格式**——扁平的 `{name: value}` 映射，
   `--params` 至今在加载它，包含当年桌面 GUI 导出的文件。
3. **一批只服务 GUI 的代码**——预热链、模型清单表、VRAM 指示器、
   `*InfoText` 提示字符串、`*Min` / `*Max` / `*Inc` 滑杆几何值等。

## 决策

**删掉第 3 类（GUI 专用代码与数据），保留第 1、2 类（命名与磁盘契约）。**

具体地：

- GUI 专用代码整条移除：`VideoManager.preload_models` 及其 worker 池、
  `Models` 里只被它触达的预热/进度计数链、`MODEL_INVENTORY` 与
  `MODEL_INVENTORY_DETAILS` 两张表、`get_gpu_memory`、`unload_model`、
  `set_model_session_mode`、`set_session_mean_embedding`、
  `pipeline_sessions_loaded`。
- GUI 专用数据移除：`_default_data.py` 里 69 个 `*InfoText` 与
  51×3 个 `*Min` / `*Max` / `*Inc`；`parameters.py` 里零调用方的
  `info_text` / `min` / `max` / `inc` 属性与 `label` 字段；
  `SwapFacesButton`（从未被读取）、`MaskViewButton` 的 mask 预览分支、
  以及随之一同失去意义的整个 `control` dict 机制。
- 幽灵 key 从 schema 移除：`MergeTextSel`（真实合并在
  `headless` 的 `combine()` 里，走的是另一个变量）、
  `ModelSessionsTextSel`（零读取方）。老文件里写了这两个 key 的，
  沿用"未知 key 静默忽略"的既有行为。
- **68 个参数 key 的名字一个不改。**

## 后果

**好处**

- 约 900 行、占八个源文件约 14% 的代码消失，且不触碰任何功能路径。
- 每一处删除都有明确的"只有 GUI 会调它"证据（零调用方，或唯一调用方
  本身也是死代码）。

**代价**

- 代码里继续存在 `SwapperTypeTextSel` 这种 widget 后缀命名，读起来
  不像一个 CLI 原生的 schema。这是有意保留的。
- `params` 子命令输出的第二列仍是 `switch` / `slider` / `select` 这类
  widget 词汇。`kind` 同时是 `parameters_io._coerce` 做类型强制的真实
  分派依据，所以它是类型信息而非装饰。

**为什么不动命名**

重命名 68 个 key 需要同步修改 `cli.py` 的 flag 映射、
`parameters.py` 的注册表、`parameters_io.py` 的过滤与 `_coerce` 分派、
`VideoManager.swap_core` 里约 50 处读取、`model_store.specs_for_parameters`
的查表，以及三个测试模块的断言。为了名字好看付这个回归面不划算，而
`saved_parameters.json` 是 CLI 唯一的手写参数入口——改名要么破坏老文件，
要么引入一层永久别名表。

## 备选方案

**重命名为 CLI 原生风格**（`swapper_model` / `threshold` / …）：被否决。
回归面覆盖全部 CLI 功能，收益仅是命名美观；且需要一份旧名→新名别名表
才能继续加载老文件，等于把 GUI 命名永久留在代码里，只是换了个位置。

**保留全部 GUI 遗留代码**：被否决。与本次目标矛盾，且这些代码零调用方，
留着只会让人误以为预热/VRAM 指示器仍在工作。

**只删代码、连 GUI 专用数据也保留**（即保留 `*InfoText` /
`*Min` / `*Max` / `*Inc`）：被否决。这些值只被零调用方的
`info_text` / `min` / `max` / `inc` 属性读取，是纯死数据；留着等于
让 `_default_data.py` 的三分之二行数为不存在的东西服务。
