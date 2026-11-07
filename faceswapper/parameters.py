"""headless CLI 的 typed parameter schema。

每一项都对应一个 key，faceswapper.VideoManager 会从 `self.parameters` 里读
它。磁盘格式没有变：`saved_parameters.json` 仍然是扁平的 {name: value} 映射，
所以当初从桌面 GUI 导出的文件在这里照样能加载。

key 的命名沿用 GUI 时代的 widget 后缀（`TextSel` / `Slider` / `Switch`）。
这是磁盘契约的一部分，改名会让已有的参数文件失效——见
docs/adr/0001-keep-gui-era-parameter-naming.md。

每个 param 只需提供两样东西：`default`（写进 parameter snapshot 的初值）
和 `kind`（决定 parameters_io 怎么把 JSON 值强转成 Python 类型）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from faceswapper._default_data import DEFAULT_DATA


@dataclass(frozen=True)
class SliderParam:
    name: str
    kind: Literal["slider"] = field(default="slider", init=False)

    @property
    def default(self) -> float: return DEFAULT_DATA[self.name + "Amount"]


@dataclass(frozen=True)
class SwitchParam:
    name: str
    kind: Literal["switch"] = field(default="switch", init=False)

    @property
    def default(self) -> bool: return bool(DEFAULT_DATA.get(self.name + "State", False))


@dataclass(frozen=True)
class SelectParam:
    name: str
    kind: Literal["select"] = field(default="select", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA[self.name + "Mode"]
    @property
    def modes(self) -> list[str]: return list(DEFAULT_DATA[self.name + "Modes"])


@dataclass(frozen=True)
class EntryParam:
    name: str
    kind: Literal["entry"] = field(default="entry", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA.get(self.name, "")


Parameter = SliderParam | SwitchParam | SelectParam | EntryParam


# CLI 暴露出来的 parameter。分组方式沿用桌面 GUI 的分组，去掉所有纯 GUI
# 的控制项。
PARAMETERS: list[Parameter] = [
    # ---- Restorer
    SwitchParam("RestorerSwitch"),
    SelectParam("RestorerTypeTextSel"),
    SelectParam("RestorerDetTypeTextSel"),
    SliderParam("RestorerSlider"),

    # ---- 相似度阈值
    SliderParam("ThresholdSlider"),

    # ---- 朝向
    SwitchParam("OrientSwitch"),
    SwitchParam("OrientAutoSwitch"),
    SliderParam("OrientSlider"),

    # ---- 强度
    SwitchParam("StrengthSwitch"),
    SliderParam("StrengthSlider"),
    SwitchParam("ColorMatchSwitch"),

    # ---- 相似度 / 保真度
    SliderParam("LikenessSlider"),
    SliderParam("EmbExtrapSlider"),
    SwitchParam("HighFidelitySwitch"),
    SliderParam("HighFidelityAlphaSlider"),
    SelectParam("HighFidelityModeTextSel"),

    # ---- 边界 / 混合
    SliderParam("BorderTopSlider"),
    SliderParam("BorderSidesSlider"),
    SliderParam("BorderBottomSlider"),
    SliderParam("BorderBlurSlider"),
    SliderParam("BlendSlider"),

    # ---- differencer / occluder / parser
    SwitchParam("DiffSwitch"),
    SliderParam("DiffSlider"),
    SwitchParam("OccluderSwitch"),
    SliderParam("OccluderSlider"),
    SwitchParam("DFLXSegSwitch"),
    SliderParam("DFLXSegSizeSlider"),
    SliderParam("DFLXSegBlurSlider"),
    SwitchParam("FaceParserSwitch"),
    SliderParam("FaceParserSlider"),
    SliderParam("MouthParserSlider"),

    # ---- 色彩
    SwitchParam("ColorSwitch"),
    SliderParam("ColorRedSlider"),
    SliderParam("ColorGreenSlider"),
    SliderParam("ColorBlueSlider"),
    SliderParam("ColorGammaSlider"),
    SliderParam("ColorContrastSlider"),
    SliderParam("ColorSaturationSlider"),

    # ---- 脸部微调
    SwitchParam("FaceAdjSwitch"),
    SliderParam("KPSXSlider"),
    SliderParam("KPSYSlider"),
    SliderParam("KPSScaleSlider"),
    SliderParam("FaceScaleSlider"),

    # ---- 检测
    SliderParam("ThreadsSlider"),
    SelectParam("DetectTypeTextSel"),
    SelectParam("DetectInputSizeTextSel"),
    SliderParam("DetectScoreSlider"),
    SelectParam("SwapperTypeTextSel"),

    # ---- 识别精度
    SwitchParam("RecognizeMirrorSwitch"),
    SwitchParam("RecognizeDimSwitch"),

    # ---- 匹配
    SliderParam("PoseCompensationSlider"),

    # ---- 检测
    SwitchParam("DetectRetrySwitch"),
    SelectParam("RecognizerTypeTextSel"),
    SelectParam("AlignTypeTextSel"),
]


PARAMETER_BY_NAME: dict[str, Parameter] = {p.name: p for p in PARAMETERS}


def default_values() -> dict[str, Any]:
    """返回所有 parameter 的 {name: default_value} 快照。"""
    return {p.name: p.default for p in PARAMETERS}



