"""Typed parameter schema for the headless CLI.

Every entry maps to a key that faceswapper.VideoManager reads out of
`self.parameters`. The on-disk format is unchanged: `saved_parameters.json`
stays a flat {name: value} mapping, so a file exported from the original
desktop GUI still loads here.

Only parameters the CLI can actually honour are listed. The GUI-only
entries (buttons, icons, timeline and capture controls) were removed
along with the Qt frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from faceswapper._default_data import DEFAULT_DATA


ParameterKind = Literal["slider", "switch", "select", "entry"]


@dataclass(frozen=True)
class SliderParam:
    name: str
    label: str
    kind: Literal["slider"] = field(default="slider", init=False)

    @property
    def default(self) -> float: return DEFAULT_DATA[self.name + "Amount"]
    @property
    def min(self) -> float: return DEFAULT_DATA[self.name + "Min"]
    @property
    def max(self) -> float: return DEFAULT_DATA[self.name + "Max"]
    @property
    def inc(self) -> float: return DEFAULT_DATA[self.name + "Inc"]
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class SwitchParam:
    name: str
    label: str
    kind: Literal["switch"] = field(default="switch", init=False)

    @property
    def default(self) -> bool: return bool(DEFAULT_DATA.get(self.name + "State", False))
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class SelectParam:
    name: str
    label: str
    kind: Literal["select"] = field(default="select", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA[self.name + "Mode"]
    @property
    def modes(self) -> list[str]: return list(DEFAULT_DATA[self.name + "Modes"])
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class EntryParam:
    name: str
    label: str
    kind: Literal["entry"] = field(default="entry", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA.get(self.name, "")
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


Parameter = SliderParam | SwitchParam | SelectParam | EntryParam


# The parameters the CLI exposes. Grouped the way the desktop GUI grouped
# them, minus everything that was a pure GUI control.
PARAMETERS: list[Parameter] = [
    # ---- Restorer
    SwitchParam("RestorerSwitch", "Restorer"),
    SelectParam("RestorerTypeTextSel", "Restorer Type"),
    SelectParam("RestorerDetTypeTextSel", "Detection Alignment"),
    SliderParam("RestorerSlider", "Blend"),

    # ---- Similarity threshold
    SliderParam("ThresholdSlider", "Similarity Threshold"),

    # ---- Orientation
    SwitchParam("OrientSwitch", "Orientation"),
    SwitchParam("OrientAutoSwitch", "Auto-detect Orientation"),
    SliderParam("OrientSlider", "Angle"),

    # ---- Strength
    SwitchParam("StrengthSwitch", "Strength"),
    SliderParam("StrengthSlider", "Amount"),
    SwitchParam("ColorMatchSwitch", "Color Match (LAB)"),

    # ---- Likeness / fidelity
    SliderParam("LikenessSlider", "Likeness"),
    SliderParam("EmbExtrapSlider", "Distinctiveness"),
    SwitchParam("HighFidelitySwitch", "High Fidelity (2-pass)"),
    SliderParam("HighFidelityAlphaSlider", "HF Correction"),
    SelectParam("HighFidelityModeTextSel", "Mode:"),

    # ---- Border / blend
    SliderParam("BorderTopSlider", "Top Border Distance"),
    SliderParam("BorderSidesSlider", "Sides Border Distance"),
    SliderParam("BorderBottomSlider", "Bottom Border Distance"),
    SliderParam("BorderBlurSlider", "Border Blend"),
    SliderParam("BlendSlider", "Overall Mask Blend"),

    # ---- Differencer / occluder / parsers
    SwitchParam("DiffSwitch", "Differencing"),
    SliderParam("DiffSlider", "Difference Amount"),
    SwitchParam("OccluderSwitch", "Occluder"),
    SliderParam("OccluderSlider", "Occluder Size"),
    SwitchParam("DFLXSegSwitch", "DFL XSeg Mask"),
    SliderParam("DFLXSegSizeSlider", "XSeg Size"),
    SliderParam("DFLXSegBlurSlider", "XSeg Blur"),
    SwitchParam("FaceParserSwitch", "Face Parser"),
    SliderParam("FaceParserSlider", "Background"),
    SliderParam("MouthParserSlider", "Mouth"),

    # ---- Color
    SwitchParam("ColorSwitch", "Color Adjustments"),
    SliderParam("ColorRedSlider", "Red"),
    SliderParam("ColorGreenSlider", "Green"),
    SliderParam("ColorBlueSlider", "Blue"),
    SliderParam("ColorGammaSlider", "Gamma"),
    SliderParam("ColorContrastSlider", "Contrast"),
    SliderParam("ColorSaturationSlider", "Saturation"),

    # ---- Face adjustments
    SwitchParam("FaceAdjSwitch", "Input Face Adjustments"),
    SliderParam("KPSXSlider", "KPS - X"),
    SliderParam("KPSYSlider", "KPS - Y"),
    SliderParam("KPSScaleSlider", "KPS - Scale"),
    SliderParam("FaceScaleSlider", "Face Scale"),

    # ---- Threading / detection
    SliderParam("ThreadsSlider", "Threads"),
    SelectParam("ModelSessionsTextSel", "Model Sessions"),
    SelectParam("DetectTypeTextSel", "Detection Type"),
    SelectParam("DetectInputSizeTextSel", "Detect Input Size"),
    SliderParam("DetectScoreSlider", "Detect Score"),
    SelectParam("MergeTextSel", "Merge Math"),
    SelectParam("SwapperTypeTextSel", "Swapper Resolution"),
]


PARAMETER_BY_NAME: dict[str, Parameter] = {p.name: p for p in PARAMETERS}


def parameters_by_kind(kind: str) -> Iterable[Parameter]:
    return (p for p in PARAMETERS if p.kind == kind)


def default_values() -> dict[str, Any]:
    """Return a {name: default_value} snapshot of every parameter."""
    return {p.name: p.default for p in PARAMETERS}


def seed_control_dict() -> dict[str, Any]:
    """Control flags the core layer reads.

    `SwapFacesButton` gates whether a frame is actually swapped, and
    `MaskViewButton` would render a side-by-side mask preview instead of
    the finished frame. The CLI always wants the finished frame, so both
    are fixed here rather than exposed as user options.
    """
    return {
        "SwapFacesButton": True,
        "MaskViewButton": False,
    }
