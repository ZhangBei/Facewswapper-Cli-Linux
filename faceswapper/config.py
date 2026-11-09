"""CLI-native 的 typed 配置对象（风格化重构的边界切片）。

目标：让「以后写新代码时面对的形状」与 GUI 遗产脱钩，同时**不改变任何现有
流程**。磁盘 saved_parameters.json 仍是 54 个 GUI 时代 key；VideoManager 仍读
self.parameters 这个扁平 dict；本模块只在两者之间做一次显式翻译。

因此本模块当前**未接入生产路径**：headless._apply_parameters 仍是唯一写入点，
本模块只提供 from_parameters / to_parameters 两个纯函数，等价性由
tests/test_config.py 的 54 key 往返测试守住。接入与否、何时接入由 ADR 0002
决定，在此之前删除本模块不会影响任何行为。

风格约定（见 CONTEXT.md）：领域术语保留英文原词；标识符用 snake_case；
默认值只有一个真值源（parameters.default_values），这里不重复任何常量；
映射表显式写出而非用命名规律推导——猜错会静默丢字段，写错则能被测试抓住。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faceswapper.parameters import default_values

__all__ = ["FIELD_MAP", "SwapConfig"]


# 旧 key -> 新字段。这是全仓库唯一允许集中出现 GUI 后缀的地方之一
# （另一处是 parameters_io.py 的磁盘读写）。少数名字按 CLI 语义而非机械
# 去后缀来命名：SwapperTypeTextSel -> swapper_model 对齐 --swapper-model，
# RecognizerTypeTextSel -> recognizer 对齐 --recognizer。
FIELD_MAP: dict[str, str] = {
    # ---- Restorer
    "RestorerSwitch": "restorer_enabled",
    "RestorerTypeTextSel": "restorer_type",
    "RestorerDetTypeTextSel": "restorer_det_type",
    "RestorerSlider": "restorer",
    # ---- 相似度阈值
    "ThresholdSlider": "threshold",
    # ---- 朝向
    "OrientSwitch": "orient_enabled",
    "OrientAutoSwitch": "orient_auto_enabled",
    "OrientSlider": "orient",
    # ---- strength / 色彩匹配
    "StrengthSwitch": "strength_enabled",
    "StrengthSlider": "strength",
    "ColorMatchSwitch": "color_match_enabled",
    # ---- 相似度 / 保真度
    "LikenessSlider": "likeness",
    "EmbExtrapSlider": "emb_extrap",
    "HighFidelitySwitch": "high_fidelity_enabled",
    "HighFidelityAlphaSlider": "high_fidelity_alpha",
    "HighFidelityModeTextSel": "high_fidelity_mode",
    # ---- 边界 / 混合
    "BorderTopSlider": "border_top",
    "BorderSidesSlider": "border_sides",
    "BorderBottomSlider": "border_bottom",
    "BorderBlurSlider": "border_blur",
    "BlendSlider": "blend",
    # ---- differencer / occluder / parser
    "DiffSwitch": "diff_enabled",
    "DiffSlider": "diff",
    "OccluderSwitch": "occluder_enabled",
    "OccluderSlider": "occluder",
    "DFLXSegSwitch": "dflxseg_enabled",
    "DFLXSegSizeSlider": "dflxseg_size",
    "DFLXSegBlurSlider": "dflxseg_blur",
    "FaceParserSwitch": "face_parser_enabled",
    "FaceParserSlider": "face_parser",
    "MouthParserSlider": "mouth_parser",
    # ---- 色彩
    "ColorSwitch": "color_enabled",
    "ColorRedSlider": "color_red",
    "ColorGreenSlider": "color_green",
    "ColorBlueSlider": "color_blue",
    "ColorGammaSlider": "color_gamma",
    "ColorContrastSlider": "color_contrast",
    "ColorSaturationSlider": "color_saturation",
    # ---- 脸部微调
    "FaceAdjSwitch": "face_adj_enabled",
    "KPSXSlider": "kpsx",
    "KPSYSlider": "kpsy",
    "KPSScaleSlider": "kps_scale",
    "FaceScaleSlider": "face_scale",
    # ---- 检测 / swapper
    "ThreadsSlider": "threads",
    "DetectTypeTextSel": "detect_type",
    "DetectInputSizeTextSel": "detect_input_size",
    "DetectScoreSlider": "detect_score",
    "SwapperTypeTextSel": "swapper_model",
    # ---- 识别精度
    "RecognizeMirrorSwitch": "recognize_mirror_enabled",
    "RecognizeDimSwitch": "recognize_dim_enabled",
    # ---- 匹配 / 鲁棒性
    "PoseCompensationSlider": "pose_compensation",
    "DetectRetrySwitch": "detect_retry_enabled",
    "RecognizerTypeTextSel": "recognizer",
    "AlignTypeTextSel": "align_type",
}


@dataclass(frozen=True)
class SwapConfig:
    """一次运行生效的参数集合，CLI-native 命名。

    字段没有默认值：唯一真值源是 ``parameters.default_values()``，构造请走
    ``default()`` 或 ``from_parameters()``，避免两处默认值漂移。类型标注表达
    期望（slider 的历史默认值可能是 int），本类不做运行时类型校验——强转是
    ``parameters_io.load`` 的职责。
    """

    # ---- Restorer
    restorer_enabled: bool
    restorer_type: str
    restorer_det_type: str
    restorer: float
    # ---- 相似度阈值
    threshold: float
    # ---- 朝向
    orient_enabled: bool
    orient_auto_enabled: bool
    orient: float
    # ---- strength / 色彩匹配
    strength_enabled: bool
    strength: float
    color_match_enabled: bool
    # ---- 相似度 / 保真度
    likeness: float
    emb_extrap: float
    high_fidelity_enabled: bool
    high_fidelity_alpha: float
    high_fidelity_mode: str
    # ---- 边界 / 混合
    border_top: float
    border_sides: float
    border_bottom: float
    border_blur: float
    blend: float
    # ---- differencer / occluder / parser
    diff_enabled: bool
    diff: float
    occluder_enabled: bool
    occluder: float
    dflxseg_enabled: bool
    dflxseg_size: float
    dflxseg_blur: float
    face_parser_enabled: bool
    face_parser: float
    mouth_parser: float
    # ---- 色彩
    color_enabled: bool
    color_red: float
    color_green: float
    color_blue: float
    color_gamma: float
    color_contrast: float
    color_saturation: float
    # ---- 脸部微调
    face_adj_enabled: bool
    kpsx: float
    kpsy: float
    kps_scale: float
    face_scale: float
    # ---- 检测 / swapper
    threads: float
    detect_type: str
    detect_input_size: str
    detect_score: float
    swapper_model: str
    # ---- 识别精度
    recognize_mirror_enabled: bool
    recognize_dim_enabled: bool
    # ---- 匹配 / 鲁棒性
    pose_compensation: float
    detect_retry_enabled: bool
    recognizer: str
    align_type: str

    @classmethod
    def default(cls) -> "SwapConfig":
        """全部取 schema 默认值。"""
        return cls.from_parameters({})

    @classmethod
    def from_parameters(cls, params: dict[str, Any]) -> "SwapConfig":
        """从扁平 parameter dict 构造；未知 key 静默忽略（沿用既有约定）。

        ``params`` 应是 ``parameters_io.load`` 清理过的值，或 ``SwapOptions``
        的 overrides。缺失的 key 取 schema 默认值，所以调用方只写差异即可。
        """
        merged = default_values()
        merged.update({k: v for k, v in params.items() if k in FIELD_MAP})
        return cls(**{field: merged[old] for old, field in FIELD_MAP.items()})

    def to_parameters(self) -> dict[str, Any]:
        """投影回 VideoManager 读的扁平 dict，顺序与 schema 一致。"""
        return {old: getattr(self, field) for old, field in FIELD_MAP.items()}
