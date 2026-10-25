"""headless CLI 各 parameter 的默认值。

schema 在 faceswapper/parameters.py，本模块存放它解析用的原始值。每个
parameter 只留 CLI 真正能兑现的那几个后缀：

    <Name>Amount    slider 的取值
    <Name>State     switch 的开 / 关
    <Name>Mode      select 当前选中的值
    <Name>Modes     select 的可选值（parameters_io 用它校验）

GUI-only 的 `*InfoText` 提示字符串与 `*Min` / `*Max` / `*Inc` 滑杆几何值
已随 Qt 前端一并移除——CLI 没有悬停帮助，滑杆范围由 argparse 自己的
choices 管。

`saved_parameters.json` 的磁盘格式没有变，所以当年从桌面 GUI 导出的文件
在这里照样能加载（含改名前的裸分辨率值，见 parameters_io._LEGACY_SWAPPER_MODES）。
"""

from __future__ import annotations

DEFAULT_DATA = {
    'RestorerSwitchState': False,
    'RestorerTypeTextSelMode': 'GFPGAN',
    'RestorerTypeTextSelModes': ['GFPGAN', 'CF', 'GPEN256', 'GPEN512'],
    'RestorerDetTypeTextSelMode': 'Blend',
    'RestorerDetTypeTextSelModes': ['Original', 'Blend', 'Reference'],
    'RestorerSliderAmount': 100,
    'ThresholdSliderAmount': 55,
    'OrientSwitchState': False,
    'OrientAutoSwitchState': False,
    'OrientSliderAmount': 0,
    'StrengthSwitchState': False,
    'StrengthSliderAmount': 100,
    'ColorMatchSwitchState': False,
    'LikenessSliderAmount': 100,
    'EmbExtrapSliderAmount': 0,
    'HighFidelitySwitchState': False,
    'HighFidelityAlphaSliderAmount': 50,
    'HighFidelityModeTextSelMode': 'Cached',
    'HighFidelityModeTextSelModes': ['Cached', 'Per-frame'],
    'BorderTopSliderAmount': 10,
    'BorderSidesSliderAmount': 10,
    'BorderBottomSliderAmount': 10,
    'BorderBlurSliderAmount': 10,
    'BlendSliderAmount': 5,
    'DiffSwitchState': False,
    'DiffSliderAmount': 4,
    'OccluderSwitchState': False,
    'OccluderSliderAmount': 0,
    'DFLXSegSwitchState': False,
    'DFLXSegSizeSliderAmount': 0,
    'DFLXSegBlurSliderAmount': 0,
    'FaceParserSwitchState': False,
    'FaceParserSliderAmount': 0,
    'MouthParserSliderAmount': 0,
    'ColorSwitchState': False,
    'ColorRedSliderAmount': 0,
    'ColorGreenSliderAmount': 0,
    'ColorBlueSliderAmount': 0,
    'ColorGammaSliderAmount': 1,
    'ColorContrastSliderAmount': 1,
    'ColorSaturationSliderAmount': 1,
    'FaceAdjSwitchState': False,
    'KPSXSliderAmount': 0,
    'KPSYSliderAmount': 0,
    'KPSScaleSliderAmount': 0,
    'FaceScaleSliderAmount': 0,
    'ThreadsSliderAmount': 5,
    'DetectTypeTextSelMode': 'Retinaface',
    'DetectTypeTextSelModes': ['Retinaface', 'SCRDF'],
    'DetectInputSizeTextSelMode': '640',
    'DetectInputSizeTextSelModes': ['320', '416', '480', '640'],
    'DetectScoreSliderAmount': 50,
    'SwapperTypeTextSelMode': 'inswapper-128',
    'SwapperTypeTextSelModes': ['inswapper-128', 'inswapper-256', 'inswapper-512', 'hyperswap-256'],
}
