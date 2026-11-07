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
    'DetectInputSizeTextSelModes': ['320', '416', '480', '640', '800', '960'],
    'DetectScoreSliderAmount': 50,
    # 识别时把裁剪图水平镜像再各跑一次 ArcFace、取平均，压低侧脸
    # embedding 的方差。默认关闭。
    'RecognizeMirrorSwitchState': False,
    # 识别用 dim=2 的网格平均（4 个 112x112 裁剪各跑一次 ArcFace 取
    # 平均），对侧脸单一裁剪容易切偏特征区的情况更稳。识别推理 x4，
    # 默认关闭。
    'RecognizeDimSwitchState': False,
    # 姿态自适应匹配阈值：--target-face 相似度筛选时，按检测关键点的
    # 正脸度下调判定线（全侧脸最多下调这个百分数）。0 = 关闭，判定与
    # 旧代码逐字等价。合理档 10-30。
    'PoseCompensationSliderAmount': 0,
    # 漏检重试：0 检出帧升一档检测输入尺寸（到 960）、抬高分数阈值
    # +20 分再检一次。只在本来不会换脸的帧上追加推理；auto-orient 开启
    # 时自动让位。默认关闭。
    'DetectRetrySwitchState': False,
    # B3：识别器权重。'w600k_r50'（默认，ResNet50）或 'glintr100'
    # （ResNet100，侧脸/难样本相似度更稳，识别推理约 x4，需下载
    # glintr100.onnx）。默认保持旧行为。
    'RecognizerTypeTextSelMode': 'w600k_r50',
    'RecognizerTypeTextSelModes': ['w600k_r50', 'glintr100'],
    # B4：识别对齐方式。'5-point'（默认，直接用检测 5 点，与 GUI 时代
    # 一致）或 '68-point'（2dfan4 landmark 对齐：检 68 点后取两眼中心/
    # 鼻尖/两嘴角，对极端侧脸的裁剪更准，需下载 2dfan4.onnx；每次识别
    # 前多一次 landmark 推理）。默认保持旧行为。
    'AlignTypeTextSelMode': '5-point',
    'AlignTypeTextSelModes': ['5-point', '68-point'],
    'SwapperTypeTextSelMode': 'inswapper-128',
    'SwapperTypeTextSelModes': ['inswapper-128', 'inswapper-256', 'inswapper-512',
                                'hyperswap-256', 'hyperswap-1b-256', 'hyperswap-1c-256'],
}
