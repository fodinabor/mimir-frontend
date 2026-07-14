# KernelBench Level 3 MimIR 导出报告

- KernelBench 目录: `/Users/zc/courses/compiler/ml-compiler/KernelBench/KernelBench/level3`
- 导出入口: `scripts/export_models_to_mimir.py`
- 模型约定: `Model` + `get_inputs()` + `get_init_inputs()`，或 `export_to_mim = export(...)`

## 总体统计

- 模型总数: 50
- 成功: 10
- 失败: 40

## 逐模型结果

| 模型 | 状态 | 阻塞点 |
| :--- | :--- | :--- |
| `10_ResNet101` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `11_VGG16` | SUCCESS | - |
| `12_VGG19` | SUCCESS | - |
| `13_DenseNet121TransitionLayer` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `14_DenseNet121DenseBlock` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `15_DenseNet121` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `16_DenseNet201` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `17_SqueezeNetFireModule` | SUCCESS | - |
| `18_SqueezeNet` | FAILED | max_pool2d ceil_mode=True is not implemented |
| `19_MobileNetV1` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `1_MLP` | SUCCESS | - |
| `20_MobileNetV2` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `21_EfficientNetMBConv` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `22_EfficientNetB0` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `23_EfficientNetB1` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `24_EfficientNetB2` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `25_ShuffleNetUnit` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `26_ShuffleNet` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `27_RegNet` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `28_VisionTransformer` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `29_SwinMLP` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `2_ShallowWideMLP` | SUCCESS | - |
| `30_SwinTransformerV2` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `31_VisionAttention` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `32_ConvolutionalVisionTransformer` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `33_VanillaRNN` | FAILED | Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors |
| `34_VanillaRNNHidden` | FAILED | 'Proxy' object cannot be interpreted as an integer |
| `35_LSTM` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `36_LSTMHn` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `37_LSTMCn` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `38_LSTMBidirectional` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `39_GRU` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `3_DeepNarrowMLP` | SUCCESS | - |
| `40_GRUHidden` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `41_GRUBidirectional` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `42_GRUBidirectionalHidden` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `43_MinGPTCausalAttention` | FAILED | slice indices must be integers or None or have an __index__ method |
| `44_MiniGPTBlock` | FAILED | slice indices must be integers or None or have an __index__ method |
| `45_UNetSoftmax` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `46_NetVladWithGhostClusters` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `47_NetVladNoGhostClusters` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `48_Mamba2ReturnY` | FAILED | No module named 'einops' |
| `49_Mamba2ReturnFinalState` | FAILED | No module named 'einops' |
| `4_LeNet5` | SUCCESS | - |
| `50_ReLUSelfAttention` | FAILED | slice indices must be integers or None or have an __index__ method |
| `5_AlexNet` | SUCCESS | - |
| `6_GoogleNetInceptionModule` | SUCCESS | - |
| `7_GoogleNetInceptionV1` | SUCCESS | - |
| `8_ResNetBasicBlock` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
| `9_ResNet18` | FAILED | symbolically traced variables cannot be used as inputs to control flow |
