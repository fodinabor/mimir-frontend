# Mimir Frontend - Gemini Instructions

## Project Goal

我的目标是, 参照 tvm 中的 @/Users/zc/courses/compiler/ml-compiler/tvm/python/tvm/relax/frontend/torch/base_fx_graph_translator.py, 为我的 mimir 实现一个 fx.Graph importer, 最终目标是顺利 import /Users/zc/courses/compiler/pytorch-play/logs/attn_debug/inductor/*/fx_graph_readable.py，并且通过 mimir 的 dependent type 支持 dynamic （在 inductor 中由 `Sym("s")` 等来表示）

需要做的事情有：

1. 使用中文回答，除非我有明确的使用英文的指示
2. *永远*使用 uv 运行当前项目；使用 uv + pytest 进行测试
3. 使用 mimir py binding. 参照 @/Users/zc/courses/compiler/MimIR/py 和 @/Users/zc/courses/compiler/pytorch-play/pyproject.toml 的配置方式，配置 editable mimir 的 python binding 依赖， pytorch 不需要用 editable dep
4. 使用 @/Users/zc/courses/compiler/MimIR/src/mim/plug/tensor/tensor.mim 中的 axiom，在 python 中定义 add/mul/relu 等 **lambda**, 可参考 /Users/zc/courses/compiler/MimIR/lit/tensor/hlo.mim 中的 F32_add 的实现方式
   1. 使用 tensor.unary/binary 实现 elementwise 算子
   2. 使用 tensor.map_reduce/tensor.map_reduce_aff 实现 reduce 算子. 
   3. 使用 tensor.broadcast 实现 broadcast 算子
   4. 使用 tensor.pad + reduce + elementwise 实现 pooling 算子
5. 参考 @/Users/zc/courses/compiler/MimIR/lit/tensor/fuse_map.mim，使用部分 compile.phase，得到 high-level 的 mimir ir
6. 必要时可以往 mimpy 中添加新的 binding API 并且构建 mim_py 这个 target
7. 上述实现，同时支持 static shape 和 dynamic shape

## 测试和验收

1. 测试驱动开发
2. 不要为了测试通过而耍滑头，如果真的有很难解决的问题，向我汇报
3. 测试用例中，单测部分应包含 static/dynamic, 1d/3d 这些不同维度的组合
4. 集成测试用例应参照 /Users/zc/courses/compiler/pytorch-play/logs/attn_debug/inductor/*/fx_graph_readable.py 实现，可以照搬到当前项目中，或者截取其中一小部分并加上 TODO

## mim_py 使用

### 注意事项

1. 直接使用 math.F32 而**不是**手动重新定义 F32 的 (23, 8) 的更加底层的 bit-level 定义


---
*Note: This file was automatically generated to provide context for a new compiler frontend project.*
