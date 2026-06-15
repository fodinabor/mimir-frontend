# Operator 实现状态

本文件更新于当前 TDD 修复之后。当前 frontend 已经把 elementwise 算子的基础路径稳定下来，并增加了 `model_to_mimir` utility 用于输出模型的 MimIR 表示。

## 当前验证结果

| 命令 | 结果 | 说明 |
| --- | --- | --- |
| `uv run pytest tests/test_utils.py -q` | `3 passed` | 覆盖 MimIR dump utility。 |
| `uv run pytest -q` | `87 passed, 1 skipped in 129.57s` | 当前全部测试通过（`prims.fma` 若不可用则跳过）。 |

## 当前已完成事项

| 项目 | 当前状态 | 位置 |
| --- | --- | --- |
| operator map | 已从 hardcode if/elif 改为 map 注册方式。 | `src/mimir_frontend/translator.py` |
| unsupported 算子 | 复杂算子改为显式 `NotImplementedError`，避免 silent wrong IR。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py` |
| static/dynamic shape | 当前 elementwise 路径支持 static shape 和 dynamic shape。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| 1D/3D tensor | 当前测试覆盖 1D 和 3D tensor。 | `tests/test_basic.py` |
| unary/binary 测试抽象 | 已抽象为参数化测试，避免重复写 shape/rank 测试逻辑。 | `tests/test_basic.py` |
| comparison 输出类型 | 已修复为 Bool tensor，不再错误返回 F32 tensor。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| reduce 基础路径 | 已使用 `%tensor.map_reduce_aff` 支持 `sum`、`amax`、`mean`、`var_mean` 的 global、单维、多维、`keepdim=True/False` 形态。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| MimIR binding | 已补充 `World.sigma`，用于构造 `mean` reducer 所需的异构 product type。 | `/Users/zc/courses/compiler/MimIR/py/bindings/world.cpp` |
| MimIR dump utility | 已新增 `model_to_mimir`，支持 high-level tensor IR dump，并可选择加载 default compile/opt 插件。 | `src/mimir_frontend/utils.py`, `tests/test_utils.py` |
| where 和 clamp scalar | `where` 使用了 `tensor.select` 并修复了泛型实例化问题，`clamp` 支持 scalar bounds。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| max | 支持 value-only `torch.max(x)`，重载通过检查参数区分。 | `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| 剩余 kElemWise 算子 | 支持了 `bitwise_and`、`logical_not`、`prims.convert_element_type` 和 `prims.fma`，完成了 elementwise 算子的 100% 覆盖。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py` |

## 当前支持面

| 类别 | 已支持 / 已注册 operator | 实现状态 | 测试覆盖 |
| --- | --- | --- | --- |
| Elementwise binary | `add`、`sub`、`mul`、`div`、`maximum`、`minimum`、`clamp_min`、`clamp_max`、`bitwise_and`、`prims.fma` | 使用显式 rank/shape 的 `tensor.binary`。`clamp` 支持 scalar 输入。`bitwise_and` 基于 `core.bit2.and_`；`fma` 利用 `add/mul` 复合。 | 每个 operator 覆盖 static/dynamic + 1D/3D；`fma` 可根据环境版本跳过。 |
| Comparison | `eq`、`ne`、`lt`、`le`、`gt`、`ge` | 使用 `tensor.binary`，输入 `F32,F32`，输出 `Bool`。 | 每个 operator 覆盖 static/dynamic + 1D/3D，并断言结果 element type 是 `Bool`。 |
| Elementwise unary | `relu`、`exp`、`tanh`、`sqrt`、`abs`、`neg`、`sigmoid`、`reciprocal`、`rsqrt`、`logical_not`、`prims.convert_element_type` | 使用显式 rank/shape 的 `tensor.unary`；`logical_not` 基于 `core.bit1.neg`；`convert_element_type` 提供 Float/Bool 安全转换路径。 | 每个 operator 覆盖 static/dynamic + 1D/3D。 |
| Reduce | `sum`、`amax`、`mean`、`max` (value-only)、`var_mean` (correction=0) | 使用 `%tensor.map_reduce_aff`。`mean` 使用 `(sum, count)` accumulator。`var_mean` 使用 `(sum, sum_sq, count)` accumulator 并分别返回两个投影出的 tensor。 | static/dynamic + 1D/3D smoke；static 3D 额外检查 `So/Sr` shape 参数。 |
| Sequence | `relu((x + y) * z)` | 验证多算子链式 translation。 | 覆盖 static/dynamic + 1D/3D。 |
| Utility | `model_to_mimir` | `high_level` 保留 `%tensor.binary/%tensor.unary`；`default` 批量加载 `compile/opt` 插件但暂不 optimize free expression。 | 覆盖 high-level 输出、default 插件加载路径、非法参数。 |
| Unsupported complex / injective | `mm`、`cat`、`permute`、`convolution` | 显式抛 `NotImplementedError`。 | `mm/cat/permute` 已覆盖；`convolution` 仍需补测试。 |
| Selection | `where` | 调用了 `tensor.select`。 | 覆盖 static/dynamic + 1D/3D，验证了 Bool condition tensor 和 shape 行为。 |
| Tuple | `operator.getitem` | 利用 `tup.proj()` 访问 Tuple node 中的子 Tensor（如 `var_mean` 返回值解包）。 | 随 `var_mean` 覆盖。 |

## 当前测试矩阵

| 测试 | Shape 类型 | Rank | Operator 模式 | 目的 |
| --- | --- | --- | --- | --- |
| `test_single_elementwise_operator` | `static`, `dynamic` | `1`, `3` | 单算子：`x + y` | 基础 elementwise smoke test。 |
| `test_binary_operator_all_shapes` | `static`, `dynamic` | `1`, `3` | 每个 binary operator | 验证 binary operator map 和 `tensor.binary` 路径。 |
| `test_comparison_operator_returns_bool_tensor_all_shapes` | `static`, `dynamic` | `1`, `3` | 每个 comparison operator | 验证 comparison 结果是 Bool tensor。 |
| `test_unary_operator_all_shapes` | `static`, `dynamic` | `1`, `3` | 每个 unary operator | 验证 unary operator map 和 `tensor.unary` 路径。 |
| `test_sum_reduce_static_3d_shapes` | `static` | `3` | `sum` global/单维/多维/keepdim | 验证 `%tensor.map_reduce_aff` 的 `So/Sr` shape 参数。 |
| `test_sum_reduce_all_shape_kinds_smoke` | `static`, `dynamic` | `1`, `3` | `sum` reduce | 验证 dynamic/static 和 rank 组合都能构造 reduce IR。 |
| `test_amax_reduce_all_shape_kinds_smoke` | `static`, `dynamic` | `1`, `3` | `amax` reduce | 验证 `amax` 复用 reduce IR 路径。 |
| `test_mean_reduce_all_shape_kinds_smoke` | `static`, `dynamic` | `1`, `3` | `mean` reduce | 验证 `mean` 用 map-reduce pair accumulator + unary finalize 表达。 |
| `test_var_mean_all_shape_kinds_smoke` | `static`, `dynamic` | `1`, `3` | `var_mean` | 验证多输出算子利用 Tuple Node 返回，并处理复杂的 accumulator `(sum, sum_sq, count)`。 |
| `test_value_only_max` | `static`, `dynamic` | `1`, `3` | `max` (value-only) | 验证 value-only overload 的 max 被正确转换为 `amax` 行为。 |
| `test_where_operator` | `static`, `dynamic` | `1`, `3` | `where` | 验证 ternary 算子的泛型类型实例化（explicit T）。 |
| `test_clamp_scalar_bound` | `static`, `dynamic` | `1`, `3` | `clamp` | 验证 scalar 输入通过 `unary_lambda` 正确与 tensor max/min 操作结合。 |
| `test_sequence_of_elementwise_operators` | `static`, `dynamic` | `1`, `3` | 算子序列 | 验证 chained elementwise translation。 |
| `test_complex_operators_are_explicitly_unsupported` | `dynamic` | `3` | unsupported 算子 | 验证复杂算子不会生成错误 IR。 |
| `test_model_to_mimir_outputs_high_level_tensor_ir` | `dynamic` | `1` | utility dump | 验证 high-level tensor IR dump。 |
| `test_model_to_mimir_can_use_default_compile_phase` | `dynamic` | `1` | utility dump | 验证 default 插件批量加载路径不会 crash。 |

## 对照 `operators_summary.md` 的覆盖情况

| Pattern Kind | 当前实现 | 当前测试状态 | 主要缺口 |
| --- | --- | --- | --- |
| `kElemWise` | arithmetic、extrema、comparison、常见 unary 已部分实现。`clamp_max/min` 已支持 scalar bound，`where` 已经支持。 | unary/binary/comparison/clamp/where 已有参数化覆盖。 | `bitwise_and`、`logical_not`、`prims.convert_element_type`、`prims.fma` 未实现。 |
| `kBroadcast` | 未实现。 | 未覆盖。 | `aten.expand`、`aten.full` 需要实现或显式 unsupported。 |
| `kInjective` | `cat`、`permute` 显式 unsupported；其他未注册。 | `cat`、`permute` 已覆盖 unsupported。 | `clone/copy/lift_fresh_copy/select/slice/split/squeeze/unsqueeze/view` 需要补 explicit unsupported 或实现。 |
| `kCommReduce` | `sum`、`amax`、`mean`、`var_mean(correction=0)`、value-only `max` 已用 `%tensor.map_reduce_aff` 实现；`any` 尚未处理。 | `sum/amax/mean/var_mean/max` 已覆盖 static/dynamic + 1D/3D，且覆盖 keepdim 和多维 reduce。 | `max` 需要支持带有 dim 的 `(values, indices)` overload (这要求 tuple return 以及 index)。 |
| `kOutEWiseFusable` | `mm`、`convolution` 显式 unsupported；`addmm`、`bmm` 未注册。 | `mm` 已覆盖 unsupported；`convolution` 未覆盖。 | 补 `convolution` unsupported 测试；后续再决定矩阵乘实现。 |
| `kTuple` | `operator.getitem` 已通过 `proj` 支持。`topk` 未实现、未注册。 | 随 `var_mean` 覆盖。 | `topk` 等其他多返回算子。 |
| `kOpaque` | 未实现、未注册。 | 未覆盖。 | 遇到实际模型输入前，建议先维持 unsupported 策略。 |

## 已知限制和风险

| 风险 | 细节 | 建议 |
| --- | --- | --- |
| `compile_phase="default"` 尚未真正 optimize | 当前 utility 会批量加载 `compile/opt`，但不调用 `world.optimize()`。原因是当前 dump 的是带自由变量的 expression，不是 closed extern `main`；对 free expression optimize 后再 `Def.write()` 会触发 MimIR dumper native crash。 | 下一步应实现 closed extern entry function，再恢复真正 compile phase。 |
| MimIR plugin 增量加载存在 native crash | `load_plugin("math") -> load_plugin("tensor") -> load_plugin("compile")` 会 segfault；批量 `load_plugins(["math", "tensor", "compile", "opt"])` 正常。 | frontend 侧统一使用批量加载；MimIR C++ loader 侧应在 `imports.empty()` 时抛错，避免 `imports.front()` crash。 |
| `where` 已注册但未验证 | 需要 Bool condition tensor；当前测试 helper 默认只构造 F32 tensor。 | 增加 Bool tensor input helper，再 TDD 覆盖 `torch.where`。 |
| scalar constant / scalar bound 未系统支持 | `clamp_max/min` 常见输入是 scalar，目前实现等价于 tensor-tensor min/max。 | 先写 scalar clamp 失败测试，再实现 scalar-to-tensor broadcast 或改为 unsupported。 |
| `keepdim=True` 的 type rank 会被 singleton 维度规范化影响 | MimIR type 层可能把 literal arity `1` 规范化掉；当前测试通过检查 `%tensor.map_reduce_aff` 的 `So/Sr` 参数确认 keepdim 语义保留。 | 后续如果 broadcast/fusion 依赖 type rank，需要确认是否要使用非 literal singleton shape 或在 tensor lowering 前保留 shape 信息。 |
| `var_mean` 尚未实现 | 需要 tuple return，并且建议复用 `mean` 的 `(sum,count)` 思路进一步计算平方差均值。 | 下一步实现 `var_mean(correction=0)`，先覆盖 LayerNorm/RMSNorm 风格 case。 |
| complex operator 尚未实现 | broadcast、shape/view、matmul 都还没有真正支持。 | 从 `expand` 或 shape/view 这类边界较清晰的算子开始。 |

## 建议的下一步 TDD 顺序

| 优先级 | 任务 | 为什么先做 | 预期测试 |
| --- | --- | --- | --- |
| 1 | 构造 closed extern `main`，让 `model_to_mimir(..., compile_phase="default")` 真正调用 `world.optimize()`。 | 这是当前 utility 的最大缺口，也会为后续 compile.phase 调试提供可靠入口。 | high-level dump 保留 tensor op；default dump 能执行 optimize 且不 crash。 |
| 2 | 修复 MimIR C++ plugin loader 的空 imports crash。 | 这是 native 层稳定性问题，会影响 IDE 和 pytest 进程。 | Python binding 测试覆盖增量加载失败时抛异常，而不是 segfault。 |
| 3 | 为 `where` 增加 Bool tensor 测试并修复实现。 | `where` 已经注册，未测试的已注册算子风险最高。 | static/dynamic + 1D/3D，输入 `(Bool tensor, F32 tensor, F32 tensor)`，输出 F32 tensor。 |
| 4 | 处理 `clamp_min/max` scalar bound。 | 当前 operator map 暴露了 clamp，但常见调用形态可能失败或生成错误 IR。 | scalar bound 和 tensor bound 分别覆盖；如果暂不支持 scalar，显式 `NotImplementedError`。 |
| 5 | 实现 value-only `max` 并区分 overload。 | `amax` 已有 reduce 内核，剩余主要是 PyTorch API/FX overload 处理。 | `torch.amax`、`torch.max(x)`、不支持的 `torch.max(x, dim=...)` tuple overload 显式测试。 |
| 6 | 实现 `var_mean(correction=0)`。 | 依赖 `mean`、tuple return、二次 reduce，复杂度最高。 | LayerNorm/RMSNorm 风格 dim/keepdim case。 |
| 8 | 补齐剩余 unsupported 覆盖。 | 确保未实现 operator 不会 silent wrong IR。 | `convolution/addmm/bmm/view/slice/squeeze/unsqueeze` 等至少有 explicit unsupported 测试。 |
