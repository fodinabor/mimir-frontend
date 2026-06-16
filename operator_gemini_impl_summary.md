# Operator 实现状态

本文件更新于当前 TDD 修复之后。当前 frontend 已经把 elementwise 算子的基础路径稳定下来，并增加了 `model_to_mimir` utility 用于输出模型的 MimIR 表示。本轮重点修复了 binary broadcast 语义、去掉 hardcoded tensor annex id，并补充了 FileCheck 风格的 IR 顺序断言。

## 当前验证结果

| 命令 | 结果 | 说明 |
| --- | --- | --- |
| `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run --no-sync pytest tests/test_basic.py -q` | `97 passed, 1 skipped` | 当前全部核心算子测试通过。 |
| `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run --no-sync pytest -q -k 'not test_model_to_mimir_can_use_default_compile_phase'` | `100 passed, 1 skipped, 1 deselected` | 排除已知 `world.optimize()` native crash 后，其余项目测试通过。 |
| `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run pytest -q` | 已知会 crash | `test_model_to_mimir_can_use_default_compile_phase` 触发 MimIR `LowerMapReduce` native segfault，尚未修复。 |

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
| MimIR binding | 已补充 `World.sigma`，用于构造 `mean` reducer 所需的异构 product type；已补充 `Lit.get_nat()`，用于 Python frontend 读取静态 Nat literal 并实现可靠 broadcast 判断。 | `/Users/zc/courses/compiler/MimIR/py/bindings/world.cpp`, `/Users/zc/courses/compiler/MimIR/py/bindings/def.cpp` |
| MimIR dump utility | 已新增 `model_to_mimir`，支持 high-level tensor IR dump，并可选择加载 default compile/opt 插件。 | `src/mimir_frontend/utils.py`, `tests/test_utils.py` |
| where 和 clamp scalar | `where` 使用了 `tensor.select` binding enum 并修复了泛型实例化问题，`clamp` 支持 scalar bounds。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| max | 支持 value-only `torch.max(x)`，重载通过检查参数区分。 | `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| kInjective 算子 | 支持了 `cat`, `reshape`, `view`, `slice`, `select`, `squeeze`, `unsqueeze`, `split`, `clone`, `copy`。完成了 injective 算子的 100% 覆盖。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py` |
| kBroadcast 算子 | 支持了 `expand` 和 `full`。`full` 通过 `tensor.map` (ni=0) 构造，`expand` 基于 `tensor.broadcast` / `tensor.broadcast_in_dim`。binary broadcast 已按 PyTorch trailing-dim 规则计算 common shape，静态不兼容 shape 会抛 `NotImplementedError`。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| get_attr 和 getattr | 支持了 FX `get_attr` 节点（从 module 提取属性）和 `builtins.getattr` (用于 `x.shape` 提取)。 | `src/mimir_frontend/translator.py` |
| hardcoded annex id | 生产代码中的 `tensor.select` 和 `tensor.product_2d` 已全部改为 binding enum；`rg "0x5463|world\\.annex\\([0-9]" src tests` 当前无命中。 | `src/mimir_frontend/operators.py`, `tests/draft_ops_end.py` |
| FileCheck 风格测试基建 | 新增 `assert_ir_contains_in_order`，用于比较 IR 内容中的预期序列，避免只测 `isinstance`。 | `tests/test_basic.py` |

## 当前支持面

| 类别 | 已支持 / 已注册 operator | 实现状态 | 测试覆盖 |
| --- | --- | --- | --- |
| Elementwise binary | `add`、`sub`、`mul`、`div`、`maximum`、`minimum`、`clamp_min`、`clamp_max`、`bitwise_and`、`prims.fma` | 使用显式 rank/shape 的 `tensor.binary`。`clamp` 支持 scalar 输入。`bitwise_and` 基于 `core.bit2.and_`；`fma` 利用 `add/mul` 复合。 | 每个 operator 覆盖 static/dynamic + 1D/3D；`fma` 可根据环境版本跳过。 |
| Comparison | `eq`、`ne`、`lt`、`le`、`gt`、`ge` | 使用 `tensor.binary`，输入 `F32,F32`，输出 `Bool`。 | 每个 operator 覆盖 static/dynamic + 1D/3D，并断言结果 element type 是 `Bool`。 |
| Elementwise unary | `relu`、`exp`、`tanh`、`sqrt`、`abs`、`neg`、`sigmoid`、`reciprocal`、`rsqrt`、`logical_not`、`prims.convert_element_type` | 使用显式 rank/shape 的 `tensor.unary`；`logical_not` 基于 `core.bit1.neg`；`convert_element_type` 提供 Float/Bool 安全转换路径。 | 每个 operator 覆盖 static/dynamic + 1D/3D。 |
| Broadcast | `expand`、`full`、binary implicit broadcast | `expand` 使用 `%tensor.broadcast` / `%tensor.broadcast_in_dim`。`full` 通过 `%tensor.map` 以 `ni=0` 调用 lambda 构造常量 Tensor。binary operator 会先计算 PyTorch 风格 common shape，再对必要输入插入 broadcast。 | 覆盖 rank 3 到 rank 4 的 expansion、全量填充、leading singleton broadcast，以及 incompatible static shape fail-fast。 |
| Injective | `cat`, `reshape`, `view`, `slice`, `select`, `squeeze`, `unsqueeze`, `split`, `clone`, `copy` | 使用 `%tensor.reshape`, `%tensor.slice`, `%tensor.concat` 实现。`squeeze/unsqueeze` 映射到 `reshape`；`select` 映射到 `slice+squeeze`；`split` 映射到多个 `slice`。 | 覆盖 1D 到 4D 转换、多维切片、张量拼接及索引提取。 |
| Reduce | `sum`、`amax`、`mean`、`max` (value-only)、`var_mean` (correction=0) | 使用 `%tensor.map_reduce_aff`。`mean` 使用 `(sum, count)` accumulator。`var_mean` 使用 `(sum, sum_sq, count)` accumulator 并分别返回两个投影出的 tensor。 | static/dynamic + 1D/3D smoke；static 3D 额外检查 `So/Sr` shape 参数。 |
| Selection | `where` | 调用了 `tensor.select`。 | 覆盖 static/dynamic + 1D/3D，验证了 Bool condition tensor 和 shape 行为。 |
| Tuple | `operator.getitem` | 支持 Tuple 投影（用于多返回算子）以及 Tensor 切片/索引（转发至 `slice/select`）。 | 随 `var_mean` 和 `slice/select` 覆盖。 |

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
| `test_reshape_operator` | `static` | `3` | `reshape` | 验证 rank 变换。 |
| `test_slice_operator` | `static` | `3` | `slice` | 验证通过 getitem 进行的多维切片。 |
| `test_cat_operator` | `static` | `3` | `cat` | 验证张量拼接。 |
| `test_squeeze_unsqueeze_operator` | `static` | `3` | `squeeze/unsqueeze` | 验证 singleton 维度的增删。 |
| `test_select_operator` | `static` | `3` | `select` | 验证降维索引提取。 |
| `test_sequence_of_elementwise_operators` | `static`, `dynamic` | `1`, `3` | 算子序列 | 验证 chained elementwise translation。 |
| `test_binary_broadcast_leading_singleton_uses_common_output_shape` | `static` | `3` | binary implicit broadcast | 验证 `(2, 3, 4) + (1, 3, 4)` 生成 common output shape `(2, 3, 4)`，并按序出现 `%tensor.broadcast_in_dim` 和 `%tensor.binary`。 |
| `test_binary_broadcast_rejects_incompatible_static_shape` | `static` | `3` + `1` | binary implicit broadcast | 验证 `(2, 3, 4) + (5,)` 会抛 `NotImplementedError`，避免 silent wrong IR。 |
| `test_complex_operators_are_explicitly_unsupported` | `dynamic` | `3` | unsupported 算子 | 验证复杂算子不会生成错误 IR。 |
| `test_model_to_mimir_outputs_high_level_tensor_ir` | `dynamic` | `1` | utility dump | 验证 high-level tensor IR dump。 |
| `test_model_to_mimir_can_use_default_compile_phase` | `dynamic` | `1` | utility dump | 当前已知会触发 MimIR native segfault，作为待修复项保留。 |

## 对照 `operators_summary.md` 的覆盖情况

| Pattern Kind | 当前实现 | 当前测试状态 | 主要缺口 |
| --- | --- | --- | --- |
| `kElemWise` | 100% 完成。 | 100% 参数化覆盖。 | 无。 |
| `kBroadcast` | 100% 完成。 | 已覆盖 `expand/full` 和 binary implicit broadcast。 | dynamic symbolic broadcast 目前采用保守策略：无法静态证明不兼容时保留左侧符号维度。 |
| `kInjective` | 100% 完成。 | 核心操作已全部覆盖。 | `split` 尚未在 `test_basic.py` 中体现，但已实现。 |
| `kCommReduce` | 90% 完成。 | `sum/amax/mean/var_mean/max` 已覆盖。 | `any` 尚未处理；`max` 的 `(values, indices)` overload 尚未处理。 |
| `kOutEWiseFusable` | `mm` 已支持。 | 已有 `test_module_gen.py` 覆盖。 | `convolution` 仍需实现。 |
| `kTuple` | `operator.getitem` 已支持。 | 已覆盖。 | `topk` 等其他多返回算子。 |
| `kOpaque` | 未实现。 | 未覆盖。 | 建议维持 unsupported 策略。 |

## 已知限制和风险

| 风险 | 细节 | 建议 |
| --- | --- | --- |
| `compile_phase="default"` 的 native crash | 仍然存在于 `LowerMapReduce` 阶段。 | 待 MimIR core 修复或进一步排查符号化广播问题。 |
| dynamic broadcast 的符号等价判断 | 当前 Python frontend 无法证明两个不同 symbol 是否相等；对于无法静态判断的维度，broadcast common shape 会保守选择左侧维度。 | 后续如果 MimIR 侧有 shape constraint 表达，可以在 importer 中记录 symbol equality/compatibility。 |
| MimIR binding 依赖 | `Lit.get_nat()` 已在 `/Users/zc/courses/compiler/MimIR/py/bindings/def.cpp` 增加，并需要通过 CMake 重新构建 `mim_py` 后在 uv 环境中使用。 | 修改 binding 后运行 `cmake --build /Users/zc/courses/compiler/MimIR/build --target mim_py -j 8`，再安装 staged package。 |
| `split` 算子的动态性 | 目前仅支持 static extent 的分割。 | 遇到动态分割模型时再考虑增强。 |
