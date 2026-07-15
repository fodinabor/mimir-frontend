# Operator 实现状态

本文件更新于当前 TDD 修复之后。当前 frontend 已经把 elementwise 算子的基础路径稳定下来，并增加了 `model_to_mimir` utility 用于输出模型的 MimIR 表示。最新一轮补齐了真实 Inductor 图中的高频 `.Tensor/.Scalar` overload、`addmm` 分解、`where` scalar branch broadcast、`expand(-1, ...)`、`split.Tensor`，并新增了直接读取 `fx_graph_readable.py` 的集成测试和 dump 脚本。

## 当前验证结果

| 命令 | 结果 | 说明 |
| --- | --- | --- |
| `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run --no-sync pytest tests/test_basic.py tests/test_real_inductor_graphs.py -q` | `113 passed, 1 skipped` | 核心算子和真实 Inductor readable graph frontier 测试通过。 |
| `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run --no-sync pytest -q -k 'not test_model_to_mimir_can_use_default_compile_phase'` | `116 passed, 1 skipped, 1 deselected` | 排除已知 `world.optimize()` native crash 后，其余项目测试通过。 |
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
| Inductor overload 注册 | 已补齐 `aten.add/sub/mul/div.{Tensor,Scalar}` 与 `aten.eq/ne/lt/le/gt/ge.{Tensor,Scalar}` 等真实图高频 overload。 | `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| addmm | 已支持 `aten.addmm.default`，按 `mm(input, mat2) + bias` 分解，bias 通过现有 broadcast 语义处理。 | `src/mimir_frontend/translator.py`, `tests/test_basic.py`, `tests/test_real_inductor_graphs.py` |
| split.Tensor | 已支持 `aten.split.Tensor`，通过现有 `tensor.slice` 组合实现；静态 literal 维度可进行按 size 分块。 | `src/mimir_frontend/translator.py`, `src/mimir_frontend/operators.py`, `tests/test_basic.py`, `tests/test_real_inductor_graphs.py` |
| reduce 基础路径 | 已使用 `%tensor.map_reduce_aff` 支持 `sum`、`amax`、`mean`、`var_mean` 的 global、单维、多维、`keepdim=True/False` 形态。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| MimIR binding | 已补充 `World.sigma`，用于构造 `mean` reducer 所需的异构 product type；已补充 `Lit.get_nat()`，用于 Python frontend 读取静态 Nat literal 并实现可靠 broadcast 判断。 | MimIR 仓库 `py/bindings/world.cpp`, `py/bindings/def.cpp` |
| MimIR dump utility | 已新增 `model_to_mimir`，支持 high-level tensor IR dump，并可选择加载 default compile/opt 插件。 | `src/mimir_frontend/utils.py`, `tests/test_utils.py` |
| where 和 clamp scalar | `where` 使用了 `tensor.select` binding enum，支持 scalar branch broadcast 到 condition/common shape；`clamp` 支持 scalar bounds。 | `src/mimir_frontend/operators.py`, `tests/test_basic.py` |
| max | 支持 value-only `torch.max(x)`，重载通过检查参数区分。 | `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| kInjective 算子 | 支持了 `cat`, `reshape`, `view`, `slice`, `select`, `squeeze`, `unsqueeze`, `split`, `clone`, `copy`。完成了 injective 算子的 100% 覆盖。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py` |
| kBroadcast 算子 | 支持了 `expand` 和 `full`。`full` 通过 `tensor.map` (ni=0) 构造，`expand` 基于 `tensor.broadcast` / `tensor.broadcast_in_dim`，并支持 `-1` 保留输入维度。binary/where broadcast 已按 PyTorch trailing-dim 规则计算 common shape，静态不兼容 shape 会抛 `NotImplementedError`。 | `src/mimir_frontend/operators.py`, `src/mimir_frontend/translator.py`, `tests/test_basic.py` |
| get_attr 和 getattr | 支持了 FX `get_attr` 节点（从 module 提取属性）和 `builtins.getattr` (用于 `x.shape` 提取)。 | `src/mimir_frontend/translator.py` |
| hardcoded annex id | 生产代码中的 `tensor.select` 和 `tensor.product_2d` 已全部改为 binding enum；`rg "0x5463|world\\.annex\\([0-9]" src tests` 当前无命中。 | `src/mimir_frontend/operators.py`, `tests/draft_ops_end.py` |
| FileCheck 风格测试基建 | 新增 `assert_ir_contains_in_order`，用于比较 IR 内容中的预期序列，避免只测 `isinstance`。 | `tests/test_basic.py` |
| 真实 Inductor readable graph 测试 | 新增 loader，可直接从 `MIMIR_INDUCTOR_LOG_ROOT/*/fx_graph_readable.py` trace FX graph，并构造 MimIR placeholder。 | `src/mimir_frontend/inductor_readable.py`, `tests/test_real_inductor_graphs.py` |
| 真实图 MimIR dump 脚本 | 新增 `scripts/dump_inductor_mimir.py`，支持完整 dump；遇到 unsupported frontier 时可用 `--partial` 打印最后一个成功节点的 IR。 | `scripts/dump_inductor_mimir.py` |

## 当前支持面

| 类别 | 已支持 / 已注册 operator | 实现状态 | 测试覆盖 |
| --- | --- | --- | --- |
| Elementwise binary | `add`、`sub`、`mul`、`div`、`maximum`、`minimum`、`clamp_min`、`clamp_max`、`bitwise_and`、`prims.fma` | 使用显式 rank/shape 的 `tensor.binary`。`clamp` 支持 scalar 输入。真实 Inductor `.Tensor/.Scalar` overload 已注册。`bitwise_and` 基于 `core.bit2.and_`；`fma` 利用 `add/mul` 复合。 | 每个 operator 覆盖 static/dynamic + 1D/3D；额外覆盖真实 `torch.ops.aten.*.{Tensor,Scalar}` overload；`fma` 可根据环境版本跳过。 |
| Comparison | `eq`、`ne`、`lt`、`le`、`gt`、`ge` | 使用 `tensor.binary`，输入 `F32,F32`，输出 `Bool`。 | 每个 operator 覆盖 static/dynamic + 1D/3D，并断言结果 element type 是 `Bool`。 |
| Elementwise unary | `relu`、`exp`、`tanh`、`sqrt`、`abs`、`neg`、`sigmoid`、`reciprocal`、`rsqrt`、`logical_not`、`prims.convert_element_type` | 使用显式 rank/shape 的 `tensor.unary`；`logical_not` 基于 `core.bit1.neg`；`convert_element_type` 提供 Float/Bool 安全转换路径。 | 每个 operator 覆盖 static/dynamic + 1D/3D。 |
| Broadcast | `expand`、`full`、binary/where implicit broadcast | `expand` 使用 `%tensor.broadcast` / `%tensor.broadcast_in_dim`，`-1` 保留输入维度。`full` 通过 `%tensor.map` 以 `ni=0` 调用 lambda 构造常量 Tensor。binary/where 会先计算 PyTorch 风格 common shape，再对必要输入插入 broadcast；scalar-to-tensor broadcast 用 `%tensor.map`。 | 覆盖 rank 3 到 rank 4 的 expansion、`expand(-1, ...)`、全量填充、leading singleton broadcast、where scalar branch broadcast，以及 incompatible static shape fail-fast。 |
| Linear Algebra | `mm`、`addmm` | `mm` 使用 `tensor.product_2d`；`addmm` 分解为 `mm + add`。 | 覆盖 `addmm` 分解和真实 `mlp_1` readable graph 完整导入。 |
| Injective | `cat`, `reshape`, `view`, `slice`, `select`, `squeeze`, `unsqueeze`, `split`, `clone`, `copy` | 使用 `%tensor.reshape`, `%tensor.slice`, `%tensor.concat` 实现。`squeeze/unsqueeze` 映射到 `reshape`；`select` 映射到 `slice+squeeze`；`split` / `aten.split.Tensor` 映射到多个 `slice` 后打包为 tuple。 | 覆盖 1D 到 4D 转换、多维切片、张量拼接及索引提取。 |
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
| `test_real_aten_tensor_binary_overloads` | `dynamic` | `3` | Inductor `.Tensor` overload | 验证 `aten.add/sub/mul.Tensor` 直接通过真实 `torch.ops` target 命中。 |
| `test_real_aten_scalar_comparison_overloads_return_bool` | `dynamic` | `3` | Inductor `.Scalar` comparison overload | 验证 `aten.le/gt/eq.Scalar` 输出 Bool tensor。 |
| `test_real_aten_scalar_mul_overload` | `dynamic` | `3` | Inductor scalar arithmetic overload | 验证 `aten.mul.Scalar`。 |
| `test_addmm_decomposes_to_mm_and_add` | `static` | `1/2` | `addmm` | 验证 `addmm` 生成 `%tensor.product_2d` 后接 `%tensor.binary`。 |
| `test_where_broadcasts_scalar_branch_to_condition_shape` | `static` | `3` + scalar | `where` | 验证 scalar branch 通过 `%tensor.map` broadcast 到 condition shape。 |
| `test_expand_negative_one_keeps_input_dimension` | `static` | `2` | `expand(-1, ...)` | 验证 PyTorch `-1` 保留输入维度语义。 |
| `test_split_tensor_overload_returns_tuple_of_slices` | `static` | `2` | `split.Tensor` | 验证 `aten.split.Tensor` 生成 tuple，并能通过 `getitem` 取出 slice 继续参与后续算子。 |
| `test_real_inductor_mlp_forward_translates_after_addmm_support` | `dynamic` | readable graph | `mlp_1` | 验证真实 Inductor MLP forward readable graph 可以完整翻译。 |
| `test_real_inductor_lstm_forward_translates_after_split_support` | `dynamic` | readable graph | `lstm_1` | 验证真实 Inductor LSTM forward readable graph 可以完整翻译。 |
| `test_real_inductor_graph_frontier` | `dynamic` | readable graph | 多模型 frontier | 固化当前真实图能力边界：`gcn_1 -> aten.index.Tensor`、`moe_1 -> max.dim`、`faster_rcnn_1 -> convolution`。 |
| `test_complex_operators_are_explicitly_unsupported` | `dynamic` | `3` | unsupported 算子 | 验证复杂算子不会生成错误 IR。 |
| `test_model_to_mimir_outputs_high_level_tensor_ir` | `dynamic` | `1` | utility dump | 验证 high-level tensor IR dump。 |
| `test_model_to_mimir_can_use_default_compile_phase` | `dynamic` | `1` | utility dump | 当前已知会触发 MimIR native segfault，作为待修复项保留。 |

## 对照 `operators_summary.md` 的覆盖情况

| Pattern Kind | 当前实现 | 当前测试状态 | 主要缺口 |
| --- | --- | --- | --- |
| `kElemWise` | 100% 完成。 | 100% 参数化覆盖。 | 无。 |
| `kBroadcast` | 100% 完成。 | 已覆盖 `expand/full` 和 binary implicit broadcast。 | dynamic symbolic broadcast 目前采用保守策略：无法静态证明不兼容时保留左侧符号维度。 |
| `kInjective` | 100% 完成。 | 核心操作已全部覆盖。 | 无。 |
| `kCommReduce` | 90% 完成。 | `sum/amax/mean/var_mean/max` 已覆盖。 | `any` 尚未处理；`max` 的 `(values, indices)` overload 尚未处理。 |
| `kOutEWiseFusable` | `mm/addmm` 已支持。 | 已覆盖 `addmm` 分解和真实 `mlp_1`。 | `bmm` 尚未实现；`convolution` 如果继续排除 CNN 可暂缓。 |
| `kTuple` | `operator.getitem` 已支持。 | 已覆盖。 | `topk` 等其他多返回算子。 |
| `kOpaque` | 未实现。 | 未覆盖。 | 建议维持 unsupported 策略。 |

## 已知限制和风险

| 风险 | 细节 | 建议 |
| --- | --- | --- |
| `compile_phase="default"` 的 native crash | 仍然存在于 `LowerMapReduce` 阶段。 | 待 MimIR core 修复或进一步排查符号化广播问题。 |
| dynamic broadcast 的符号等价判断 | 当前 Python frontend 无法证明两个不同 symbol 是否相等；对于无法静态判断的维度，broadcast common shape 会保守选择左侧维度。 | 后续如果 MimIR 侧有 shape constraint 表达，可以在 importer 中记录 symbol equality/compatibility。 |
| MimIR binding 依赖 | `Lit.get_nat()` 已在 MimIR 仓库 `py/bindings/def.cpp` 增加，并需要通过 CMake 重新构建 `mim_py` 后在 uv 环境中使用。 | 修改 binding 后在 MimIR 仓库运行 `cmake --build build --target mim_py -j 8`，再安装 staged package。 |
| `split` 算子的动态性 | 目前仅支持 static extent 的分割。 | 遇到动态分割模型时再考虑增强。 |
| `mlp_0` native crash | `mlp_0` 在越过 `le.Scalar/where` 后触发 MimIR native crash，当前不放入 pytest 普通路径。 | 单独用 `scripts/dump_inductor_mimir.py mlp_0 --partial` 或最小化 IR 后排查 MimIR core / shape 表达。 |
| 非 CNN 下一批 blocker | 非 CNN 真实图下一批主要卡在 `index/scatter_add`、`max.dim`、`bmm`、random/dropout prims。 | 建议下一轮先做 `index.Tensor` + `scatter_add` 或 `max.dim`；Transformer 方向则优先 `bmm`。 |

## 对照 TVM Translator 的最新进度评估

本节对照以下两个基准实现重新评估当前 frontend 的位置：

- `tvm/python/tvm/relax/frontend/torch/base_fx_graph_translator.py`
- `tvm/python/tvm/relax/frontend/torch/exported_program_translator.py`

结论先说：

1. 我们在 `FX graph -> operator map -> high-level tensor IR` 这条主路径上已经站稳，尤其是 dynamic shape 的 shape source 已经比早期版本干净很多。
2. 我们和 TVM 的主要差距已经不在“有没有基本 importer”，而在“输入语义是否完整”和“operator surface 是否足够大”。
3. 当前实现更像一个面向 MimIR tensor plugin 的最小可用 importer，而不是像 TVM 那样覆盖 `ExportedProgram`、参数绑定、range constraint、控制流和大规模 operator registry 的通用 importer。

### 已经和 TVM 对齐的部分

| 维度 | TVM 做法 | 当前状态 | 结论 |
| --- | --- | --- | --- |
| 统一 `shape_of` 入口 | `shape_of` 优先看 `relax.Expr.struct_info.shape`，其次接受 `torch.Tensor.shape`。部分算子再回退到 `node.meta["val"]`。 | 我们已经建立统一 `shape_of`，优先读取 `mim.Def.type()`，同时支持直接对象 `.shape` 和 `fx.Node.meta["val"]`。 | 这一方向已经对齐，而且已经摆脱了 annotation/side-channel 主导 shape 的旧做法。 |
| operator map | TVM 使用大规模 `create_convert_map()` 注册 `aten`/`operator`/高阶 op。 | 我们已从 `if/elif` 切到 map 注册。 | 架构方向对齐，但覆盖面还远小于 TVM。 |
| dynamic shape 来源 | TVM 的符号 shape 来自 placeholder meta、`torch.SymInt`、struct info。 | 我们已经把 `FakeTensor/node.meta["val"]`、`mim.Def.type()`、annotation fallback 收敛到统一 shape source。 | 主路线正确，剩余问题集中在 SymInt expression 映射能力不够。 |
| 避免 shape annotation 驱动语义 | TVM 不依赖 readable annotation 作为真实 shape 语义来源。 | 我们已经把 readable annotation 降为 fallback。 | 这一点已明确纠偏。 |
| 多返回值/tuple 处理 | TVM 对 tuple output、`getitem`、部分多返回算子有稳定路径。 | 我们已支持 `operator.getitem`、`var_mean` 等 tuple 路径。 | 基础能力够用，但覆盖的多返回算子仍少。 |

### 明显落后于 TVM 的部分

| 维度 | TVM 做法 | 当前不足 | 影响 |
| --- | --- | --- | --- |
| `ExportedProgram` 支持 | TVM 有单独的 `ExportedProgramImporter`，直接处理 `graph_signature`、`input_specs`、`named_parameters`、`named_buffers`、`constants`。 | 我们目前只有 `fx.Graph` / `fx.GraphModule` 路径，没有真正的 `torch.export.ExportedProgram` importer。 | 目前仍然偏依赖 `symbolic_trace` 或 readable FX fixture，离真实生产输入还有一层。 |
| 参数/Buffer/Constant 语义 | TVM 会区分 `USER_INPUT`、`PARAMETER`、`BUFFER`、`CONSTANT_TENSOR`，并支持 bind params。 | 我们目前把 `get_attr` 提取到 trailing args，但没有 graph-signature 级别的输入分类，也没有常量绑定机制。 | utility 和测试可用，但模型入口语义不完整。 |
| range constraint | TVM 会从 `exported_program.range_constraints` 提取上下界，并挂到函数属性。 | 我们没有保存任何 symbol bound / equality / guard 信息。 | dynamic shape 目前只有“符号名保真”，没有约束语义。 |
| SymInt / Sympy expression | TVM 会把 `torch.SymInt` 映射为 `SizeVar`，并通过 `_process_derived_symbol` 处理派生符号。 | 我们当前 `ShapeEnv` 只稳定支持 literal 和“按名称复用 symbol”；复杂表达式大多仍会退化。 | `s0 + 1`、`2 * s0`、guarded dim 等情况还没有严肃支持。 |
| 控制流/高阶子图 | TVM 已处理 `cond` 分支子图导入，并缓存 branch function。 | 我们没有 `cond` / higher-order op 支持。 | 真实导出图一旦出现控制流，会直接中断。 |
| operator surface | TVM 的 convert map 覆盖面非常大，包括 NN、image、index/scatter、norm、loss、RNN、upsample 等。 | 我们当前仍聚焦 elementwise/broadcast/injective/reduce/mm/addmm/split。 | 当前能力更像“非 CNN、非复杂 indexing、非控制流”的子集 importer。 |
| dtype 处理 | TVM 有完整 `_convert_data_type` 和大量 dtype-sensitive 逻辑。 | 我们按当前目标明确暂不处理 dtype。 | 这是有意 scope 裁剪，但也意味着和 TVM 还不在同一完整度层级。 |

### 当前实现相对 TVM 的一个优势

有一件事目前我们的方向比 TVM 基线更贴近 MimIR 的约束，那就是 frontend 和 tensor plugin 的职责边界正在变清晰：

- TVM 许多 shape 语义最终落在 Relax op 的 `struct_info` / shape expr 上。
- 我们这边已经明确，像 `pad_shape`、`concat_shape`、`dot_general_shape` 这类 shape function 应优先由 MimIR tensor plugin 的返回类型承担。
- 这意味着我们后续不应该把 shape 逻辑重复硬编码在 frontend，而是把 frontend 限定为：
  - 提供正确的 `shape_of`
  - 正确构造 dependent input type
  - 在必须显式给出 `s_out` 的场景做最小必要 shape 计算

这条边界如果守住，长期维护成本会比继续在 frontend 手写 shape 推导更低。

### 重新评估当前进度

如果以 TVM 的两个 translator 为标杆，可以把当前进度分成四层：

| 层级 | 当前状态 | 评价 |
| --- | --- | --- |
| Level 0: 能翻译简单 FX 图 | 已完成 | 这部分已经稳定。 |
| Level 1: dynamic shape 主路径正确 | 基本完成 | `shape_of`、FakeTensor meta、dependent sigma、`ShapeEnv` 都已经到位。 |
| Level 2: 中等规模真实图可持续扩展 | 部分完成 | `mlp_1`、`lstm_1` 已打通，但 frontier 仍卡在 `index/max.dim/convolution`。 |
| Level 3: 类 TVM 的通用 importer | 明显未完成 | 缺 `ExportedProgram`、range constraints、控制流、参数绑定体系、复杂 operator surface。 |

更直接一点说：我们已经脱离“玩具 importer”，但距离 TVM 那种“通用前端”还有一整层基础设施要补。

### 当前最关键的不足

从 MimIR frontend 的现实优先级看，不足之处不是平均分布的，优先级很清楚：

1. `ExportedProgram` 缺失
2. SymInt expression / shape constraint 缺失
3. indexing/scatter/max.dim/bmm 这批真实模型 blocker
4. plugin-shape op 的系统性接入，例如 `pad`
5. `compile_phase="default"` 的稳定性问题

其中前两项属于“前端基础设施缺口”，后三项属于“能力面和可用性缺口”。

### 建议的下一阶段计划

| 优先级 | 事项 | 原因 |
| --- | --- | --- |
| P0 | 新增 `ExportedProgram -> MimIR` importer 骨架，先只支持 `USER_INPUT/PARAMETER/BUFFER/CONSTANT_TENSOR` 四类 input spec，不求一次补全全部 operator。 | 这是和 TVM 最大的架构差距，补上后输入语义会立刻正规化。 |
| P1 | 把 `ShapeEnv` 从“按名字复用 Nat”升级为“支持简单 affine SymInt expression 和约束记录”。 | 这决定 dynamic shape 能否继续往真实模型推进。 |
| P1 | 为 `index.Tensor`、`max.dim`、`bmm` 建立最小可用支持，继续推进真实图 frontier。 | 这是当前非 CNN 模型最直接的 blocker。 |
| P2 | 开始系统接入 plugin-shape operator，首批建议 `pad`。 | 这会验证“frontend 不重复 shape 逻辑”的边界是否真的成立。 |
| P2 | 给 `model_to_mimir` / importer 增加更严格的 IR FileCheck 风格测试，不只断言“能翻译”，而是断言 dependent function signature、symbol 复用和关键 op 序列。 | 当前测试方向是对的，但还可以继续向 TVM 那种结构性校验靠拢。 |
| P3 | 单独排查 `default compile phase` crash，确认是 frontend IR 非法还是 MimIR lowering bug。 | 这是稳定性问题，但前提是先把 high-level importer 能力继续往前推。 |

### 结论

对照 TVM 之后，当前 frontend 的定位可以明确成一句话：

> 我们已经完成了 MimIR 版 FX importer 的第一阶段，核心是“围绕 tensor plugin 的 high-level FX 导入 + 基本 dynamic shape 保真”；下一阶段不应再零散补小算子，而应先补 `ExportedProgram` 输入语义和 SymInt/constraint 基建，再继续推真实图 frontier。
