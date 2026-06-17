# Dynamic Shape Plan

## 背景

参考资料：

- PyTorch Dynamic Shapes: https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html
- TVM importer 标杆：`/Users/zc/courses/compiler/ml-compiler/tvm/python/tvm/relax/frontend/torch/base_fx_graph_translator.py`
- TVM Relax: `/Users/zc/Readings/relax.pdf`
- MimIR tensor axioms: `/Users/zc/courses/compiler/MimIR/src/mim/plug/tensor/tensor.mim`

当前阶段暂停额外算子覆盖，等待 MimIR 侧 `gather`、`scatter`、`conv2d` 等 axiom。本文只规划不依赖这些新 axiom 的 dynamic shape 基建。

## 核心修正

shape 的语义来源不应该是 `fx_graph_readable.py` 的 annotation。

正确方向应参考 TVM 的 `BaseFXGraphImporter.shape_of`：

1. 如果输入已经是 IR 表达式，从表达式自身的类型/struct info 读取 shape。
2. 如果输入还是 `torch.Tensor` / `FakeTensor`，从 `tensor.shape` 读取 shape，其中动态维度可能是 `torch.SymInt`。
3. 某些 fallback 场景可以从 FX node 的 `meta["val"]` 读取 FakeTensor shape。
4. operator 内部做 shape 计算时，直接操作这些符号 shape 表达式，不尝试取具体整数值。

映射到当前项目：

1. `mim.Def` 的 tensor type 是首要 shape source，等价于 TVM `relax.Expr.struct_info.shape`。
2. `torch.Tensor` / `FakeTensor` / `node.meta["val"]` 是从 PyTorch 图导入时的 shape source。
3. `model_to_mimir(input_shapes=...)` 是用户显式输入签名，用于创建 placeholder 的初始 MimIR tensor type。
4. `fx_graph_readable.py` annotation 只能作为 readable fixture 的输入类型合成 fallback，不能成为 operator shape 语义来源。

## Scope

本阶段要做：

1. 建立类似 TVM `shape_of(x)` 的统一入口，抹平 `mim.Def`、`torch.Tensor`、`FakeTensor`、FX node meta 的差异。
2. 保留 PyTorch `SymInt` / Sympy shape expression 的 identity，不要过早退化成 `top_nat()`。
3. 明确哪些 shape 计算由 frontend 做，哪些交给 tensor plugin 的 shape axiom 做。
4. 让测试从“能翻译”升级为“IR 中的 shape 语义正确”。
5. 确保 shape 错误尽量在 Python frontend 层以 `ValueError` / `NotImplementedError` 暴露，而不是进入 MimIR native crash。

本阶段不做：

1. 不考虑 dtype。
2. 不实现依赖新 axiom 的 `gather`、`scatter`、`conv2d`。
3. 不做完整 Presburger / SMT shape solver。
4. 不要求支持任意复杂的 PyTorch `SymExpr`，复杂表达式可以先保守降级并记录诊断。

## 当前状态

### 已有能力

1. `model_to_mimir(..., input_shapes=[("n", ...)])` 可以创建 Nat shape 参数。
2. `Translator.translate_as_function` 支持把 Nat 参数和 tensor 参数一起放进函数 domain。
3. `MimirOperators._shape_dims` 可以从 `mim.Def.type()` 读取 tensor shape。
4. `MimirOperators._shape_dims` 目前还能通过 `input_to_syms` 把输入 tensor 的符号维度恢复成 MimIR Nat def。
5. `Lit.get_nat` 已经可以读静态 Nat literal，用于 static shape 判断和测试。
6. elementwise、broadcast、where、expand、reduce、reshape、slice、split 已经有基础 shape 构造逻辑。
7. `inductor_readable.py` 可以加载真实 Inductor `fx_graph_readable.py`，绕过重新跑 Inductor。

### 主要缺口

1. 当前没有统一的 `shape_of(x)`，operator 直接调用 `_shape_dims`，输入类型被限制在 `mim.Def`。
2. `inductor_readable.py` 目前把 annotation 当作主要来源合成输入，且遇到非十进制维度时经常使用 `world.top_nat()`，会丢失 `s0 == s0`。
3. `model_to_mimir` 构造 tensor type 时仍然用 `top_nat` 表示动态维度，符号参数和 tensor 类型之间主要靠 `input_to_syms` 事后恢复。
4. shape 推导分散在 `operators.py` 各个 operator 方法中，缺少统一错误信息和统一 shape expression 表示。
5. broadcast 现在是保守实现：静态不兼容会报错，动态场景大多放行，但不能表达或记录“这两个动态维度应相等”。
6. reshape 只构造 `tensor.reshape`，没有在 frontend 层验证明显的静态元素个数错误。
7. 默认 compile phase 仍有已知 native crash，dynamic shape 测试应优先覆盖 high-level tensor IR。

## Shape 信息的 Provider 和 Consumer

### Provider

按优先级：

1. `mim.Def.type()`：当值已经是 MimIR 表达式时，从类型读取 shape。这是 frontend 内部 operator 的标准来源。
2. `torch.Tensor.shape` / `FakeTensor.shape`：当 FX 参数或 metadata 中仍保留 PyTorch tensor 对象时，读取其 shape，动态维度可能是 `torch.SymInt`。
3. `node.meta["val"].shape`：作为 fallback，用于处理 `_expand` 这类需要原输入 shape 但当前表达式 shape 不够完整的场景。
4. 用户显式 `input_shapes`：仅用于创建 placeholder 初始类型和符号参数。
5. `fx_graph_readable.py` annotation：仅用于没有 FakeTensor/meta 时的测试 fixture 输入类型合成，不作为长期语义来源。

### Consumer

1. MimIR tensor type 构造。
2. `tensor.unary` / `tensor.binary` / `tensor.broadcast` / `tensor.broadcast_in_dim` / `tensor.map_reduce_aff` 等 axiom 的 shape 参数。
3. tensor plugin 中已有 shape axiom，例如 `pad_shape`、`concat_shape`、`dot_general_shape`、`transpose_shape`。
4. frontend shape checker，用于提前发现明显 shape 错误。
5. 后端 bufferization、tiling、vectorization。当前只保留足够信息，不直接实现这些 consumer。

## Frontend 和 Tensor Plugin 的职责边界

### Frontend 应该做

1. 提供统一 `shape_of(x)`。
2. 把 PyTorch `SymInt` / Sympy expression 映射到 MimIR Nat def 或 Nat expression。
3. 做 PyTorch 语义相关的 rank/axis 规范化，例如 negative dim、`keepdim`、`expand(-1)`。
4. 对 axiom 需要显式传入 output shape 的算子计算 `s_out`，例如 `tensor.binary`、`tensor.unary`、`tensor.broadcast`、`tensor.map_reduce_aff`。
5. 对明显错误做前置校验，例如静态 broadcast 不兼容、静态 reshape 元素个数不一致。

### Tensor Plugin 应该做

1. 如果 axiom 类型本身已经包含 shape function，应优先让 plugin 计算输出类型。
2. `pad` 应使用 `pad_shape(lo, s_in, hi)` 推导输出 shape。
3. `concat` 应使用 `concat_shape(ax, Sis)` 推导输出 shape。
4. `dot_general` / 后续 matmul family 应使用 `dot_general_shape` 推导输出 shape。
5. `transpose` 应使用 `transpose_shape` 推导输出 shape。
6. `reshape` 的 product equality 属于 axiom 语义，frontend 只做静态明显错误检查，不复制完整 solver。

### 判断准则

1. 如果 tensor axiom 要求 frontend 传入 `s_out`，frontend 必须计算。
2. 如果 tensor axiom 的 return type 已经通过 shape function 计算输出 shape，frontend 不应重复实现。
3. 如果 shape 计算依赖 PyTorch 特有语义，例如 `-1` infer、negative dim、`keepdim`，frontend 必须先规范化。

## Shape 表示策略

### 核心原则

1. 同一个 PyTorch 符号维度必须映射到同一个 MimIR Nat def。
2. 静态维度使用 `world.lit_nat(n)`。
3. 简单符号维度使用函数参数 Nat，例如 `s0`、`s1`。
4. `top_nat()` 只作为无法解析的 fallback，不作为普通 dynamic shape 的默认表示。
5. 复杂表达式先分类处理：能解析的 affine/product 表达式保留；暂时不能解析的表达式降级并记录诊断。

### 需要支持的约束类型

优先级从高到低：

1. Equality：`s0 == s0`、broadcast 中非 1 维度相等、reshape 前后元素个数相等。
2. Product：`reshape((n, 2), (m,))` 需要表达或验证 `m == n * 2`。
3. Divisibility：`reshape((n, 6), (m, 3))` 隐含 `m * 3 == n * 6`。先不做求解，只做静态和简单表达式校验。
4. Range：`min/max` 作为 metadata，暂缓实现。

## 实施计划

### Phase 0: 建立 `shape_of` 基线

目标：把所有 operator shape 读取统一到一个入口。

文件：

- `src/mimir_frontend/operators.py`
- `src/mimir_frontend/translator.py`
- `tests/test_basic.py`
- `tests/test_utils.py`

任务：

1. 新增 `shape_of(x, node=None)`，先支持 `mim.Def`。
2. 把 `_rank_and_shape`、binary、where、expand、reduce、reshape、slice、split 迁移到 `shape_of`。
3. 为 `mim.Def` 输入增加测试：shape 必须来自 tensor type，而不是外部 annotation。
4. 保留 `_shape_dims` 作为内部实现或逐步改名，避免一次性大改。

验收标准：

1. `uv run --no-sync pytest tests/test_basic.py tests/test_utils.py -q` 通过。
2. operator 测试中不依赖 readable annotation 获取 shape。

### Phase 1: 输入 shape source 正规化

目标：placeholder 的初始类型来自用户签名、FakeTensor/meta 或真实 tensor shape；annotation 只作为测试 fixture fallback。

文件：

- `src/mimir_frontend/utils.py`
- `src/mimir_frontend/inductor_readable.py`
- `src/mimir_frontend/translator.py`
- `tests/test_utils.py`
- `tests/test_real_inductor_graphs.py`

任务：

1. `model_to_mimir(input_shapes=...)` 创建 tensor type 时，动态符号维度直接使用对应 Nat 参数，而不是先放 `top_nat()` 再靠 `input_to_syms` 修补。
2. 在 translator 初始化 placeholder 时，把 FX node 的 `meta["val"].shape` 作为可选 shape source。
3. `inductor_readable.py` 保留 annotation parser，但明确它只用于 readable fixture 生成初始 input type。
4. readable fixture 中的非十进制维度可以映射到稳定 Nat 参数，但该映射只发生在 placeholder type construction 阶段。
5. 后续所有 operator shape 读取必须通过 `shape_of(mim_def)`。

验收标准：

1. 同一输入符号在初始 tensor type 中就是同一个 MimIR Nat def。
2. 删除或禁用 `input_to_syms` 后，核心 operator 仍能从 `mim.Def.type()` 得到正确 shape。
3. `mlp_1`、`lstm_1` 的 high-level translation 仍然通过。

### Phase 2: PyTorch SymInt 到 MimIR Nat 的映射

目标：支持真实 FakeTensor/meta 中的 dynamic shape。

文件：

- 新增：`src/mimir_frontend/shape.py`
- 修改：`src/mimir_frontend/utils.py`
- 修改：`src/mimir_frontend/translator.py`
- 测试：`tests/test_utils.py`

建议接口：

```python
class ShapeEnv:
    def to_nat(self, dim) -> mim.Def:
        ...

    def tensor_type_from_shape(self, shape, elem_type) -> mim.Def:
        ...
```

任务：

1. `int` 维度映射为 `world.lit_nat(n)`。
2. `torch.SymInt` 或 Sympy symbol 映射为稳定 Nat 参数。
3. 同一 SymInt/Sympy expression 重复出现时复用同一个 Nat def。
4. 简单表达式保留原始 debug name，复杂表达式暂时降级为 `top_nat()` 并记录 diagnostic。

验收标准：

1. 两个输入共享同一 `SymInt` batch 时，MimIR 类型中共享同一 Nat def。
2. 不再依赖 string annotation 来证明符号相等。

### Phase 3: 统一 operator shape function

目标：把 frontend 必须负责的 shape 计算抽出来，降低重复逻辑。

文件：

- `src/mimir_frontend/shape.py`
- `src/mimir_frontend/operators.py`
- `tests/test_basic.py`

建议接口：

```python
def broadcast_shapes(lhs: tuple[mim.Def, ...], rhs: tuple[mim.Def, ...]) -> tuple[mim.Def, ...]:
    ...

def reduce_shape(input_shape: tuple[mim.Def, ...], dim, keepdim: bool) -> tuple[mim.Def, ...]:
    ...

def expand_shape(input_shape: tuple[mim.Def, ...], target_dims) -> tuple[mim.Def, ...]:
    ...
```

任务：

1. 抽出 broadcast shape 推导，替换 `_broadcast_shape_dims`。
2. 抽出 reduce shape 推导，覆盖全局 reduce、单 dim reduce、多 dim reduce、`keepdim=True/False`。
3. 抽出 expand shape 推导，保留 `-1` 语义和 rank 提升语义。
4. 抽出 split/slice shape 推导，静态 split size 继续生成 tuple of slices。
5. 错误信息包含 operator 名、输入 shape、目标 shape。

验收标准：

1. 现有 operator 测试全部通过。
2. broadcast、reduce、expand、split 的 shape 错误由统一 helper 抛出。
3. 复杂 operator 方法只负责构造 MimIR axiom，不再内联大量 shape 判断。

### Phase 4: Plugin shape axiom 优先

目标：避免 frontend 重复实现 tensor plugin 已经提供的 shape 逻辑。

文件：

- `src/mimir_frontend/operators.py`
- `tests/test_basic.py`

任务：

1. 检查 `pad`、`concat`、`dot_general`、`transpose` 当前实现，优先调用已有 axiom 并从返回类型读取输出 shape。
2. 对仍需 frontend 传入 `s_out` 的 axiom 保持 frontend shape function。
3. 增加 IR 检查，确认这些算子使用 `%tensor.pad` / `%tensor.concat` / `%tensor.transpose` 等 high-level axiom，而不是手写低层 map。

验收标准：

1. frontend 不复制 `pad_shape`、`concat_shape`、`dot_general_shape`、`transpose_shape` 的逻辑。
2. high-level IR 中保留对应 tensor axiom。

### Phase 5: Reshape 兼容性校验

目标：frontend 能发现明显错误 reshape，但不复制完整 solver。

文件：

- `src/mimir_frontend/shape.py`
- `src/mimir_frontend/operators.py`
- `tests/test_basic.py`

任务：

1. 支持静态 reshape 的元素个数校验。
2. 支持 `-1` infer dimension 的有限场景：只有一个 `-1`，且其他维度静态或同符号可比。
3. 对无法证明的 dynamic reshape 保守放行，让 `%tensor.reshape` axiom 表达 product equality。
4. 记录 diagnostic，说明该 reshape 依赖 tensor axiom 检查。

验收标准：

1. 静态不兼容 reshape 在 Python 层报 `ValueError`。
2. 静态兼容 reshape 和 dynamic reshape 继续生成 `%tensor.reshape`。
3. 不引入复杂 solver 依赖。

### Phase 6: Shape 诊断和错误注入测试

目标：让失败原因可定位，支持“故意破坏 FX graph shape 后应失败”的测试。

文件：

- `src/mimir_frontend/shape.py`
- `src/mimir_frontend/translator.py`
- `tests/test_basic.py`
- `tests/test_real_inductor_graphs.py`

任务：

1. 增加 shape diagnostic 数据结构，记录 `source_node`、`operator`、`input_shapes`、`output_shape`。
2. 在 operator wrapper 抛错时附带 FX node target。
3. 增加错误注入测试：构造不兼容 broadcast 或静态 reshape。
4. 对 unsupported operator 继续使用 `NotImplementedError`，对 shape 不合法使用 `ValueError`。

验收标准：

1. 错误消息能定位到 operator 和 shape。
2. 错误注入测试覆盖 static 和 dynamic 两类输入。
3. 不依赖新 MimIR axiom。

### Phase 7: 真实模型能力边界固化

目标：在等待新 axiom 期间，持续知道 frontend 的上限在哪里。

文件：

- `tests/test_real_inductor_graphs.py`
- `scripts/dump_inductor_mimir.py`
- `operator_gemini_impl_summary.md`

任务：

1. 保留 success 列表：`mlp_1`、`lstm_1`。
2. 保留 frontier 列表：`gcn_1 -> aten.index.Tensor`、`moe_1 -> torch.max(dim)`、`faster_rcnn_1 -> aten.convolution`。
3. 对每个 frontier case 增加 partial translation dump，确保失败点之前的 IR 可检查。
4. 每次 dynamic shape 改动后更新 summary 文档，避免能力边界漂移。

验收标准：

1. 成功模型仍成功。
2. 失败模型失败在预期 unsupported operator，而不是 shape 丢失或 native crash。
3. dump 脚本能输出 high-level tensor IR，用于回归分析。

## 推荐下一步

优先执行 Phase 0 和 Phase 1。

原因：

1. `shape_of` 是后续所有 dynamic shape 工作的统一接口。
2. 先把 shape source 改对，后面再做 shape 推导抽象才不会建立在 annotation side-channel 上。
3. Phase 0/1 不依赖新 tensor axiom，风险低，收益直接体现在真实 Inductor 图上。

建议第一批提交拆分：

1. `refactor: add shape_of entry for mimir tensor defs`
2. `test: ensure operator shapes come from mimir def types`
3. `feat: build symbolic placeholder types without top_nat fallback`
4. `test: cover fake tensor meta as shape source`
5. `docs: document frontend and tensor plugin shape responsibilities`

## 长期方向

MimIR 侧如果后续提供更完整的 Nat expression / metadata 支持，frontend 可以逐步把以下信息从诊断升级为真实类型或 metadata：

1. `m == n * 2` 这类 product equality。
2. `m % 3 == 0` 这类 divisibility。
3. `min/max` range metadata。
4. 从 PyTorch ShapeEnv 导入的 guard 条件。

在此之前，frontend 的目标应保持务实：通过 `shape_of` 保留符号 identity、构造正确 high-level tensor IR、优先复用 tensor plugin shape axiom，并在明显 shape 错误时提前失败。
