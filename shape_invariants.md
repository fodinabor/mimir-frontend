# Shape Invariants 设计草案

本文档定义 MimIR frontend / tensor plugin 当前应遵守的 shape information 设计原则，目标是回答三个问题：

1. 每个 primitive tensor axiom 的 `input_shapes -> output_shape` 关系应如何定义？
2. 每个 derived frontend operator 的 shape invariant 应如何定义？
3. frontend 和 MimIR tensor plugin 各自应该承担哪些 shape 语义职责？

结论先说：

- `reduce`、`squeeze`、`split` 这类高层 operator 不一定要做成 axiom。
- 但不管是否是 axiom，它们都必须有统一的 canonical shape function。
- primitive tensor axiom 层最好提供统一的 shape helper。
- frontend 负责把 PyTorch 语义翻译成 canonical shape 规则，而不是各处零散硬编码。

## 1. 术语

### 1.1 Primitive Tensor Axiom

指 tensor plugin 中已经存在的底层原语，例如：

- `%tensor.pad`
- `%tensor.concat`
- `%tensor.transpose`
- `%tensor.reshape`
- `%tensor.slice`
- `%tensor.broadcast`
- `%tensor.broadcast_in_dim`
- `%tensor.map_reduce_aff`
- `%tensor.dot_product`

### 1.2 Derived Frontend Operator

指 frontend 通过若干 primitive axiom 组合出来的高层 operator，例如：

- `sum`
- `mean`
- `amax`
- `var_mean`
- `squeeze`
- `unsqueeze`
- `split`
- `select`
- `where`
- `expand`

### 1.3 Shape Information

这里的 shape information 不是单纯“rank 对不对”，而是包括：

- 哪些维度是直接复用输入的同一个 `Def`
- 哪些维度是新构造的 literal `Nat`
- 哪些维度是通过 `Nat` expression 计算出来的
- 哪些维度因为当前能力不足退化成 `top_nat()`

### 1.4 Shape Invariant

shape invariant 指：

> 对任意 operator，只要输入 shape 和 attributes 固定，输出 shape 的语义必须有唯一、明确、可测试的定义。

这里的“可测试”包括：

- `shape_of(result)` 能恢复 canonical shape
- 或 primitive app 的 `type()` 本身能恢复 canonical shape
- 或至少 frontend 的临时 shape cache 与 canonical shape 一致

## 2. 总体设计原则

### 2.1 不是每个 operator 都必须是 axiom

高层 operator 完全可以由已有 axiom 组合出来，例如：

- `sum` 通过 `%tensor.map_reduce_aff`
- `squeeze` / `unsqueeze` 通过 `%tensor.reshape`
- `split` 通过 `%tensor.slice`
- `select` 通过 `slice + squeeze`

这没有问题。

问题不在于“是否是 axiom”，而在于：

> 对每个组合步骤，都必须有明确的 shape semantics。

### 2.2 每个 primitive 都需要明确的 shape contract

对每个 primitive tensor axiom，都必须能回答：

- 输出 shape 是直接由返回 type 推导出来的吗？
- 还是由显式输入的 `s_out` 指定？
- 如果是显式 `s_out`，frontend 或 plugin 是否已经足以唯一确定该输出 shape？

### 2.3 Shape helper 不一定是 axiom

shape helper 的目标是定义语义，不一定要定义可执行原语。

例如适合新增到 `tensor.mim` 的 helper：

- `broadcast_shape`
- `broadcast_in_dim_shape`
- `reduce_shape`
- `squeeze_shape`
- `unsqueeze_shape`
- `split_shapes`
- `reshape_shape` 或 frontend 侧 canonical reshape rules

这些 helper 可以只是 `lam`，不一定要变成用户直接调用的 axiom。

### 2.4 Frontend 不应长期承担 shape 真相来源

frontend 当前可以维护临时 `_shape_cache`，但长期不应作为 shape truth source。

长期目标应是：

1. primitive axiom 的 type / helper 足够表达 shape 语义
2. `shape_of(def)` 主要从 `def.type()` 或 canonical helper 恢复 shape
3. frontend cache 只作为过渡机制，而不是主语义层

## 3. Primitive Tensor Axiom 的 Shape Invariant

本节定义：

> 对每个 primitive tensor axiom，必须有什么级别的 `input_shapes -> output_shape` 对应关系。

### 3.1 `%tensor.pad`

当前 plugin 已有：

- `pad_shape(lo, s_in, hi)`

Invariant：

- `shape_of(out) == pad_shape(lo, s_in, hi)`
- rank 不变
- 第 `d` 维满足：`out_d = lo_d + in_d + hi_d`

状态：

- 设计正确
- frontend 后续应直接复用 plugin shape 语义，不要自己再手写一份

### 3.2 `%tensor.concat`

当前 plugin 已有：

- `concat_shape(ax, Sis)`

Invariant：

- `shape_of(out) == concat_shape(ax, Sis)`
- 非 `ax` 维必须与所有输入共享
- `ax` 维等于各输入在该维的和

状态：

- 设计正确
- 当前 frontend `cat` 最终 shape 仍主要依赖 plugin type，不应再用 ad-hoc cache 覆盖其真实语义

### 3.3 `%tensor.transpose`

当前 plugin 已有：

- `transpose_shape(s, permutation)`

Invariant：

- `shape_of(out) == transpose_shape(s, permutation)`
- 只是维度重排，不引入新 extent
- 被重排的维 identity 应保住

状态：

- 设计正确
- frontend 的 `shape_of` cache 可作为过渡层，但最终应尽量依赖 plugin return type

### 3.4 `%tensor.dot_product`

当前 plugin 已有：

- `dot_general_shape(...)`

Invariant：

- `shape_of(out) == dot_general_shape(...)`
- batch dim 保持
- contracting dim 被消去
- remaining dims 按规则拼接

状态：

- 设计正确
- 后续 `bmm` / generalized matmul 应直接依赖此规则

### 3.5 `%tensor.reshape`

当前 plugin 只有：

- 输出 shape 显式来自 `s_out`

Invariant：

- `shape_of(out) == s_out`
- 必须满足 `prod(s_in) == prod(s_out)`

建议：

- 可定义 `reshape_shape(s_in, target_spec)`，但 PyTorch `-1` 推断更适合 frontend 先 canonicalize
- 如果 frontend 已经给出唯一确定的 `s_out`，则 primitive 层不强制要求额外 `check`

### 3.6 `%tensor.slice`

当前 plugin：

- 输出 shape 显式来自 `s_out`

Invariant：

- `shape_of(out) == s_out`
- 对每一维都满足合法范围和 step 语义
- 若是纯窄化，未切维度的 identity 应保持

建议：

- frontend 仍可先算 `s_out`
- 如果 frontend 已经唯一确定 `s_out`，则 primitive 层不强制要求额外 `check`

### 3.7 `%tensor.broadcast`

当前 plugin：

- 输出 shape 显式来自 `s_out`

缺口：

- 缺少 canonical `broadcast_shape`

Invariant：

- `shape_of(out) == s_out`
- 每一维都必须满足 broadcast compatibility：
  - `in_d == out_d`
  - 或 `in_d == 1`

建议：

- 新增 `broadcast_shape`
- 如果 frontend 已经按 canonical broadcast 规则给出唯一 `s_out`，则 primitive 层不强制要求额外 `check`

### 3.8 `%tensor.broadcast_in_dim`

当前 plugin：

- 输出 shape 显式来自 `s_out`

缺口：

- 缺少 `broadcast_in_dim_shape`

Invariant：

- `shape_of(out) == s_out`
- 输入维通过 `dims` 映射到输出维
- 未映射维必须是 broadcast hole
- 映射维必须满足 equal-or-one 规则

建议：

- 新增 `broadcast_in_dim_shape`
- 如果 frontend 已经按 canonical 规则给出唯一 `s_out` 和 `dims`，则 primitive 层不强制要求额外 `check`

### 3.9 `%tensor.map_reduce_aff`

当前 plugin：

- 输出 shape 显式来自 `So`

问题：

- 它是足够通用的 primitive，但高层 reduce 语义不能只停留在“传个 `So`”

Invariant：

- `shape_of(out) == So`
- `So` 必须对应某个 canonical reduce 规则或更一般的 affine map 规则
- `map_out` 和 `maps` 必须与 `So/Sr` 一致

建议：

- 不一定给 `%tensor.map_reduce_aff` 本身加专门 helper
- 但必须给 derived operator 提供 `reduce_shape`
- 更长期如果需要更强约束，可再讨论 `map_reduce_aff` 级别的合法性约束，但不是当前前提

## 4. Derived Frontend Operator 的 Shape Invariant

本节定义：

> 即使某个高层 operator 不是 axiom，它也必须有统一的 canonical shape function。

### 4.1 Unary / Shape-preserving

适用：

- `relu`
- `exp`
- `neg`
- `logical_not`
- `clone`
- `copy`

Invariant：

- `shape_of(out) == shape_of(in)`
- 每一维 identity 尽量保持

### 4.2 Binary / Ternary Broadcast-producing

适用：

- `add/sub/mul/div`
- `maximum/minimum`
- `where`

Invariant：

- `shape_of(out) == broadcast_shape(lhs_shape, rhs_shape)` 或多输入 broadcast 扩展
- 对于 `where`：
  - `shape_of(out) == broadcast_shape(broadcast_shape(cond_shape, x_shape), y_shape)`

注意：

- 当前 frontend 在动态不可判定情形下保守偏向 lhs dim，这只是临时策略
- 长期应以 canonical `broadcast_shape` 为准

### 4.3 Reduce-producing

适用：

- `sum`
- `mean`
- `amax`
- `var_mean`

Invariant：

- `shape_of(out) == reduce_shape(in_shape, dim, keepdim)`

其中：

- `keepdim=False`：删除被 reduce 的维
- `keepdim=True`：对应位置保留 rank，并插入 literal `1`

对多返回值：

- `var_mean` 的两个返回 tensor 都必须满足同一个 `reduce_shape`

### 4.4 Reindexing / Layout-changing

适用：

- `transpose`
- `reshape`
- `squeeze`
- `unsqueeze`

Invariant：

- `transpose`：
  - `shape_of(out) == transpose_shape(in_shape, permutation)`
- `reshape`：
  - `shape_of(out) == target_shape`
  - frontend 必须保证 target shape 语义明确
- `squeeze`：
  - `shape_of(out) == squeeze_shape(in_shape, dim)`
- `unsqueeze`：
  - `shape_of(out) == unsqueeze_shape(in_shape, dim)`

### 4.5 Region / Indexing-changing

适用：

- `slice`
- `select`
- `split`

Invariant：

- `slice`：
  - `shape_of(out) == slice_shape(in_shape, dim, start, end, step)`
- `select`：
  - `shape_of(out) == select_shape(in_shape, dim)`
- `split`：
  - 每个输出都满足 `split_shapes(in_shape, split_spec, dim)[i]`

关键原则：

- 未被修改的维度 identity 必须尽量保住
- 被修改的维度必须有明确 computed expression
- 只有真正无法表达时，才能退化成 `top_nat()`

## 5. Frontend 和 Tensor Plugin 的职责边界

### 5.1 Frontend 负责什么

frontend 负责：

- 从 `mim.Def.type()` / FakeTensor / `node.meta["val"]` / explicit input shapes 提取 shape
- 把 PyTorch 语义规范化为 canonical operator 参数
  - `dim=-1`
  - `dims` tuple/list
  - `keepdim`
  - `expand(-1, ...)`
  - `split_size` vs `sections`
- 将简单 `SymInt` / Sympy 表达式翻译为 MimIR `Nat` expression
- 对明显非法的静态 case 早期报错
- 在 primitive 尚不能从 `def.type()` 恢复精确 shape 时，用临时 cache 保住 canonical shape

frontend 不应负责：

- 长期维护一套独立于 plugin 的 shape 真相
- 为每个 op 重复发明 shape 语义

### 5.2 Tensor Plugin 负责什么

tensor plugin 负责：

- 为 primitive tensor axiom 提供统一的 shape helper
- 把 primitive 的 shape semantics 固化在 `tensor.mim`
- 让不同 frontend 共用同一套 canonical shape algebra
- 尽量让 primitive app 的返回 type 可以直接恢复精确 shape

tensor plugin 不一定要负责：

- PyTorch 专属参数规范化
- `-1` 这类前端语法糖的直接解析

## 6. 推荐新增的 Shape Helper

建议优先在 `tensor.mim` 中新增以下 helper：

### P0

- `broadcast_shape`
- `broadcast_in_dim_shape`
- `reduce_shape`
- `squeeze_shape`
- `unsqueeze_shape`

### P1

- `reshape_shape` 或 frontend 侧 canonical reshape rules
- `split_shapes`
- `select_shape`

### P2

- `gather_shape`
- `scatter_shape`
- 更一般的 `map_reduce_aff_check`

## 7. 关于更一般的 symbolic affine shape expression

这个问题必须分成两层：

### 7.1 Frontend 层

frontend 应该把来自 PyTorch 的 shape 表达式翻译到 MimIR `Nat` expression。

例如：

- `n + 1`
- `n - 1`
- `n * 4`
- `(n + 3) // 4`

这里的职责是“翻译来源表达式”，不是“定义 operator 语义”。

### 7.2 Plugin 层

plugin 应该定义 primitive / canonical helper 的 shape algebra。

例如：

- `pad_shape(lo, s_in, hi)`
- `broadcast_shape(s1, s2)`
- `reduce_shape(s_in, dims, keepdim)`

这里的职责是“定义 operator 的 shape 真相”。

### 7.3 `%affine.index` 的作用边界

`%affine.index` 很适合表达：

- 读索引映射
- 写索引映射
- linearize / delinearize
- affine access transformation

但它不应该直接替代全部 `Nat` shape helper。

原因：

- shape helper 主要工作在 `Nat` level
- affine map 主要工作在 index transformation level

两者相关，但不是一层抽象。

更准确的关系是：

- shape helper 负责定义 tensor extent
- affine helper 负责定义 extent 内部的 index movement

## 8. 当前实现的主要不足

### 8.1 Frontend 不足

- shape rules 仍然散落在 operator 实现内部
- `_shape_cache` 仍然是过渡方案
- 对 `SymInt` expression 的支持仍然不足
- `reshape` 的 `-1` / product equality 还未系统化

### 8.2 Plugin / Core 不足

- 很多 primitive 缺少 canonical shape helper
- Python 侧 `def.type()` 不能总是恢复精确 shape

## 9. 推荐实施顺序

### Phase 0

- 明确所有现有 operator 的 canonical shape invariant
- 用测试固化 `shape_of(result)` 的行为

### Phase 1

- 在 `tensor.mim` 增加：
  - `broadcast_shape`
  - `broadcast_in_dim_shape`
  - `reduce_shape`
  - `squeeze_shape`
  - `unsqueeze_shape`

### Phase 2

- frontend 抽离统一 shape rules 模块
- operator 实现统一走 shape rules，再发 primitive axiom

### Phase 3

- 将 `_shape_cache` 收缩为过渡机制
- 尽量让 shape 真相回到 primitive return type / plugin helper

### Phase 4

- 扩展到更一般的 symbolic affine shape expression
- 接入 `gather/scatter` shape helper

## 10. 一句话结论

当前最合理的方向不是“把所有高层 operator 都做成 axiom”，而是：

> 为 primitive tensor axiom 建立统一的 shape helper 体系；再让 frontend 的 derived operator 组合严格遵守这些 canonical shape invariants。
