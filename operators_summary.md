| Pattern Kind | Value | Description | Operators |
| --- | --- | --- | --- |
| kElemWise | 0 | Elementwise: one-to-one input/output mapping | `aten.add`, `aten.bitwise_and`, `aten.clamp_max`, `aten.clamp_min`, `aten.div`, `aten.eq`, `aten.exp`, `aten.ge`, `aten.gt`, `aten.le`, `aten.logical_not`, `aten.mul`, `aten.neg`, `aten.reciprocal`, `aten.relu`, `aten.rsqrt`, `aten.sigmoid`, `aten.sqrt`, `aten.sub`, `aten.tanh`, `aten.where`, `prims.convert_element_type`, `prims.fma` |
| kBroadcast | 1 | Broadcasting: axes mapping in order with broadcast | `aten.expand`, `aten.full` |
| kInjective | 2 | Injective: each output element depends on a single input element | `aten.cat`, `aten.clone`, `aten.copy`, `aten.lift_fresh_copy`, `aten.permute`, `aten.select`, `aten.slice`, `aten.split`, `aten.squeeze`, `aten.unsqueeze`, `aten.view` |
| kCommReduce | 3 | Communicative reduction: output elements aggregate over input elements | `aten.amax`, `aten.any`, `aten.max`, `aten.mean`, `aten.sum`, `aten.var_mean` |
| kOutEWiseFusable | 4 | Complex operation whose output can accept elementwise followers | `aten.addmm`, `aten.bmm`, `aten.convolution`, `aten.mm` |
| kTuple | 7 | Tuple node | `aten.topk` |
| kOpaque | 8 | Opaque: cannot be fused | `aten._unsafe_index`, `aten.convolution_backward`, `aten.copy_`, `aten.gather`, `aten.index`, `aten.index_put`, `aten.scatter`, `aten.scatter_add`, `aten.select_scatter`, `aten.slice_scatter`, `aten.split_with_sizes`, `aten.sym_numel`, `aten.sym_size`, `prims._low_memory_max_pool_offsets_to_indices`, `prims._low_memory_max_pool_with_offsets`, `prims.inductor_lookup_seed`, `prims.inductor_random`, `prims.inductor_seeds`, `prims.iota` |
