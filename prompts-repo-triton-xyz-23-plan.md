# triton-xyz: AnalysisAddress 渐进增强实施方案（不大改 ptrAnalysis）

## 1. 背景与目标

当前 `ptrAnalysis`（`PtrExprAnalysis` / `PtrAnalysis`）已经承担了较多历史兼容逻辑。为了降低回归风险，本计划将其固定为“粗粒度 seed 解析器”，逐步把 mixed structured/unstructured 的分析能力集中到 `AnalysisAddress`，并形成稳定的地址语义契约供 TTA 路线消费。

目标：

1. 不做 `ptrAnalysis` 大改动。
2. 逐步提升 `AnalysisAddress` 对不同 dim 混合 structured/unstructured 的表达能力。
3. 将地址推断、规范化、lowering 的能力边界清晰化，减少 scattered 限制与重复判断。
4. 保持每阶段可回归、可回滚、可 A/B。

---

## 2. 约束与非目标

### 2.1 约束（必须满足）

1. 不改变 `PtrState` 现有字段语义和主要推导规则。
2. 不重写 `PtrExprAnalysis::visitOperand*` 主逻辑。
3. 允许仅做小型辅助增强（例如更清晰的失败原因、轻量接口），不改变核心行为。
4. 任何语义扩展优先在 `AnalysisAddress` + TTA downstream 完成。

### 2.2 非目标（本计划不做）

1. 不在第一阶段支持所有复杂 control-flow 递推。
2. 不在第一阶段打通 block + indirect + atomic 全覆盖。
3. 不在单个 PR 内完成端到端全改，必须分阶段落地。

---

## 3. 当前问题摘要（作为改造驱动）

1. 单 indirect dim 假设广泛存在：分析、发射、lowering 多处硬编码单维。
2. `AnalysisAddress` 对 non-gather dim 的限制偏强，导致可表达场景 fallback。
3. loop-carried indirect recurrence 仍是硬失败路径。
4. 多个 pass 对失败条件/错误原因重复判断，维护成本高。

---

## 4. 目标架构

将 `AnalysisAddress` 内部拆分为三层：

1. Seed Layer（输入解析层）
2. Refine Layer（语义补全层）
3. Contract Layer（输出契约层）

### 4.1 Seed Layer

职责：

1. 接收 `tta.make_addr`/`tta.reindex`/`tta.indirect_reindex`/`tta.advance` 链。
2. 对 Triton pointer-like 值调用现有 `ptrAnalysis` 获取 `PtrState` seed。
3. 保留当前行为作为 baseline，不改变 `ptrAnalysis` 本体逻辑。

### 4.2 Refine Layer

职责：

1. 在 `PtrState` 过于保守或无法完全表达时，基于地址表达式做补全。
2. 逐步支持 mixed 维度语义（先单 indirect，再多 indirect）。
3. 将 wrap/indirect/structured 信息统一建模到 `AddressDescriptor`。

### 4.3 Contract Layer

职责：

1. 输出可 lowering 的 `AddressDescriptor`（唯一中间契约）。
2. 保证失败原因可诊断、可测试、可归因。
3. 为 emitter/lowering 提供稳定输入，不让后端反向“猜表达式”。

---

## 5. 分阶段执行计划（PR-by-PR）

## Phase 0: 结构化重构（无行为变化）

### PR0-1: `AnalysisAddress` 内部分层与接口整理

TODO:

1. 引入 `AddressAnalysisOptions` 与 `AddressAnalysisResult`（保留旧 API 兼容）。
2. 拆分 `analyzeDescriptor` 内部流程为：
   - `analyzeFromTTAChain`
   - `analyzeFromPtrStateSeed`
   - `refineDescriptor`
   - `validateDescriptor`
3. 统一 `failureReason` 写入规则，消除散乱字符串路径。
4. 增加 descriptor debug dump（按维输出 size/stride/offset/wrap/indirect）。

验收标准：

1. 语义不变（IR 结果不变或仅 canonicalization 等价变化）。
2. 现有 `lit` 全绿。

---

## Phase 1: 单 indirect mixed 能力增强（保持单维模型）

### PR1-1: AnalysisAddress 补全器（Refiner v1）

TODO:

1. 新增“表达式补全器”，当 `ptrAnalysis` seed 不足时做局部补全。
2. v1 支持 op：
   - `arith.addi`
   - `arith.muli`
   - `arith.extsi`
   - `tt.make_range`
   - `tt.expand_dims`
   - `tt.broadcast`
   - `tt.splat`
3. 严格限制只做维度语义补全，不回写 `ptrAnalysis` 全局状态。
4. 对不支持图形保持清晰失败原因。

### PR1-2: 放宽非 gather dim 限制（保守放宽）

TODO:

1. 在 `toAddressDescriptorFromSingleIndirectState` 路径放宽“非 gather dim 必须 singleton/broadcast”限制。
2. 允许更一般 structured 步长/offset 模式进入 descriptor。
3. 加 feature gate（临时选项）用于 A/B。

验收标准：

1. 原先 fallback 的部分 2D/3D case 可进入 TTA 地址路径。
2. 不引入错误 lowering；对不支持模式仍显式 fallback/error。

---

## Phase 2: 多 indirect dim 语义建模

### PR2-1: Descriptor 多维 indirect 表达

TODO:

1. 将 `AddressDescriptor` 语义从“最多一个 indirect dim”扩展为“可多维 indirect 共存”。
2. 保持同维链式 indirect 的 merge 逻辑（index add / mask and）。
3. 明确跨维 indirect 的组合顺序语义（定义 deterministic order）。

### PR2-2: TTAEmitter 多维发射

TODO:

1. 修改 `TTAEmitter::emitAddress`：
   - 现状：只允许单 indirect dim。
   - 目标：按维顺序串联发射多个 `tta.indirect_reindex`。
2. 保持 `emitMakeAddr` 仅处理无 indirect 的 base 地址。
3. 对多维冲突场景给出可测试失败信息。

验收标准：

1. `tta-to-memref-diagnostics` 中 mixed-indirect 相关 case 从预期失败迁移为正向测试（除真正非法 case）。
2. 新增 canonicalize 用例覆盖“同维 merge + 跨维保留顺序”。

---

## Phase 3: TTAToMemref 从单 gather 特化到通用 lowering

### PR3-1: Indirect 信息结构升级

TODO:

1. 将 `collectIndirectInfo` 从单结构升级为多维集合结构（如 `SmallVector<IndirectDimInfo>`）。
2. 更新 path 分类逻辑（Direct/Indirect/WrapAware）支持多维 indirect。

### PR3-2: 通用 indirect lowering

TODO:

1. 将 load/store indirect lowering 从“单 `gatherDim` 特化”改为“按维通用 loop-nest 生成”。
2. 保留单维路径作为优化 fastpath。
3. wrap + multi-indirect 统一走同一线性化流程，避免 duplicated code paths。

验收标准：

1. `tta.load/tta.store` 在多 indirect 维组合下可稳定 lowered。
2. wrap + indirect 混合 case 通过。
3. 旧单维性能路径不退化（至少结构不明显变差）。

---

## Phase 4: loop-carried 递推支持（渐进）

### PR4-1: structured recurrence 基线增强

TODO:

1. 保持现有常量 lb/step 支持。
2. 先完成 address seed + step expression 的稳定抽取。

### PR4-2: indirect recurrence v1（同维）

TODO:

1. 支持同维 indirect 的 loop-carried 递推。
2. 限制为常量步长与可解析表达式。

### PR4-3: indirect recurrence v2（跨维）

TODO:

1. 支持跨维 indirect loop-carried 递推（在 v1 稳定后启用）。
2. 若表达式不可解析，继续给出明确诊断，不 silent fallback。

验收标准：

1. 移除/缩小 `unsupported loop-carried indirect recurrence` 失败面。
2. 保留动态 lb/step 等不可证明场景的失败诊断。

---

## 6. 文件级 TODO 清单

## 6.1 Analysis 层

1. `include/triton-shared/Analysis/AnalysisAddress.h`
2. `lib/Analysis/AnalysisAddress.cpp`

TODO:

1. 增加 options/result 类型与分层函数声明。
2. 引入 refiner 上下文与诊断上下文。
3. 实现 descriptor 级别 validate/normalize 管线。

## 6.2 TTA 发射层

1. `lib/Analysis/AnalysisAddress.cpp` (`TTAEmitter::*`)

TODO:

1. `emitAddress` 支持多 indirect dim 链式发射。
2. 保持 `emitMakeAddr` 的纯 base-address 职责。

## 6.3 TTAToMemref

1. `lib/Conversion/TritonToLinalgTTA/TTAToMemrefPass.cpp`

TODO:

1. 多维 indirect 数据结构升级。
2. lowering path 分类器与通用 loop-nest 逻辑改造。
3. wrap + indirect 统一线性化函数。
4. loop-carried recurrence 增量支持。

## 6.4 TTA Structured/Unstructured 前端 pass

1. `lib/Conversion/TritonToLinalgTTA/TritonToTTAStructuredPass.cpp`
2. `lib/Conversion/TritonToLinalgTTA/TritonToTTAUnstructuredPass.cpp`

TODO:

1. 保持接口兼容，逐步切换到增强后的 `AnalysisAddress` 能力。
2. 保留 fallback attr/reason，避免大面积行为突变。

---

## 7. 测试计划（必须同步）

## 7.1 新增/改造 Conversion 测试

1. `test/Conversion/triton-to-tta-structured.mlir`
2. `test/Conversion/triton-to-tta-unstructured.mlir`
3. `test/Conversion/tta-to-memref.mlir`
4. `test/Conversion/tta-to-memref-diagnostics.mlir`
5. `test/Conversion/triton-to-linalg-tta-lowering.mlir`

TODO:

1. 将“可支持场景”从 fallback/diagnostic 转为正向 lowering case。
2. 保留真正非法场景为 diagnostics。
3. 覆盖：
   - 单 indirect + 多 structured 维
   - 多 indirect 维组合
   - wrap + indirect 组合
   - loop-carried indirect recurrence（分层覆盖）

## 7.2 Dialect 层测试

1. `test/Dialect/triton-address-canonicalize.mlir`
2. `test/Dialect/triton-address-dialect*.mlir`

TODO:

1. 同维 indirect merge 规则回归。
2. 跨维 indirect 组合顺序与 canonicalize 一致性。

---

## 8. 回归与验收标准（DoD）

每个阶段必须满足：

1. 变更范围内 `lit` 通过。
2. 全量 `lit -v test/Conversion` 无非预期回归。
3. 新增能力有正向测试。
4. 保留失败路径有 diagnostics 测试。
5. fallback reason 可解释、可定位。

---

## 9. 开发顺序与提交策略

1. 每个 PR 限定一个主题，避免功能+重构混合过大。
2. 先重构不改行为，再引入语义变更。
3. 每个 PR 附最小测试集命令与结果摘要。
4. 每个 PR 预留回滚点（feature gate 或 guarded branch）。

建议顺序：

1. PR0-1
2. PR1-1
3. PR1-2
4. PR2-1
5. PR2-2
6. PR3-1
7. PR3-2
8. PR4-1
9. PR4-2
10. PR4-3

---

## 10. 风险与缓解

风险：

1. 多 indirect 维组合导致 lowering 复杂度激增。
2. wrap + indirect 组合路径可能引入边界错误。
3. loop recurrence 容易出现“看似可解析、实则错误归纳”。

缓解：

1. 保留 fastpath 与通用路径并行，先 correctness 后优化。
2. 加强 diagnostics，宁可失败也不 silent miscompile。
3. 每阶段引入最小可证明规则，逐步放开。

---

## 11. 当前阶段任务 (Next Actions)

1. PR4-2a 收尾, 已完成: 放宽 no-seed recurrence 的 direct-dim 前提, 支持非 `offset=0, stride=1` 形态.
2. PR4-2b 收尾, 已完成: 补齐 no-seed + dynamic mask 的 recurrence 路径.
3. PR4-2c 收尾, 已完成: 支持 no-seed + dynamic 1D step index.
4. 测试补齐, 已完成: 新增 "dynamic index/mask recurrence 正向 + 非法 recurrence 诊断" 成对用例.
5. PR4-3a 已完成: `TTAToMemref` loop-carried recurrence 支持跨维 indirect 组合, 覆盖 mixed seed/no-seed 维度.
6. 下一步: 推进 PR4-3b, 补齐跨维 recurrence 的边界诊断与更多组合回归.

---

## 12. 进度更新 (2026-02-26)

审查结论:

1. Phase 0 - Phase 3 已完成, 且 `lit -v test/Conversion` 全绿.
2. Phase 4 处于进行中, PR4-1 已完成, PR4-2 已完成, PR4-3a 已完成, PR4-3b 未开始.

已完成:

1. `AnalysisAddress` 已完成分层入口与选项化接口 (`AddressAnalysisOptions` / `AddressAnalysisResult`)
2. 已落地 descriptor dump 与 failure reason 统一写入路径
3. 已新增 `test/Conversion/analysis-address-snapshot.mlir` 作为行为快照基线
4. 已支持多 indirect dim descriptor 建模, `TTAEmitter` 多维链式发射, 以及 `TTAToMemref` 的多维 indirect + wrap 通用 lowering
5. 已新增/更新 Conversion 回归用例, 覆盖 multi-indirect 和 wrap + indirect 组合

本次新增:

1. 放开 `TTAToMemref` 对 loop-carried 地址 seed 含 indirect 的硬限制, 允许 `iter_args` 初值已含 indirect 时进行地址描述符收集
2. 扩展 loop-carried step 表达式解析范围, 在 `advance` / `reindex` 基础上支持 `tta.indirect_reindex`
3. 新增正向测试 `loop_carried_addr_supported_indirect_seed`, 验证 loop-carried indirect seed 可正确 lower
4. 新增同维 indirect step recurrence v1: 当 loop seed 在同维已含 indirect 时, 支持 step 中 `tta.indirect_reindex` 递推, 并以 `seed + iterCount * step` 形式构造当前迭代 index tensor
5. 更新 diagnostics: 对“无 indirect seed 却在 step 引入 indirect”的场景保留失败, 并给出明确原因 `loop-carried indirect recurrence requires indirect seed on same dim`
6. 新增正向测试 `loop_carried_addr_supported_indirect_recurrence`, 覆盖同维 indirect 递推 lower 行为
7. 支持 no-seed 同维 recurrence v1: 在 canonical direct dim (`offset=0, stride=1`) 上, 用 `iterCount == 0` 分支保留 direct 语义, `iterCount > 0` 使用 `iterCount * stepIndex`
8. 支持同维 recurrence 的 mask 递推 v1: 合并 step mask, 并对 `iterCount == 0` 保持 seed mask / all-true 语义
9. 放宽 recurrence step index: seed-indirect 路径已支持 dynamic 1D index
10. 测试迁移: `loop_carried_addr_unsupported_indirect_recurrence` 已从 diagnostics 迁移到 `tta-to-memref.mlir` 正向覆盖 (`loop_carried_addr_supported_indirect_recurrence_no_seed`)
11. 支持 no-seed recurrence 的非 canonical direct dim: 去除 `offset=0, stride=1` 硬前提, 保留原 direct dim 的 stride/offset 参与线性化.
12. 支持 no-seed + dynamic 1D step index: `iterCount` splat 和迭代 0 identity index 均支持动态长度, 不再要求 static 1D index.
13. 支持 no-seed + dynamic mask recurrence: 迭代 0 all-true mask 改为动态生成, 并修复单步 indirect recurrence 下 step mask 首次合并遗漏问题.
14. 新增正向测试 `loop_carried_addr_supported_indirect_recurrence_no_seed_dynamic`, 覆盖 dynamic index + dynamic mask + 非 canonical direct dim.
15. 新增诊断测试 `loop_carried_addr_unsupported_indirect_recurrence_no_seed_non_zero_direct_step`, 固化非法 recurrence 报错.
16. 支持跨维 indirect recurrence: 去除“step 中只允许单 recurrence dim”限制, 按 dim 独立聚合 step index/mask, 并以固定 dim 顺序更新 descriptor.
17. 支持跨维 mixed seed/no-seed recurrence: 同一 loop step 内可同时处理 seed-indirect 维与 no-seed 维.
18. 新增正向测试 `loop_carried_addr_supported_indirect_recurrence_multi_dim_mixed_seed`, 覆盖跨维 mixed recurrence lower 行为.
19. 新增诊断测试 `loop_carried_addr_unsupported_indirect_recurrence_multi_dim_mixed_seed_non_zero_direct_step`, 覆盖跨维 mixed recurrence 非法 direct-step 约束.

下一步:

1. PR4-3a 已落地, 下一步推进 PR4-3b: 补齐跨维 recurrence 的异常组合诊断 (shape/type/mask mismatch).
2. 为跨维 recurrence 增加更多 conversion 组合回归, 包括 dynamic dim + wrap + multi-indirect 的交叉覆盖.

验证快照:

1. `cmake --build build --target triton-xyz-opt -j8` 通过
2. `PATH="$(pwd)/build/bin:$PATH" .pixi/envs/default/bin/llvm-lit -v test/Conversion/tta-to-memref.mlir test/Conversion/tta-to-memref-diagnostics.mlir` 通过
3. `PATH="$(pwd)/build/bin:$PATH" .pixi/envs/default/bin/llvm-lit -v test/Conversion` 32 / 32 通过

---

## 13. 进度更新 (2026-02-27)

审查结论:

1. PR4-3b 已完成, 跨维 recurrence 的边界诊断与交叉组合回归已补齐.
2. Phase 4 维持进行中状态, PR4-1 / PR4-2 / PR4-3a / PR4-3b 均已落地.

本次新增:

1. 在 `TTAToMemref` loop-carried recurrence 路径新增精确诊断, 区分 step 同维 index shape/type mismatch 与 seed/step index shape/type mismatch.
2. 补充 mask 相关防御性诊断, 对 1D i1 约束与 shape mismatch 给出显式 failure reason, 避免统一落到 `unsupported loop-carried indirect recurrence`.
3. 新增 diagnostics 用例 `loop_carried_addr_unsupported_indirect_recurrence_multi_dim_step_index_shape_mismatch`, 覆盖跨维 recurrence 中同维 step index 合并 mismatch.
4. 新增 diagnostics 用例 `loop_carried_addr_unsupported_indirect_recurrence_multi_dim_seed_step_index_shape_mismatch`, 覆盖跨维 recurrence 中 seed/step index mismatch.
5. 新增正向用例 `loop_carried_addr_supported_indirect_recurrence_multi_dim_dynamic_wrap`, 覆盖 dynamic dim + wrap + multi-indirect 的跨维 mixed recurrence 交叉路径.

验证快照:

1. `cmake --build build --target triton-xyz-opt -j8` 通过.
2. `PATH="$(pwd)/build/bin:$PATH" .pixi/envs/default/bin/llvm-lit -v test/Conversion/tta-to-memref.mlir test/Conversion/tta-to-memref-diagnostics.mlir` 通过.
3. `PATH="$(pwd)/build/bin:$PATH" .pixi/envs/default/bin/llvm-lit -v test/Conversion` 32 / 32 通过.

下一步:

1. 若进入 PR4 收尾, 可评估是否将 recurrence 的 mask mismatch 细化为可恢复 merge 语义 (dynamic/static 兼容 cast), 以进一步缩小失败面.
