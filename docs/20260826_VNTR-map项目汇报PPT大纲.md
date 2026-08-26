# VNTR-map 项目汇报 PPT 大纲

## 汇报定位

- 建议时长：15-20 分钟，另留 5 分钟讨论。
- 建议页数：16 页正文 + 1 页备用材料。
- 核心叙事：为什么普通线性表示不足 -> 如何分阶段确定边界、主体 motif、概率分解、read 证据和循环拓扑 -> 当前结果是否可信 -> 下一步如何补齐科学证据。
- 汇报对象：课题组、合作研究人员或阶段性项目评审。

## 第 1 页：封面

**标题**：VNTR-map：面向复杂 VNTR 的位置特异多环图构建

**副标题**：从主体 motif 分解到逐路径循环次数与变异证据

**页面内容**：项目名称、汇报人、单位、日期、GitHub 仓库。

**建议配图**：P5 SCC 宏观顺序与逐路径 copy-number 热图的局部裁剪。

**讲述重点**：本项目不是单纯调用 PGGB，而是在已有 pangenome graph 上增加适合 VNTR 解读的位置、重复次数、变异和证据层。

## 第 2 页：研究背景与问题

**标题**：为什么 VNTR 难以用普通线性序列或聚合图表达？

**核心信息**：

- 不同路径的重复次数不同，线性参考容易压缩或错配。
- 真实 VNTR 不是完全规则复制，包含变异 motif、移码、局部插入和异源 motif 穿插。
- 聚合可视化中的一个环可能混合所有路径，无法说明每条路径具体循环多少次。
- 相同 motif 可在不同位置出现，不能因序列相似而合并成同一全局环。

**建议配图**：线性序列、聚合单环、位置特异多环三种表示的概念对比。

## 第 3 页：项目目标与设计原则

**标题**：目标：既保留逐碱基序列，又提高图结构的可读性

**核心信息**：

- 每条输入路径必须能够从输出 GFA 精确重建。
- 先确定 VNTR 同源窗口与主体 motif，再处理局部复杂情况。
- 每个 repeat location 独立建模，每条路径保留各位置 copy number。
- 模型置信度、组装复现和原始 read 支持必须分层表达。
- GFA 服务于机器重建和图算法，TSV/PNG 服务于人工解释。

**建议配图**：五条设计原则组成的横向流程或层级图。

## 第 4 页：总体技术路线

**标题**：P1-P5 分阶段证据管线

**核心信息**：

1. P1：唯一侧翼锚点与共识边界。
2. P2：de novo primitive motif 与 MDL 分解。
3. P3：隐藏半马尔可夫概率分解与不确定性。
4. P4：组装事件目录与可选 read 证据分级。
5. P5：真实路径支持的位置特异 SCC 循环图。

**建议配图**：`PGGB GFA -> P1 -> P2 -> P3 -> P4 -> P5 -> GFA/TSV/PNG` 的主流程图。

**讲述重点**：后续阶段增加注释和证据，但不允许破坏前一阶段已经验证的路径序列。

## 第 5 页：P1 确定同源窗口与 VNTR 边界

**标题**：P1：先把不同路径放到可比较的同源窗口

**核心信息**：

- 在主体重复两侧 180 bp 窗口寻找 21 bp cohort unique anchors。
- 根据 motif run、概率动态规划和周期环三类内部方法形成边界共识。
- 使用 0-based、half-open 坐标记录边界和不确定区间。
- 当前 HPRC：47/47 路径具有左右唯一锚点并可精确重建。
- 当前状态仍为 `provisional_internal_consensus`，尚缺外部 TR 工具确认。

**建议配图**：P1 唯一侧翼锚定的共识边界总览 PNG。

## 第 6 页：P2 从头发现主体 motif

**标题**：P2：不预设 CAG/GCC，直接从队列发现 primitive motif

**核心信息**：

- 扫描 1-18 bp 周期，要求最低 copy 数、串联长度和路径支持率。
- 通过 primitive root、循环移位和反向互补归一化 motif family。
- 使用最小描述长度选择共享 motif 字典并控制过拟合。
- 动态规划将序列拆为 exact motif、variant motif、局部插入和复杂背景。
- HPRC 选择 AGC、CCG 两个 3 bp family；MDL 从 23554 降至 15050.5 bits。

**建议配图**：P2 motif 支持条形图和三个位置区块总览。

## 第 7 页：主体重复区块与逐路径次数

**标题**：把“小 run”提升为位置特异 VNTR repeat block

**核心信息**：

- HPRC 共形成 141 个 block，即 47 条路径 × R1/R2/R3 三个位置。
- R1 与 R3 即使同属 CCG family，也因位置不同保持为两个独立 block。
- 当前 copy-number 范围约为 R1 12-16、R2 19-32、R3 15-20。
- 单独散落在复杂背景中的 motif 不自动计入重复段。
- 变异 motif 与插入保留显式序列，不以“平均次数”替代。

**建议配图**：选择 4-6 条代表路径绘制 block 结构对比，并标注 copies/variant/insertion。

## 第 8 页：P3 概率模型与不确定性

**标题**：P3：隐藏半马尔可夫模型区分稳定主体与局部复杂片段

**核心信息**：

- 状态包括复杂背景和各 primitive motif family，状态段具有显式持续长度。
- 采用确定性 60/20/20 train/tune/test 划分。
- k-best Viterbi 给出最佳和第二佳分解。
- forward-backward 给出 segment/token posterior 和边界置信区间。
- HPRC：141 个 block，2 个低置信 block，47/47 精确重建。

**建议配图**：P3 block 平均持续长度和逐路径 posterior 透明度图。

**风险提示**：posterior 是当前模型和字典条件下的分解置信度，不是测序准确率或变异真实性概率。

## 第 9 页：局部复杂情况案例

**标题**：案例：短 motif-like 片段导致的错误切块如何修复？

**核心信息**：

- APG `seq12_1.0` 在 P2 中曾被短片段切成两个 6-copy 小 block。
- P3 将短片段按 3 bp 重新解释，并保留 6 bp AGC-family interruption。
- 最终恢复为一个 14-copy R1，而不是两个虚假的重复区块。
- 说明“先确定主体，再解释局部复杂性”比逐小 run 展示更稳定。

**建议配图**：P2 与 P3 对该路径的 before/after token 或 block 示意。

## 第 10 页：P4 变异事件与 read 证据

**标题**：P4：把组装中观察到的差异与真实 read 支持分开

**核心信息**：

- 事件类型：motif substitution、motif indel、local insertion、foreign-family interruption。
- read 证据接口记录覆盖、支持数、MAPQ、base quality、链方向和 caller。
- 状态分为高置信、uncertain、error-like 和 assembly-only unvalidated。
- 当前 HPRC 755 个事件、APG 450 个事件均未接入 reads，因此不能直接宣称是真实生物学变异。
- P4 不直接解析 BAM/CRAM，事件级 read TSV 由上游 GraphAligner/PanAligner 或位点 caller 生成。

**建议配图**：P4 事件位置散点图和 validation status 计数图。

## 第 11 页：P5 图论建模

**标题**：P5：用真实路径邻接和 SCC 表示位置特异循环

**核心信息**：

- 只保留 P path 实际支持的有向边，不依赖聚合 GFA 中的所有候选 L 边。
- Tarjan 强连通分量识别可反复绕行的最小拓扑单元。
- 将 SCC 压缩后得到 DAG，表达 VNTR 从左到右的宏观位置顺序。
- SCC 内可包含 variant motif 或插入形成的多节点循环。
- 通过 location 标签检查跨位置误合并和 R1/R2/R3 逆序路径。

**建议配图**：`B0 -> R1 -> R2 -> B2 -> R3 -> B3` 的 SCC condensation 图。

## 第 12 页：每条路径如何表达不同循环次数

**标题**：同一张图上多个环，每条路径具有独立 traversal count

**核心信息**：

- 环表示“可重复拓扑”，次数属于具体路径而不是环的全局属性。
- P 行保留重复节点的真实访问序列，逐路径环遍历表汇总每个位置的 copies。
- 热图每行对应一条路径，每列对应 R1/R2/R3，颜色与格内数字表示 copy number。
- HPRC：34 nodes、47 条路径支持边、25 个 SCC，其中 3 个 cyclic SCC。
- `unsupported_edge_count=0`、`cross_position_cycle_count=0`、condensation 为 DAG。

**建议配图**：P5 SCC 宏观顺序与逐路径 copy-number 热图。

## 第 13 页：端到端验证结果

**标题**：所有阶段均保持逐路径无损重建

**核心信息表格**：

| 数据集 | 路径数 | P2 blocks | P3 图 | P4 events | P5 cyclic SCC | 精确重建 |
|---|---:|---:|---|---:|---:|---:|
| HPRC | 47 | 141 | 34 nodes / 47 edges | 755 | 3 | 47/47 |
| APG + Refs | 28 | 84 | 29 nodes / 43 edges | 450 | 3 | 28/28 |

**补充信息**：当前仓库 19 项回归测试全部通过；统一调用器真实运行 P1-P5 成功。

**建议配图**：验证指标对勾矩阵，不使用夸大的性能百分比。

## 第 14 页：统一调用与可复现性

**标题**：注册表驱动的跨平台统一调用器

**核心信息**：

- `list / describe / stage / pipeline` 统一调用 P1-P5。
- 支持 dry-run、断点续跑、阶段参数覆盖和已有产物注入。
- 自动生成日志、状态 TSV、运行 manifest、历史 attempt 和 SHA256SUMS。
- 程序返回 0 但必需产物缺失时，明确标记为 `missing_output`。
- 新增 P6 或外部工具时，只需在 JSON 注册命令、依赖和产物。

**建议配图**：注册表 -> 调度器 -> P1-P5 -> manifest/artifacts 的软件架构图。

## 第 15 页：当前局限与科学风险

**标题**：现阶段哪些结论还不能过度解释？

**核心信息**：

- P1 边界主要来自内部共识，尚缺 TRF、MotifScope、uTR 等独立外部证据。
- P2/P3 motif 字典是当前 cohort 和窗口下的最简解释，不一定是唯一生物学生成机制。
- P3 posterior 不是变异真实性概率。
- P4 尚未接入 HiFi/ONT 原始 read 事件证据。
- 当前验证脚本中部分 HPRC 固定期望值需要改为位点配置或动态不变量。
- 简单 bubble 和高阶周期检测仍需更严格的图论与统计模型。

**建议配图**：按“已验证、概率支持、待 read 验证”分层的证据金字塔。

## 第 16 页：下一步计划与结论

**标题**：下一阶段：从可解释原型走向多位点、read-backed 验证

**优先顺序**：

1. 为 P1 接入 TRF/MotifScope/uTR 外部边界证据并统一坐标。
2. 建立 HiFi 与 ONT 的事件级 read evidence 生成流程，校准 P4 阈值。
3. 在每个训练折内重新发现 motif，完成无信息泄漏的 held-out 评估。
4. 将 P5 扩展到 path-supported superbubble、最小循环基和容错高阶周期。
5. 在更多基因和 VNTR 位点验证泛化能力。

**结论句**：VNTR-map 已实现从 PGGB 图到位置特异多环图的无损、可解释、可扩展流程；当前主要缺口已从“如何画图”转向“如何补足独立边界和原始 read 证据”。

## 第 17 页：讨论与备用材料

**标题**：Questions / Discussion

**备用内容**：

- GFA H/S/L/P 记录与项目标签速查。
- P1-P5 默认参数表。
- APG `seq12_1.0` 详细分解。
- 统一调用器命令示例。
- GitHub 分支、commit 和复现实验路径。

## 制作建议

- 每页只保留一个中心结论，标题尽量直接表达该结论。
- P1-P5 各使用统一颜色，但 P4 的“证据等级”应使用独立的中性色/警示色。
- 图中明确区分 assembly path、model posterior 与 read evidence。
- 所有 copy number 都注明是“每路径、每位置”的统计。
- 正文中不展示完整 GFA P 行；将其放入备用页，用 TSV/PNG 讲解主要结果。
- 在图注中注明数据集、路径数、坐标体系和是否接入 reads。
