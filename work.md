你是一名资深机器学习工程师与代码实现助手。请你在**当前工作目录**中直接完成代码实现。

## 当前工作上下文

- 我当前所在目录是：`git clone` 下来的 **MultimodalQA 仓库根目录**
- 你必须先检查当前仓库已有的目录结构、脚本、数据文件、依赖文件
- **优先增量修改现有仓库**
- 不要无视现有代码从零重新造一个完全独立项目，除非当前仓库中完全没有可复用结构
- 所有新增代码都要尽量与现有仓库风格兼容
- 如果已有 README、requirements、data loader、训练脚本，请优先复用或扩展
- 如果需要新增目录，请保持简洁清晰

---

## 总目标

请你在当前 MultimodalQA 仓库下，直接实现一个**可运行的 HGT 异构图训练流水线**，用于：

> 基于 **MultiModalQA / MMQA** 数据，构建 question-specific heterogeneous graph，  
> 使用 **PyTorch Geometric + HGT** 做多模态证据选择 / 重排序，  
> 并使用 **Qwen3-VL-Embedding-8B** 作为统一跨模态编码模型。

请不要只输出设计思路。  
请你直接：

1. 检查仓库
2. 创建/修改代码
3. 补充配置
4. 补充 README
5. 给出可运行命令
6. 确保最小可运行

---

# 1. 编码模型要求（非常重要）

本项目的节点编码统一优先使用：

- `Qwen/Qwen3-VL-Embedding-8B`

请你围绕这个模型设计特征提取模块，并满足以下要求：

## 1.1 编码目标
该模型需要用于编码以下内容：

- `Query`
- `TextBlock`
- `Table`
- `Cell`
- `Caption`
- `Image`

也就是说，需要支持：
- 文本编码
- 图像编码
- 跨模态统一 embedding 空间

## 1.2 实现要求
请封装统一接口，例如：

- `QwenVLFeatureEncoder`
- 或等价命名

接口需至少支持：

- `encode_texts(texts: list[str]) -> torch.Tensor`
- `encode_images(image_paths: list[str]) -> torch.Tensor`
- `encode_text_image_pairs(...)`（如有必要）
- 批量推理
- device 管理
- fp16/bf16 可选
- 图像文件缺失时 fallback

## 1.3 兼容与降级策略
如果当前环境中：
- 没有安装该模型依赖
- 显存不足
- 某些 API 不兼容

请实现**清晰的降级方案**：

### 第一优先
- 仍然保留 `Qwen3-VL-Embedding-8B` 接口
- 并在配置中作为默认编码器

### 第二优先
- 提供 fallback encoder：
  - 文本：`sentence-transformers` / `transformers`
  - 图像：CLIP 或随机/零向量兜底

但请注意：
- **默认配置必须优先指向 `Qwen3-VL-Embedding-8B`**
- README 中必须明确说明如何启用 / 下载 / 切换该模型

---

# 2. 图 schema（严格固定）

请严格使用以下节点与边，不要擅自增加新的核心节点类型。

## 2.1 节点类型
- `Query`
- `TextBlock`
- `Table`
- `Cell`
- `Image`
- `Caption`

其中：
- `Caption` 不来自原始标注
- **始终使用 `image.title` 伪造 caption**

---

## 2.2 边类型
- `query_to_text`
- `query_to_table`
- `query_to_cell`
- `query_to_image`
- `table_contains_cell`
- `text_refers_table`
- `text_refers_image`
- `cell_to_cell_row`
- `cell_to_cell_col`
- `image_has_caption`

---

# 3. 数据读取要求

请兼容 MultiModalQA 常见文件格式，并自动识别：

- `train.jsonl` / `dev.jsonl` / `test.jsonl`
- `texts.jsonl`
- `tables.jsonl`
- `images.jsonl`

同时兼容：
- `.jsonl`
- `.jsonl.gz`

请优先检查当前仓库已有的数据目录和加载脚本，尽量复用。

---

# 4. 图构建规范

请针对**每个问题**构建一个 question-specific graph。

## 4.1 节点构造

### Query
- `node_id = q::{qid}`

### TextBlock
- `node_id = text::{doc_id}`

### Table
- `node_id = table::{table_id}`

### Cell
- `node_id = cell::{table_id}::{row_idx}::{col_idx}`

### Image
- `node_id = image::{image_id}`

### Caption
- `node_id = caption::{image_id}`
- `text = image.title`

---

## 4.2 边构造

### query_to_text
Query 指向当前问题候选 texts

### query_to_table
Query 指向当前问题候选 table

### query_to_cell
Query 指向候选 cells
- 第一版允许直接连向该表所有 cells
- 若 `answers[*].table_indices` 可得，务必确保正例 cell 被连接

### query_to_image
Query 指向当前问题候选 images

### table_contains_cell
Table 指向所有 cells

### text_refers_table
第一版保守启发式：
- 同一 question graph 内，若 text 和 table 同时存在，则连边

### text_refers_image
第一版保守启发式：
- 同一 question graph 内，若 text 和 image 同时存在，则连边

### cell_to_cell_row
同一行相邻 cell 连边

### cell_to_cell_col
同一列相邻 cell 连边

### image_has_caption
Image 指向 Caption

---

# 5. 标签构造要求

请根据 MMQA 原始字段自动生成监督标签。

## 5.1 TextBlock 正例
若 text 出现在：
- `supporting_context`
- `answers[*].text_instances`

则标为正例，否则负例

## 5.2 Table 正例
若 table 出现在：
- `supporting_context`
- 或与 `answers[*].table_indices` 对应

则为正例

## 5.3 Cell 正例
若 cell 命中：
- `answers[*].table_indices`

则为正例

## 5.4 Image 正例
若 image 出现在：
- `supporting_context`
- `answers[*].image_instances`

则为正例

## 5.5 Caption 正例
Caption 标签与对应 Image 同步

---

# 6. 节点特征构造要求

请实现统一的 feature builder，默认优先调用 `Qwen3-VL-Embedding-8B`。

## 6.1 文本节点输入格式

### Query
直接使用 question 文本

### TextBlock
建议：
- `title [SEP] text`

### Table
建议 flatten：
- `table_title [SEP] header1 | header2 | ... [SEP] row1 ; row2 ; ...`

### Cell
不要只用 cell text
请尽量拼接：
- `header_text [SEP] row_context [SEP] cell_text`

### Caption
直接使用 `image.title`

---

## 6.2 图像节点
对于 `Image`：
- 若本地图像存在，使用 `Qwen3-VL-Embedding-8B` 图像编码
- 若图片缺失，返回 fallback embedding，并打印 warning
- 不允许因为图像缺失而中断流程

---

# 7. 训练目标

第一版只要求先做：

## 7.1 evidence selection / reranking

重点预测以下节点是否为 supporting evidence：
- `TextBlock`
- `Cell`
- `Image`

可选支持：
- `Caption`
- `Table`

---

## 7.2 模型要求
请使用：
- `PyTorch Geometric`
- `HeteroData`
- `HGTConv`

要求：
- 2 层或 3 层 HGT，可配置
- hidden dim 可配置
- heads 可配置
- dropout 可配置

---

## 7.3 打分头
请至少实现两类 scoring head，并可配置切换：

### 方案 A
- `MLP(node_embedding) -> logit`

### 方案 B
- `score(query, node) = h_q^T W h_node`

至少对以下节点有打分输出：
- `TextBlock`
- `Cell`
- `Image`

---

# 8. Loss 要求

请实现以下训练模式并可配置：

## 8.1 BCE 节点分类损失
用于：
- `TextBlock`
- `Cell`
- `Image`
- `Caption`（可选）

## 8.2 Ranking Loss
实现 margin ranking loss

## 8.3 多任务总损失
形如：
- `loss = lambda_text * L_text + lambda_cell * L_cell + lambda_image * L_image + lambda_caption * L_caption + lambda_rank * L_rank`

---

# 9. 候选集策略

请支持：

## 9.1 Oracle 模式
直接使用 question metadata 中给出的：
- text_doc_ids
- table_id
- image_doc_ids

这是第一版必须完整实现的模式。

## 9.2 Retrieval-lite 模式
如果时间允许，再补充：
- 基于 embedding similarity 的 top-k 检索

但优先保证 Oracle 模式完全可跑。

---

# 10. 项目结构要求

请先检查当前仓库结构，再决定新增哪些文件。  
如果仓库中没有合适结构，可参考以下组织：

- `configs/default.yaml`
- `src/data/io.py`
- `src/data/mmqa_loader.py`
- `src/data/graph_builder.py`
- `src/data/feature_builder.py`
- `src/data/label_builder.py`
- `src/models/hgt_model.py`
- `src/models/heads.py`
- `src/trainers/trainer.py`
- `src/trainers/losses.py`
- `src/trainers/metrics.py`
- `src/utils/seed.py`
- `src/utils/logging.py`
- `src/utils/serialization.py`
- `src/scripts/build_graphs.py`
- `src/scripts/train.py`
- `src/scripts/evaluate.py`
- `src/scripts/predict.py`

如果当前仓库已有类似模块，请优先复用和扩展。

---

# 11. 必须暴露的核心函数

请在代码中明确提供并使用下列函数：

- `load_mmqa_questions(...)`
- `load_mmqa_texts(...)`
- `load_mmqa_tables(...)`
- `load_mmqa_images(...)`
- `build_question_graph(...)`
- `build_node_features(...)`
- `build_labels(...)`
- `convert_to_heterodata(...)`

README 中也要解释这些函数的职责。

---

# 12. 中间结果保存

请支持保存以下中间结果：

- 规范化 question 样本
- graph json
- node features
- HeteroData 对象

建议输出目录：

- `outputs/graphs/`
- `outputs/features/`
- `outputs/hetero/`
- `outputs/checkpoints/`
- `outputs/predictions/`

---

# 13. 评估指标

请实现并在验证阶段输出：

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`
- `Evidence Precision`
- `Evidence Recall`
- `Evidence F1`

分别统计：
- TextBlock
- Cell
- Image
- overall

---

# 14. 日志与健壮性要求

## 14.1 日志
训练和预处理时打印：
- 数据量
- 节点数
- 边数
- 正负样本比例
- loss
- 指标

## 14.2 健壮性
必须处理：
- 缺字段
- 空表
- 空文本
- 缺图像
- `.jsonl.gz`
- 不同版本 MMQA 字段名差异

不要轻易崩溃，应给 warning 并继续。

---

# 15. README 要求

请补充或更新 README，至少包含：

1. 项目简介
2. 当前仓库中新增/修改了哪些内容
3. MMQA 数据格式说明
4. 图 schema 说明
5. Qwen3-VL-Embedding-8B 的使用方式
6. 若模型不可用时如何 fallback
7. 如何构图
8. 如何训练
9. 如何验证
10. 如何预测
11. 已知限制：
   - Caption 为 `image.title`
   - `text_refers_table` / `text_refers_image` 为启发式
   - 第一版以 oracle candidate 为主

---

# 16. 最低可运行标准

最终必须满足：

1. 可以构图
2. 可以训练
3. 可以输出 checkpoint
4. 可以验证
5. 可以预测
6. 不允许只留 TODO 或伪代码

---

# 17. 命令行要求

请尽量让我可以直接运行以下命令：

```bash
python -m src.scripts.build_graphs --config configs/default.yaml
python -m src.scripts.train --config configs/default.yaml
python -m src.scripts.evaluate --config configs/default.yaml
python -m src.scripts.predict --config configs/default.yaml --question_id q_001
```

如果当前仓库结构不适合该命令格式，请在 README 中明确给出替代命令。

---

# 18. 实施顺序要求

请按以下顺序执行，不要停留在计划阶段：

### Phase 1
- 检查当前仓库结构
- 识别可复用模块
- 实现数据读取
- 实现 graph builder
- 在小样本上导出 graph json

### Phase 2
- 实现 `Qwen3-VL-Embedding-8B` 特征编码模块
- 实现 fallback encoder
- 实现 `convert_to_heterodata`

### Phase 3
- 实现 HGT 模型
- 实现 loss
- 实现训练/验证/预测脚本

### Phase 4
- 保存中间结果
- 更新 README
- 给出运行说明

---

# 19. 最终输出要求

完成后请不要只说“已完成”。  
请明确告诉我：

1. 你检查到了哪些现有文件/目录
2. 你新增了哪些文件
3. 你修改了哪些文件
4. 每个核心文件负责什么
5. 如何安装依赖
6. 如何准备数据
7. 如何运行构图、训练、验证、预测

---

# 20. 最后提醒

- 直接开始修改当前仓库
- 优先复用现有结构
- 默认编码模型必须优先是 `Qwen/Qwen3-VL-Embedding-8B`
- 如果需要 fallback，也必须保留 Qwen 接口
- 不要偏离以下固定 schema

## 节点
- Query
- TextBlock
- Table
- Cell
- Image
- Caption

## 边
- query_to_text
- query_to_table
- query_to_cell
- query_to_image
- table_contains_cell
- text_refers_table
- text_refers_image
- cell_to_cell_row
- cell_to_cell_col
- image_has_caption

现在请直接检查当前仓库并开始实现。

补充要求：

1. 请优先查找当前环境是否已安装 transformers、qwen 相关依赖。
2. 如果 `Qwen3-VL-Embedding-8B` 的标准调用方式与你熟悉的接口不同，请先在代码中封装适配层，不要把模型调用逻辑散落在各处。
3. 请把 Qwen 编码器封装在单独文件中，例如：
   - `src/data/qwen_vl_encoder.py`
4. 请让 feature builder 通过配置文件选择：
   - `encoder.name = qwen3_vl_embedding_8b`
5. 若模型不可用，请自动 fallback，但日志中必须明确提示：
   - 当前未成功加载 Qwen3-VL-Embedding-8B
   - 已切换到备用编码器



     ---
  最终评分

  ┌────────────┬───────────────────┬────────────────────┐
  │    维度    │  Embedding-Only   │        HGT         │
  ├────────────┼───────────────────┼────────────────────┤
  │ 整体性能   │ ⭐⭐⭐⭐ (52.2%)  │ ⭐⭐⭐⭐ (55.3%)   │
  ├────────────┼───────────────────┼────────────────────┤
  │ TextBlock  │ ⭐⭐⭐ (51.5%)    │ ⭐⭐⭐⭐⭐ (69.3%) │
  ├────────────┼───────────────────┼────────────────────┤
  │ Table      │ ⭐⭐⭐⭐⭐ (100%) │ ⭐⭐⭐⭐⭐ (100%)  │
  ├────────────┼───────────────────┼────────────────────┤
  │ Cell       │ ⭐⭐ (20.2%)      │ ⭐ (16.5%)         │
  ├────────────┼───────────────────┼────────────────────┤
  │ 实现复杂度 │ ⭐⭐⭐⭐⭐        │ ⭐⭐               │
  └────────────┴───────────────────┴────────────────────┘

  总结: HGT的价值主要体现在TextBlock (+34.6%)，但Cell性能下降抵消了部分收益。如果修复Cell问题，HGT将是明确的最优选择。

  完整对比分析已保存至: /root/multimodalqa/docs/Embedding_vs_HGT_Comparison.md

增益偏小。
  从消融看：

  - Embedding-only：MRR 0.5222
  - HGT：MRR 0.5530
  - 绝对提升：+0.0308
  - 相对提升：+5.9%

  我看了当前实现，主要有几个原因。

  1. 输入特征本身已经很强

  configs/default.yaml:20-21
  - Qwen3-VL 原始 4096 维
  - 先投影到 256 维

  纯 embedding 在 dev 上已经有：

  - overall MRR = 0.5222
  - table MRR = 1.000
  - image MRR = 0.4247
  - caption MRR = 0.4691

  说明基础向量已经很强，尤其 table 已经封顶了。
  这会让 HGT 的可提升空间天然变小。

  2. 图结构信息不够“强监督”

  src/data/graph_builder.py
  当前图里真正可靠的边主要是：

  - query → text/table/cell/image
  - table → cell
  - cell ↔ cell

  但还有一部分边是启发式的：

  - text_refers_table
  - text_refers_image

  而且 caption 其实只是 image.title 伪造出来的，不是真实 caption。
  这意味着：

  - 图里有些边的信息增益有限
  - 甚至可能引入噪声
  - HGT未必能比直接相似度学到更多

  3. 训练目标和“检索目标”不完全一致

  src/trainers/losses.py
  当前 loss 是：

  - per-type BCE
  - margin ranking loss

  问题在于：

  - BCE 强调分类阈值
  - 你的核心评估却是 Recall@K / MRR
  - 这两者并不完全一致

  所以模型可能学会了“更像分类器”，而不是“更好的排序器”。

  这也解释了为什么：
  - HGT 的 Precision 提高很多
  - 但 Recall 降很多
  - 排序指标只小幅提升

  4. Cell 极度不平衡，拉低了整体收益

  你这里最明显的短板就是 cell：

  - Embedding-only cell MRR = 0.2024
  - HGT cell MRR = 0.1648

  也就是 HGT 在 cell 上反而退步。

  原因很可能就是类别极不平衡：
  - cell 正负比大约 1:173

  而当前 BCE 没有做 pos_weight
  src/trainers/losses.py:43
  - 直接 nn.BCEWithLogitsLoss()

  这会导致模型在 cell 上倾向于全判负。
  结果就是：

  - HGT在text上提升明显
  - 但被cell退化抵消了

  5. HGT结构本身偏轻量

  configs/default.yaml:33-37
  - hidden_dim = 256
  - num_heads = 4
  - num_layers = 2

  src/models/hgt_model.py:66-74
  只有 2层 HGTConv。

  这套结构比较保守，优点是稳，但缺点是：
  - 图推理深度有限
  - 跨模态/多跳传播能力有限
  - 对强embedding只能做“轻微修正”，很难带来大幅跃升

  ---
  当前结果怎么理解

  可以这样总结：

  HGT有效，但没有“充分发挥价值”

  因为：

  - 对 textblock 提升很明显
  - 对 image 有小幅提升
  - 对 table 没提升空间
  - 对 cell 反而退化

  所以最终只得到一个 小幅整体提升。

  换句话说，不是HGT完全没用，而是收益被几个瓶颈压住了。

  ---
  我认为最值得优先优化的点

  优先级1：先修 Cell loss

  这是最大瓶颈。

  建议优先做：

  方案A：给 BCE 加 pos_weight

  这是最小改动、最直接的。

  现在是：
  src/trainers/losses.py:43
  self.bce = nn.BCEWithLogitsLoss()

  更合理的是按类型加权，特别是 cell。
  例如 cell 的正样本太少，就应显著提高正样本权重。

  预期效果：
  - cell MRR 回到至少 embedding-only 水平
  - overall MRR 可能直接再涨一截

  方案B：Focal Loss

  如果想更系统地处理极端不平衡，Focal Loss 比 BCE 更合适。

  ---
  优先级2：让训练目标更贴近排序

  你最终看的是：

  - Recall@1/3/5
  - MRR

  那训练也应该更偏 ranking。

  当前只有一个比较基础的 margin ranking loss：
  src/trainers/losses.py:99-131

  可以考虑：

  - 提高 lambda_rank
  - 降低部分 BCE 权重
  - 或改成更强的 listwise / pairwise ranking objective

  当前配置：
  configs/default.yaml:47-52
  - lambda_text = 1.0
  - lambda_cell = 1.0
  - lambda_image = 1.0
  - lambda_caption = 0.5
  - lambda_rank = 0.5

  这个 ranking loss 权重其实偏保守。
  如果目标是提升 MRR/Recall@K，我会优先尝试：

  - lambda_rank: 1.0 或 2.0

  ---
  优先级3：不要让所有模态共享同一种优化难度

  现在所有类型都走统一套路，但它们难度差异很大：

  - table：已经几乎 solved
  - text：HGT收益最大
  - image：中等
  - cell：最难且极不平衡

  所以最合理的是做 typed optimization：

  - table 保守
  - text 加强 ranking
  - cell 单独加权/采样
  - caption 降权甚至去掉

  特别是 caption，目前本身就是伪caption，可能对整体帮助有限。

  ---
  优先级4：检查图边是否真的提供增益

  当前一些边是启发式的，可能噪声大于收益：

  - text_refers_table
  - text_refers_image

  可以做边消融：

  1. 只保留 query→* + table→cell + cell↔cell
  2. 再加 heuristic edges
  3. 比较性能

  有可能：
  - 文本提升来自真正有用的边
  - 某些 heuristic edge 在拖后腿

  ---
  如果只做最小成本优化，我建议这三步

  第一组实验：先救 cell

  1. lambda_cell 从 1.0 提到 3.0 / 5.0
  2. BCE 改为 带 pos_weight
  3. 重新训练看 cell MRR 是否至少超过 0.20

  第二组实验：强化 ranking

  1. lambda_rank 从 0.5 提到 1.0
  2. 保持其他不变
  3. 看 overall MRR / Recall@5 是否提升

  第三组实验：去掉噪声模态/边

  1. 降低 caption 权重，甚至设为 0
  2. 检查 heuristic edges 的影响

  ---
  我对当前结论的判断

  结论一句话

  目前不是“HGT没用”，而是“HGT被数据不平衡、训练目标不匹配和图结构噪声限制住了”。

  更具体一点

  - textblock 上 HGT 很值
  - table 上没必要期待大提升
  - cell 是最大短板
  - 整体提升少，主要不是模型深度不够，而是优化目标没打准

  ---
  最建议你先做的改动顺序

  1. 给 cell 加正样本权重
  2. 提高 ranking loss 权重
  3. 弱化 caption
  4. 做边消融
  5. 再考虑更深层的 HGT / 更复杂结构

  如果你愿意，我下一步可以直接帮你做一版最小改动优化方案，只改：
  - losses.py
  - configs/default.yaml

  目标就是专门提升：
  - cell MRR
  - overall MRR

  这样最省事，也最容易验证。

   Recommended approach

     1. 统一更新边 schema，移除 query_to_cell
       - 修改 src/data/graph_builder.py
           - 从 edges 字典中删除 query_to_cell
         - 删除为每个 cell 添加 query_to_cell 的逻辑
         - 保留 text_refers_table / text_refers_image 作为可被后续重写的边名
       - 修改 src/data/hetero_converter.py
           - 从 EDGE_SCHEMA 中删除 ("query", "query_to_cell", "cell")
       - 修改 src/models/hgt_model.py
           - 从 EDGE_TYPES 中删除 ("query", "query_to_cell", "cell")
     2. 新增图配置项，控制相似度建边
       - 修改 configs/default.yaml 的 graph: 段，新增：
           - text_refers_table_threshold
         - text_refers_image_threshold
       - 阈值直接基于当前模型输入空间中的余弦相似度（特征已归一化；投影后也再次归一化）
     3. 在共享转换层实现“基于向量相似度重建跨模态边”
       - 优先在 src/data/hetero_converter.py 增加一个共享辅助函数，例如：
           - 输入：graph, features, graph_cfg
         - 输出：重写后的 graph["edges"]["text_refers_table"] 与 graph["edges"]["text_refers_image"]
       - 规则：
           - text_refers_table：对每个 textblock 节点与 table 节点做相似度比较，仅当 sim >= text_refers_table_threshold 时加边
         - text_refers_image：对每个 textblock 节点与 image 节点做相似度比较，仅当 sim >= text_refers_image_threshold 时加边
       - 采用与现有 embedding-only 一致的余弦相似度实现风格，可参考 src/scripts/evaluate_embedding_only.py
       - 当没有任何 pair 超阈值时，允许边集为空，不回退到旧的全连接启发式
     4. 把共享边增强接入 train / predict 路径
       - 修改 src/scripts/train.py
           - 在 feature cache 已完成、调用 convert_to_heterodata(...) 之前，对每个 graph 应用相似度边增强
         - 这样 train 与 evaluate（其复用 load_hetero_dataset）会共享同一逻辑
       - 修改 src/scripts/predict.py
           - 在 build_node_features(...) 和可选投影之后、convert_to_heterodata(...) 之前，应用同一边增强函数
       - 这样单样本预测与批量训练/评估都使用一致的边构造方式
     5. 确保使用“最终输入给 HGT 的向量空间”做相似度判断
       - 在 train 路径中，相似度应基于 feature cache 中最终送入模型的向量（包含投影后的 256 维特征）
       - 在 predict 路径中，相似度应基于投影后的 features
       - 这样阈值含义在训练和推理中一致
     6. 重建受 schema 影响的缓存/产物
       - 重新生成 graph JSON：src/scripts/build_graphs.py
           - 目的是彻底去掉历史 query_to_cell 字段
       - 重新生成 hetero .pt 缓存：src/scripts/train.py 产出的 outputs/hetero/...
           - 因为旧 .pt 内嵌了旧 relation schema
       - 现有 feature cache 通常可复用，但如果实现中修改了投影/归一化假设，再评估是否需要重建