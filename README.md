# MultiModalQA: Complex Question Answering over Text, Tables and Images

MultiModalQA is a challenging question answering dataset that requires joint reasoning over text, tables and images, consisting of 29,918 examples. This repository contains the MultiModalQA dataset, format description, and link to the images file.

For more details check out our ICLR21 paper ["MultiModalQA: Complex Question Answering over Text, Tables and Images"](https://openreview.net/pdf?id=ee6W5UgQLa),
and [website](https://allenai.github.io/multimodalqa/).

---

## HGT Evidence Selection Pipeline (新增)

本仓库新增了基于 **Heterogeneous Graph Transformer (HGT)** 的多模态证据选择流水线，使用 **Qwen3-VL-Embedding-8B** 作为统一跨模态编码器。

### 新增/修改的文件

| 文件 | 说明 |
|------|------|
| `configs/default.yaml` | 全局配置（编码器、图、训练超参等） |
| `src/data/io.py` | `.jsonl` / `.jsonl.gz` 读写工具 |
| `src/data/mmqa_loader.py` | `load_mmqa_questions/texts/tables/images` |
| `src/data/graph_builder.py` | `build_question_graph` — 异构图构造 |
| `src/data/label_builder.py` | `build_labels` — 监督标签生成 |
| `src/data/qwen_vl_encoder.py` | `QwenVLFeatureEncoder` — Qwen3-VL + fallback |
| `src/data/feature_builder.py` | `build_node_features` — 节点特征编码 |
| `src/data/hetero_converter.py` | `convert_to_heterodata` — 转 PyG HeteroData |
| `src/models/hgt_model.py` | `HGTEvidenceModel` — HGT + 打分头 |
| `src/trainers/losses.py` | 多任务损失（BCE + Ranking） |
| `src/trainers/metrics.py` | Recall@K / MRR / P/R/F1 |
| `src/trainers/trainer.py` | 训练 / 验证 / 预测 / checkpoint |
| `src/scripts/build_graphs.py` | 构图脚本 |
| `src/scripts/train.py` | 训练脚本 |
| `src/scripts/evaluate.py` | 验证脚本 |
| `src/scripts/predict.py` | 预测脚本 |
| `requirements_hgt.txt` | 新增依赖 |

---

### 安装依赖

```bash
pip install -r requirements_hgt.txt
```

> **注意**：`torch-geometric` 需要与 PyTorch 版本匹配，见 [PyG 安装指南](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)。

---

### 编码器：Qwen3-VL-Embedding-8B

默认编码器为 `Qwen/Qwen3-VL-Embedding-8B`，需要：
- `transformers >= 4.51.0`
- 约 16GB 显存（bf16）

**启用方式**（`configs/default.yaml`）：
```yaml
encoder:
  name: qwen3_vl_embedding_8b
  model_name_or_path: Qwen/Qwen3-VL-Embedding-8B
  dtype: bf16
  device: cuda
```

**下载模型**：
```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B
```

**若 Qwen 不可用，自动 fallback**：
- 文本编码切换到 `sentence-transformers/all-MiniLM-L6-v2`
- 图像编码返回零向量（打印 WARNING）
- 切换 fallback 方式：
```yaml
encoder:
  name: sentence_transformers
  fallback_text_model: sentence-transformers/all-MiniLM-L6-v2
```

---

### 图 Schema

**节点类型**：`query` / `textblock` / `table` / `cell` / `image` / `caption`

> `caption` 节点使用 `image.title` 伪造，不来自原始标注。

**边类型**：

| 边 | 方向 |
|----|------|
| `query_to_text` | query → textblock |
| `query_to_table` | query → table |
| `query_to_cell` | query → cell |
| `query_to_image` | query → image |
| `table_contains_cell` | table → cell |
| `text_refers_table` | textblock → table（启发式）|
| `text_refers_image` | textblock → image（启发式）|
| `cell_to_cell_row` | cell ↔ cell（同行相邻）|
| `cell_to_cell_col` | cell ↔ cell（同列相邻）|
| `image_has_caption` | image → caption |

---

### 核心函数说明

| 函数 | 位置 | 说明 |
|------|------|------|
| `load_mmqa_questions(dataset_dir, split)` | `mmqa_loader.py` | 加载问题，返回 list |
| `load_mmqa_texts/tables/images(dataset_dir)` | `mmqa_loader.py` | 加载文档，返回 id→dict |
| `build_question_graph(question, texts, tables, images)` | `graph_builder.py` | 构建异构图 dict |
| `build_labels(question, graph)` | `label_builder.py` | 生成 node_id→0/1 标签 |
| `build_node_features(graph, encoder, image_dir)` | `feature_builder.py` | 编码节点特征 |
| `convert_to_heterodata(graph, features, labels)` | `hetero_converter.py` | 转 PyG HeteroData |

---

### 运行流程

#### 1. 构图（必须先运行）

```bash
python -m src.scripts.build_graphs --config configs/default.yaml --split train
python -m src.scripts.build_graphs --config configs/default.yaml --split dev
```

快速调试（只跑100条）：
```bash
python -m src.scripts.build_graphs --config configs/default.yaml --split train --max_samples 100
```

输出保存到 `outputs/graphs/`。

#### 2. 训练

```bash
python -m src.scripts.train --config configs/default.yaml
```

Checkpoint 保存到 `outputs/checkpoints/`，最优模型为 `outputs/checkpoints/best.pt`。

#### 3. 验证

```bash
python -m src.scripts.evaluate --config configs/default.yaml --split dev
```

输出指标：`Recall@1/3/5`、`MRR`、`Precision/Recall/F1`（按 textblock / cell / image / overall）。

#### 4. 预测

```bash
# 单条问题
python -m src.scripts.predict --config configs/default.yaml --question_id 165d0bb820558b332d126e1ca216dde1

# 整个 split
python -m src.scripts.predict --config configs/default.yaml --split dev
```

---

### 已知限制

- `Caption` 节点使用 `image.title` 伪造，非真实 caption
- `text_refers_table` / `text_refers_image` 为保守启发式（同图内即连边）
- 第一版以 **oracle candidate** 模式为主，未实现检索模式
- 图像文件需单独下载放入 `dataset/images/`

---

### Changelog

- `23/04/2021` Initial release.
- `2025-XX` 新增 HGT 证据选择流水线



# MultiModalQA Dataset Format

In the [dataset](https://github.com/allenai/multimodalqa/tree/master/dataset) folder you will find the following file question and contexts files:
1) `MultiModalQA_train/dev/test.jsonl.gz` - contains questions and answers, for train, dev and test set respectively
2) `tables.jsonl.gz` - contains the tables contexts
3) `texts.jsonl.gz` - contains the texts contexts
4) `images.jsonl.gz` - contains the metadata of the images, the images themselves can be downloaded from [images.zip](https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip) 

# QA Files Format

Each line of the examples files (e.g. `MultiModalQA_train/dev.jsonl.gz`) contains one question, alongside its answers, metadata (described below, the all related context documents will be found there) and supporting context ids (the exact context ids that contain the answers and intermediate answers)

```json
{
  "qid": "5454c14ad01e722c2619b66778daa98b",
  "question": "who owns the rights to little shop of horrors?",
  "answers": ["answer1", "answer2"],
  "metadata": {},
  "supporting_context": [{
      "doc_id": "46ae2a8e7928ed5a8e5f9c59323e5e49",
      "doc_part": "table"
    },
    {
      "doc_id": "d57e56eff064047af5a6ef074a570956",
      "doc_part": "image"
    }]
}
```

`MultiModalQA_test.jsonl.gz` contains is of similar format, but does not contain `answers` 
nor `supporting_context`.

## A Single Answer

Each answer in the `answers` field contains an answer string that may be of type string or yesno, each answer points to the text, table or image context documents where it can be found (see context files for matching ids):

```json
{
  "answer": "some string here",
  "type": "string/yesno",
  "modality": "text/image/table",
  "text_instances": [{
          "doc_id": "b95b35eabfc80a0f1a8fd8455cd6d109",
          "part": "text",
          "start_byte": 345,
          "text": "AnswerText"
        }],
  "table_indices": [[5, 2]],
  "image_instances": [{
              "doc_id": "d57e56eff064047af5a6ef074a570956",
              "doc_part": "image"
            }]
}
```

## A Single Question Metadata

The metadata of each question contains its type, modalities required to solve it, the wikipedia entities that appear in the question and in the answers, the machine generated question (the question before human rephrasing), as well as an annotation field containing the rephrasing accuracy and confidence (between 0 and 1), and a list of texts docs ids and image docs ids and table id that are part of the full context for
this question (some context docs contain the answer and some are distractors).
We include a list of intermediate answers, these are the answers of the sub-questions composing the multi-modal question, providing supervision for multi-step training.  

```json
{
    "type": "Compose(TableQ,ImageListQ)",
    "modalities": [
      "image",
      "table"
    ],
    "wiki_entities_in_question": [
      {
        "text": "Domenico Dolce",
        "wiki_title": "Domenico Dolce",
        "url": "https://en.wikipedia.org/wiki/Domenico_Dolce"
      }
    ],
    "wiki_entities_in_answers": [],
    "pseudo_language_question": "In [Members] of [LGBT billionaires] what was the [Net worth USDbn](s) when the [Name] {is completely bald and wears thick glasses?}",
    "rephrasing_meta": {
      "accuracy": 1.0,
      "edit_distance": 0.502092050209205,
      "confidence": 0.7807520791930855
    },
    "image_doc_ids": [
      "89c1b7c3c061cc80bb98d99cbbec50dd",
      "0f3858e2186b2030b77c759fc727e20b"
    ],
    "text_doc_ids": [
      "498369348c988d866b5fac0add45bac5",
      "57686242cf542e30cbad13037017b478"
    ],
    "intermediate_answers": ["single_answer_format(1)", "single_answer_format(2)"], 
    "table_id": "46ae2a8e7928ed5a8e5f9c59323e5e49"
  }
```

# A Single Table Format

Each line of `tables.jsonl.gz` represents a single table. `table_rows` is a list of rows, and each row contains is a list of cells. Each cell is provided with its text string and wikipedia entities. `header` provides for each column in the table: its name alongside parsing metadata computed such as NERs and item types. 

```json
{
  "title": "Dutch Ruppersberger",
  "url": "https://en.wikipedia.org/wiki/Dutch_Ruppersberger",
  "id": "dcd7cb8f23737c6f38519c3770a6606f",
  "table": {
    "table_rows": [
      [
        {
          "text": "Baltimore County Executive",
          "links": [
            {
              "text": "Baltimore County Executive",
              "wiki_title": "Baltimore County Executive",
              "url": "https://en.wikipedia.org/wiki/Baltimore_County_Executive"
            }
          ]
        },
      ]
    ],
    "table_name": "Electoral history",
    "header": [
      {
        "column_name": "Year",
        "metadata": {
          "parsed_values": [
            1994.0,
            1998.0
          ],
          "type": "float",
          "num_of_links": 9,
          "ner_appearances_map": {
            "DATE": 10,
            "CARDINAL": 1
          },
          "is_key_column": true,
          "entities_column": true
        }
      }
    ]
  }
}
```

# A Single Image Metadata Format

Each line in `images.jsonl.gz` holds metadata for each image. The `path` provided points to the image file  in the provided images directory.

```json
{
  "title": "Taipei",
  "url": "https://en.wikipedia.org/wiki/Taipei",
  "id": "632ea110be92836441adfb3167edf8ff",
  "path": "Taipei.jpg"
}
```

# A Single Text Metadata Format
Each line in `texts.jsonl.gz` represents a single text paragraph. 

```json
{
  "title": "The Legend of Korra (video game)",
  "url": "https://en.wikipedia.org/wiki/The_Legend_of_Korra_(video_game)",
  "id": "16c61fe756817f0b35df9717fae1000e",
  "text": "Over three years after its release, the game was removed from sale on all digital storefronts on December 21, 2017."
}
```
