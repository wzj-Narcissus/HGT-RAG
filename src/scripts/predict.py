"""Run prediction for a single question or a full split.

Usage:
    # Single question (by qid):
    python -m src.scripts.predict --config configs/default.yaml --question_id q_001

    # All questions in a split:
    python -m src.scripts.predict --config configs/default.yaml --split dev
"""
import argparse
import json
import os

import torch
import yaml

from src.data.mmqa_loader import (
    load_mmqa_questions,
    load_mmqa_texts,
    load_mmqa_tables,
    load_mmqa_images,
)
from src.data.graph_builder import build_question_graph
from src.data.feature_builder import build_node_features
from src.data.hetero_converter import convert_to_heterodata
from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.models.hgt_model import HGTEvidenceModel
from src.trainers.losses import EvidenceSelectionLoss
from src.trainers.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.serialization import save_json

logger = get_logger(__name__)


def predict_question(question, texts, tables, images, trainer, encoder, cfg, proj_matrix=None):
    data_cfg = cfg["data"]
    graph_cfg = cfg.get("graph", {})
    image_dir = data_cfg.get("image_dir", "dataset/images")

    graph = build_question_graph(
        question, texts, tables, images,
        max_cells_per_table=graph_cfg.get("max_cells_per_table", 200),
        candidate_mode=graph_cfg.get("candidate_mode", "oracle"),
    )
    features = build_node_features(graph, encoder, image_dir=image_dir)

    if proj_matrix is not None:
        features = {
            nid: torch.nn.functional.normalize(emb @ proj_matrix, dim=-1)
            for nid, emb in features.items()
        }

    data = convert_to_heterodata(graph, features, labels=None, graph_cfg=graph_cfg)

    qid = question.get("qid", "unknown")
    return trainer.predict(data, qid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question_id", default=None, help="Single qid to predict")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    enc_cfg = cfg["encoder"]
    model_cfg = cfg["model"]
    output_cfg = cfg["output"]

    device = enc_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    enc_cfg_copy = dict(enc_cfg)
    enc_cfg_copy["device"] = device
    encoder = QwenVLFeatureEncoder.from_config(enc_cfg_copy)

    # Load corpus
    dataset_dir = data_cfg["dataset_dir"]
    split = args.split
    questions = load_mmqa_questions(dataset_dir, split=split)
    texts = load_mmqa_texts(dataset_dir)
    tables = load_mmqa_tables(dataset_dir)
    images = load_mmqa_images(dataset_dir)

    # Filter to single question if qid given
    if args.question_id:
        questions = [q for q in questions if q.get("qid") == args.question_id]
        if not questions:
            for fallback_split in ["dev", "train", "test"]:
                if fallback_split == split:
                    continue
                fallback_questions = load_mmqa_questions(dataset_dir, split=fallback_split)
                fallback_questions = [
                    q for q in fallback_questions if q.get("qid") == args.question_id
                ]
                if fallback_questions:
                    questions = fallback_questions
                    split = fallback_split
                    break
        if not questions:
            raise ValueError(f"Question {args.question_id} not found.")

    # Build model
    proj_dim = enc_cfg.get("projection_dim", None)
    proj_matrix = None
    if proj_dim and proj_dim < encoder.hidden_dim:
        proj_generator = torch.Generator(device="cpu")
        proj_generator.manual_seed(42)
        proj_matrix = torch.randn(
            encoder.hidden_dim,
            proj_dim,
            generator=proj_generator,
        ) / (encoder.hidden_dim ** 0.5)
        hidden_dim_in = proj_dim
    else:
        hidden_dim_in = encoder.hidden_dim
    model = HGTEvidenceModel(
        in_dim=hidden_dim_in,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        scoring_head=model_cfg.get("scoring_head", "mlp"),
    )

    loss_fn = EvidenceSelectionLoss()
    trainer = Trainer(model, loss_fn, cfg, device=device)

    ckpt_path = args.checkpoint or os.path.join(
        output_cfg["checkpoints_dir"], "best.pt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path, load_optimizer=False)

    os.makedirs(output_cfg["predictions_dir"], exist_ok=True)
    all_preds = []
    for q in questions:
        pred = predict_question(q, texts, tables, images, trainer, encoder, cfg, proj_matrix=proj_matrix)
        all_preds.append(pred)
        if args.question_id:
            print(json.dumps(pred, indent=2, ensure_ascii=False))

    if not args.question_id:
        out_path = os.path.join(
            output_cfg["predictions_dir"], f"predictions_{split}.json"
        )
        save_json(all_preds, out_path)
        logger.info(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
