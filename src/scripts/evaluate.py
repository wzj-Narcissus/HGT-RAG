"""Evaluate a trained checkpoint on dev/test split.

Usage:
    python -m src.scripts.evaluate --config configs/default.yaml [--split dev] [--checkpoint outputs/checkpoints/best.pt]
"""
import argparse
import glob
import os

import torch
import yaml

from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.models.hgt_model import HGTEvidenceModel
from src.scripts.train import load_hetero_dataset
from src.trainers.losses import EvidenceSelectionLoss
from src.trainers.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.serialization import save_json

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    enc_cfg = cfg["encoder"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    device = enc_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    graphs_dir = output_cfg["graphs_dir"]
    image_dir = data_cfg.get("image_dir", "dataset/images")

    enc_cfg_copy = dict(enc_cfg)
    enc_cfg_copy["device"] = device
    encoder = QwenVLFeatureEncoder.from_config(enc_cfg_copy)

    # Load graphs
    pattern = os.path.join(graphs_dir, f"{args.split}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No graphs found: {pattern}")
    dataset = load_hetero_dataset(
        graphs_dir, args.split, encoder, image_dir, cfg,
        cache_dir=output_cfg.get("features_dir", "outputs/features"),
        chunk_size=train_cfg.get("chunk_size"),
    )

    # Build model
    proj_dim = enc_cfg.get("projection_dim", None)
    hidden_dim_in = proj_dim if proj_dim and proj_dim < encoder.hidden_dim else encoder.hidden_dim
    model = HGTEvidenceModel(
        in_dim=hidden_dim_in,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        scoring_head=model_cfg.get("scoring_head", "mlp"),
    )

    loss_fn = EvidenceSelectionLoss(
        lambda_text=train_cfg.get("lambda_text", 1.0),
        lambda_cell=train_cfg.get("lambda_cell", 1.0),
        lambda_image=train_cfg.get("lambda_image", 1.0),
        lambda_caption=train_cfg.get("lambda_caption", 0.5),
        lambda_rank=train_cfg.get("lambda_rank", 0.5),
        margin=train_cfg.get("margin", 1.0),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha"),
    )

    trainer = Trainer(model, loss_fn, cfg, device=device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(output_cfg["checkpoints_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path, load_optimizer=False)

    metrics = trainer.evaluate(dataset)

    out_path = os.path.join(output_cfg["predictions_dir"], f"metrics_{args.split}.json")
    os.makedirs(output_cfg["predictions_dir"], exist_ok=True)
    save_json(metrics, out_path)
    logger.info(f"Metrics saved to {out_path}")

    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
