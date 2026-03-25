"""Training / evaluation / prediction loop."""
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import torch
from torch.amp import GradScaler, autocast
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.models.hgt_model import HGTEvidenceModel
from src.trainers.losses import EvidenceSelectionLoss
from src.trainers.metrics import EvidenceMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: HGTEvidenceModel,
        loss_fn: EvidenceSelectionLoss,
        cfg: Dict,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        self.metrics = EvidenceMetrics()

        train_cfg = cfg.get("training", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.output_cfg = cfg.get("output", {})
        self._best_metric_name: Optional[str] = None
        self._best_metric_value: Optional[float] = None
        self._best_metrics: Dict[str, float] = {}

        # Automatic Mixed Precision (AMP) support
        self.scaler = GradScaler('cuda') if train_cfg.get("use_amp", False) and device == "cuda" else None
        if self.scaler:
            logger.info("Automatic Mixed Precision (AMP) enabled")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: Union[DataLoader, List[HeteroData]], epoch: int) -> float:
        """Train for one epoch.

        Args:
            dataloader: DataLoader or list of HeteroData objects
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_pos_total = {ntype: 0 for ntype in ["textblock", "cell", "image", "caption", "table"]}
        n_neg_total = {ntype: 0 for ntype in ["textblock", "cell", "image", "caption", "table"]}

        for step, data in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP autocast
            with autocast('cuda') if self.scaler else nullcontext():
                logits = self.model(data)
                loss_dict = self.loss_fn(logits, data)
                loss = loss_dict["total"]

            # Backward pass with optional gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item()

            # Count positive/negative samples (handle batched data)
            for ntype in ["textblock", "cell", "image", "caption", "table"]:
                if hasattr(data[ntype], "y") and data[ntype].y is not None:
                    y = data[ntype].y
                    n_pos_total[ntype] += int(y.sum().item())
                    n_neg_total[ntype] += int((1 - y).sum().item())

            if (step + 1) % 100 == 0:
                logger.info(
                    f"  [Train] epoch={epoch} step={step+1} "
                    f"loss={total_loss/(step+1):.4f}"
                )

        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else step + 1
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(
            f"[Train] epoch={epoch} avg_loss={avg_loss:.4f} "
            f"pos/neg: text={n_pos_total['textblock']}/{n_neg_total['textblock']} "
            f"cell={n_pos_total['cell']}/{n_neg_total['cell']} "
            f"img={n_pos_total['image']}/{n_neg_total['image']} "
            f"caption={n_pos_total['caption']}/{n_neg_total['caption']} "
            f"table={n_pos_total['table']}/{n_neg_total['table']}"
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: Union[DataLoader, List[HeteroData], None]) -> Dict[str, float]:
        """Evaluate on validation set.

        Args:
            dataloader: DataLoader, list of HeteroData objects, or None

        Returns:
            Dictionary of evaluation metrics
        """
        if dataloader is None or (hasattr(dataloader, "__len__") and len(dataloader) == 0):
            return {}

        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        for data in dataloader:
            data = data.to(self.device, non_blocking=True)
            logits = self.model(data)
            loss_dict = self.loss_fn(logits, data)
            total_loss += loss_dict["total"].item()
            self.metrics.update(logits, data)

        result = self.metrics.compute()
        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else 1
        result["loss"] = total_loss / max(num_batches, 1)

        logger.info("[Eval] " + " | ".join(f"{k}={v:.4f}" for k, v in result.items()))
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, data: HeteroData, qid: str, node_ids_map: Optional[Dict] = None
    ) -> Dict:
        self.model.eval()
        data = data.to(self.device)
        logits = self.model(data)

        predictions = {"qid": qid, "scores": {}}
        for ntype, logit in logits.items():
            if logit.shape[0] == 0:
                continue
            scores = torch.sigmoid(logit).cpu().tolist()
            if hasattr(data[ntype], "node_ids"):
                node_ids = data[ntype].node_ids
            elif node_ids_map and ntype in node_ids_map:
                node_ids = node_ids_map[ntype]
            else:
                node_ids = list(range(len(scores)))
            predictions["scores"][ntype] = [
                {"node_id": nid, "score": s}
                for nid, s in zip(node_ids, scores)
            ]
            # Sort by score descending
            predictions["scores"][ntype].sort(key=lambda x: -x["score"])

        return predictions

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: Dict, resume_epoch: Optional[int] = None):
        """Save model checkpoint with training state.

        Args:
            epoch: Current epoch number (-1 for best checkpoint)
            metrics: Current evaluation metrics
            resume_epoch: Epoch to resume from (defaults to epoch)
        """
        ckpt_dir = self.output_cfg.get("checkpoints_dir", "outputs/checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        checkpoint_resume_epoch = epoch if resume_epoch is None else resume_epoch
        torch.save(
            {
                "epoch": epoch,
                "resume_epoch": checkpoint_resume_epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_metric_name": self._best_metric_name,
                "best_metric_value": self._best_metric_value,
                "best_metrics": self._best_metrics,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint and restore training state.

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state

        Returns:
            Resume epoch number (next epoch to train)
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

        # Restore best-checkpoint selection state
        best_metrics = ckpt.get("best_metrics", {})
        self._best_metrics = best_metrics if isinstance(best_metrics, dict) else {}
        # Legacy: best.pt may store metrics in "metrics" field
        if not self._best_metrics and ckpt.get("epoch") == -1:
            metrics = ckpt.get("metrics", {})
            if isinstance(metrics, dict):
                self._best_metrics = metrics

        # Select best metric from available metrics
        selected_metric = self._select_checkpoint_metric(self._best_metrics)
        if selected_metric is not None:
            self._best_metric_name, self._best_metric_value = selected_metric
        else:
            # Legacy: fallback to stored metric name/value
            self._best_metric_name = ckpt.get("best_metric_name")
            self._best_metric_value = ckpt.get("best_metric_value")
            # Synthesize best_metrics for legacy checkpoints
            if (
                not self._best_metrics
                and self._best_metric_name is not None
                and self._best_metric_value is not None
            ):
                self._best_metrics = {self._best_metric_name: self._best_metric_value}

        # If still no best-checkpoint state, try to recover from best.pt
        if not self._best_metrics:
            best_path = os.path.join(os.path.dirname(path), "best.pt")
            if os.path.abspath(path) != os.path.abspath(best_path) and os.path.exists(best_path):
                (
                    self._best_metric_name,
                    self._best_metric_value,
                    self._best_metrics,
                ) = self._load_checkpoint_selection_state(best_path)
                if self._best_metric_name is not None and self._best_metric_value is not None:
                    logger.info(
                        f"Recovered best checkpoint baseline from {best_path}: "
                        f"{self._best_metric_name}={self._best_metric_value:.4f}"
                    )
            elif load_optimizer:
                logger.warning(
                    f"Checkpoint {path} has no persisted best-checkpoint metadata; "
                    "best.pt selection will restart from the next evaluation."
                )

        # Restore optimizer state if requested
        if load_optimizer and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            logger.info(
                f"Checkpoint loaded: {path} (epoch={ckpt.get('epoch')}) with optimizer state"
            )
        else:
            logger.info(
                f"Checkpoint loaded: {path} (epoch={ckpt.get('epoch')}) - model only"
            )

        resume_epoch = ckpt.get("resume_epoch", ckpt.get("epoch", 0))
        return max(int(resume_epoch), 0)

    def _select_checkpoint_metric(self, metrics: Dict[str, float]):
        """Select best metric from available metrics (priority: MRR > F1 > loss)."""
        for metric_name in ["overall/mrr", "overall/f1"]:
            if metric_name in metrics:
                return metric_name, metrics[metric_name]
        if "loss" in metrics:
            return "loss", metrics["loss"]
        return None

    def _is_better_checkpoint(self, metric_name: str, candidate_value: float, best_value: Optional[float]) -> bool:
        """Check if candidate is better than current best (lower for loss, higher for others)."""
        if best_value is None:
            return True
        if metric_name == "loss":
            return candidate_value < best_value
        return candidate_value > best_value

    def _load_checkpoint_selection_state(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        best_metric_name = ckpt.get("best_metric_name")
        best_metric_value = ckpt.get("best_metric_value")

        best_metrics = ckpt.get("best_metrics", {})
        if not isinstance(best_metrics, dict):
            best_metrics = {}

        if not best_metrics and ckpt.get("epoch") == -1:
            metrics = ckpt.get("metrics", {})
            if isinstance(metrics, dict):
                best_metrics = metrics

        if best_metrics:
            selected_metric = self._select_checkpoint_metric(best_metrics)
            if selected_metric is not None:
                best_metric_name, best_metric_value = selected_metric
        elif best_metric_name is not None and best_metric_value is not None:
            best_metrics = {best_metric_name: best_metric_value}

        return best_metric_name, best_metric_value, best_metrics


    # ------------------------------------------------------------------
    # Full train loop
    # ------------------------------------------------------------------

    def run(
        self,
        train_dataloader: Union[DataLoader, List[HeteroData]],
        dev_dataloader: Union[DataLoader, List[HeteroData], None],
        start_epoch: int = 1,
    ):
        """Run full training loop.

        Args:
            train_dataloader: Training DataLoader or list of HeteroData
            dev_dataloader: Validation DataLoader, list of HeteroData, or None
            start_epoch: Starting epoch number
        """
        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("epochs", 10)
        save_every = self.output_cfg.get("save_every", 1)
        best_metric_name = self._best_metric_name
        best_metric_value = self._best_metric_value

        train_size = len(train_dataloader) if hasattr(train_dataloader, "__len__") else "unknown"
        dev_size = len(dev_dataloader) if dev_dataloader and hasattr(dev_dataloader, "__len__") else 0

        logger.info(
            f"Starting training: {epochs} epochs, "
            f"{train_size} train batches, "
            f"{dev_size} dev batches, "
            f"start_epoch={start_epoch}"
        )

        if dev_dataloader and best_metric_name is not None and best_metric_value is not None:
            logger.info(
                f"Starting best checkpoint baseline: {best_metric_name}={best_metric_value:.4f}"
            )

        for epoch in range(start_epoch, epochs + 1):
            self.train_epoch(train_dataloader, epoch)
            metrics = self.evaluate(dev_dataloader) if dev_dataloader else {}
            if not metrics:
                logger.info(f"[Eval] skipped: no dev data available for epoch={epoch}")

            selected_metric = self._select_checkpoint_metric(metrics)
            if selected_metric is not None:
                metric_name, metric_value = selected_metric
                baseline_value = self._best_metrics.get(metric_name)
                should_update_best = False

                if best_metric_name is None:
                    should_update_best = True
                elif baseline_value is None:
                    should_update_best = True
                    logger.info(
                        f"Switching best-checkpoint metric to {metric_name} because the existing best checkpoint has no comparable {metric_name}."
                    )
                elif self._is_better_checkpoint(metric_name, metric_value, baseline_value):
                    should_update_best = True

                if should_update_best:
                    best_metric_name = metric_name
                    best_metric_value = metric_value
                    self._best_metric_name = best_metric_name
                    self._best_metric_value = best_metric_value
                    self._best_metrics = dict(metrics)
                    self.save_checkpoint(-1, metrics, resume_epoch=epoch)  # best model
                    best_path = os.path.join(
                        self.output_cfg.get("checkpoints_dir", "outputs/checkpoints"),
                        "best.pt",
                    )
                    # rename last saved to best.pt
                    last = os.path.join(
                        self.output_cfg.get("checkpoints_dir", "outputs/checkpoints"),
                        "epoch_-01.pt",
                    )
                    if os.path.exists(last):
                        os.replace(last, best_path)
                    logger.info(
                        f"New best model ({best_metric_name}={best_metric_value:.4f}): {best_path}"
                    )

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, metrics)

        if not dev_dataloader:
            best_path = os.path.join(
                self.output_cfg.get("checkpoints_dir", "outputs/checkpoints"),
                "best.pt",
            )
            self.save_checkpoint(-1, {}, resume_epoch=epochs)
            last = os.path.join(
                self.output_cfg.get("checkpoints_dir", "outputs/checkpoints"),
                "epoch_-01.pt",
            )
            if os.path.exists(last):
                os.replace(last, best_path)
            logger.info(f"Saved final model as best checkpoint (no dev set): {best_path}")

        logger.info("Training complete.")
