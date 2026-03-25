"""Qwen3-VL-Embedding-8B encoder with automatic fallback.

Priority:
  1. Qwen/Qwen3-VL-Embedding-8B  (transformers + torchvision required)
  2. sentence-transformers (text) + title-text fallback (image)

Usage:
    encoder = QwenVLFeatureEncoder.from_config(cfg)
    text_embs = encoder.encode_texts(["hello world", "foo bar"])
    img_embs  = encoder.encode_images(["path/to/img.jpg"])
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = "/model/ModelScope/Qwen/Qwen3-VL-Embedding-8B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_from_str(dtype_str: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        dtype_str, torch.float32
    )


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_exp = mask.unsqueeze(-1).float()
    return (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseEncoder:
    hidden_dim: int = 384

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_image_pairs(
        self, texts: List[str], image_paths: List[str]
    ) -> torch.Tensor:
        t = self.encode_texts(texts)
        i = self.encode_images(image_paths)
        return (t + i) / 2.0


# ---------------------------------------------------------------------------
# Qwen3-VL-Embedding-8B encoder
# ---------------------------------------------------------------------------

class QwenVLEncoder(BaseEncoder):
    """Wraps Qwen3-VL-Embedding-8B using the official processor + chat template API."""

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype_str: str = "bf16",
        batch_size: int = 8,
        max_text_length: int = 512,
    ):
        from transformers import AutoTokenizer, AutoModel, AutoProcessor

        self._model_path = model_name_or_path
        self._device = device
        self._dtype = _dtype_from_str(dtype_str)
        self._batch_size = batch_size
        self._max_length = max_text_length

        logger.info(f"Loading Qwen3-VL model from: {model_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=self._dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # hidden_dim from text_config
        try:
            self.hidden_dim = self.model.config.text_config.hidden_size
        except Exception:
            self.hidden_dim = 4096
        logger.info(
            f"Qwen3-VL encoder ready. dim={self.hidden_dim}, "
            f"dtype={dtype_str}, device={device}"
        )

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.hidden_dim)
        all_embs = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            batch = [t if t.strip() else "[empty]" for t in batch]

            # Build messages using chat template
            prompts = []
            for t in batch:
                messages = [{"role": "user", "content": [{"type": "text", "text": t}]}]
                prompts.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )

            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)

            outputs = self.model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images using Qwen3-VL vision encoder."""
        if not image_paths:
            return torch.zeros(0, self.hidden_dim)

        from PIL import Image as PILImage

        all_embs = []
        for i in range(0, len(image_paths), self._batch_size):
            batch_paths = image_paths[i: i + self._batch_size]
            batch_embs = []
            for path in batch_paths:
                if not path or not os.path.exists(path):
                    logger.debug(f"Image not found: {path}, will use text fallback upstream.")
                    batch_embs.append(torch.zeros(self.hidden_dim))
                    continue
                try:
                    img = PILImage.open(path)
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    img = img.convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": ""},
                            ],
                        }
                    ]
                    text_prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text_prompt],
                        images=[img],
                        return_tensors="pt",
                        padding=True,
                    ).to(self._device)

                    outputs = self.model(**inputs)
                    emb = _mean_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    emb = F.normalize(emb, dim=-1).squeeze(0).cpu().float()
                    batch_embs.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to encode image {path}: {e}. Using zero vector.")
                    batch_embs.append(torch.zeros(self.hidden_dim))
            all_embs.append(torch.stack(batch_embs))
        return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Fallback: sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.hidden_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"[Fallback] SentenceTransformer: {model_name} (dim={self.hidden_dim})"
        )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.hidden_dim)
        texts = [t if t.strip() else "[empty]" for t in texts]
        return self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        ).cpu().float()

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        # No visual encoding in fallback — feature_builder handles title-text fallback
        return torch.zeros(len(image_paths), self.hidden_dim)


# ---------------------------------------------------------------------------
# Last-resort random encoder (unit tests / CI without GPU)
# ---------------------------------------------------------------------------

class _RandomEncoder(BaseEncoder):
    def __init__(self, dim: int = 384):
        self.hidden_dim = dim
        logger.warning(f"[ENCODER] Using RandomEncoder (dim={dim}). Outputs are meaningless.")

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return F.normalize(torch.randn(len(texts), self.hidden_dim), dim=-1)

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        return F.normalize(torch.randn(len(image_paths), self.hidden_dim), dim=-1)


# ---------------------------------------------------------------------------
# Public factory: QwenVLFeatureEncoder
# ---------------------------------------------------------------------------

class QwenVLFeatureEncoder:
    """Public API. Auto-selects Qwen or fallback based on environment."""

    def __init__(self, encoder: BaseEncoder):
        self._enc = encoder

    @property
    def hidden_dim(self) -> int:
        return self._enc.hidden_dim

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return self._enc.encode_texts(texts)

    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        return self._enc.encode_images(image_paths)

    def encode_text_image_pairs(
        self, texts: List[str], image_paths: List[str]
    ) -> torch.Tensor:
        return self._enc.encode_text_image_pairs(texts, image_paths)

    @classmethod
    def from_config(cls, cfg: dict) -> "QwenVLFeatureEncoder":
        enc_name = cfg.get("name", "qwen3_vl_embedding_8b")
        device = cfg.get("device", "cuda")
        dtype_str = cfg.get("dtype", "bf16")
        batch_size = cfg.get("batch_size", 8)
        max_text_length = cfg.get("max_text_length", 512)

        if enc_name == "qwen3_vl_embedding_8b":
            try:
                enc = QwenVLEncoder(
                    model_name_or_path=cfg.get("model_name_or_path", _DEFAULT_MODEL_PATH),
                    device=device,
                    dtype_str=dtype_str,
                    batch_size=batch_size,
                    max_text_length=max_text_length,
                )
                return cls(enc)
            except Exception as e:
                logger.warning(
                    f"[ENCODER] 当前未成功加载 Qwen3-VL-Embedding-8B: {e}\n"
                    f"[ENCODER] 已切换到备用编码器: sentence-transformers"
                )

        # fallback
        fallback_model = cfg.get(
            "fallback_text_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            enc = SentenceTransformerEncoder(fallback_model, device="cpu")
        except Exception as e2:
            logger.warning(f"[ENCODER] sentence-transformers 也加载失败: {e2}. 使用 RandomEncoder。")
            enc = _RandomEncoder(cfg.get("fallback_hidden_dim", 384))
        return cls(enc)
