from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipModel, BlipProcessor

from .clip_embeddings import pick_device


class BLIPEmbedder:
    """
    Minimal BLIP wrapper for image/text embeddings.

    Defaults:
      - model_name: Salesforce/blip-itm-base-coco
      - outputs: L2-normalised np.float32 vectors
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        device: str | None = None,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = self._resolve_cache_dir(cache_dir)

        self.processor = BlipProcessor.from_pretrained(self.model_name, cache_dir=str(self.cache_dir))
        self.model = BlipModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir))
        self.model.eval().to(self.device)
        self.embedding_dim = int(
            getattr(self.model.config.vision_config, "hidden_size", None)
            or getattr(self.model.config.text_config, "hidden_size", 768)
        )

    # ------------------------------------------------------------------ helpers
    def _iter_image_paths(
        self,
        root: Path,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> List[Path]:
        exts = {e.lower() for e in extensions} | {e.upper() for e in extensions}
        paths: List[Path] = []
        for ext in exts:
            paths.extend(root.rglob(f"*{ext}"))
        return sorted({p for p in paths}, key=lambda p: str(p).lower())

    def _load_images(self, paths: Sequence[Path]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                print(f"[warn] skipping unreadable image: {p} ({exc})")
        return images

    def _resolve_cache_dir(self, cache_dir: str | Path | None) -> Path:
        if cache_dir is not None:
            out = Path(cache_dir).expanduser().resolve()
        else:
            base = Path(__file__).resolve().parent / "artifacts" / "model_cache" / "blip"
            out = base
        out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(out))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(out / "hf"))
        return out

    # ------------------------------------------------------------------- public
    def embed_folder(
        self,
        folder: str | Path,
        *,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        progress_every: int = 200,
    ) -> List[Tuple[str, np.ndarray]]:
        root = Path(folder)
        if not root.exists():
            raise FileNotFoundError(f"image folder not found: {root}")

        paths = self._iter_image_paths(root, extensions)
        if not paths:
            print(f"No images found under {root}")
            return []

        results: List[Tuple[str, np.ndarray]] = []
        bs = self.batch_size
        total = len(paths)

        for start in range(0, total, bs):
            batch_paths = paths[start : start + bs]
            imgs = self._load_images(batch_paths)
            if not imgs:
                continue
            embs = self.embed_images(imgs)
            for idx, path in enumerate(batch_paths[: embs.shape[0]]):
                results.append((str(path), embs[idx]))
            done = min(start + bs, total)
            if done == total or done % progress_every == 0:
                print(f"...processed {done}/{total}")
        return results

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype="float32")
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        cls = vision_out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
        return cls.cpu().numpy().astype("float32", copy=False)

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.embed_images(images)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        imgs = self._load_images([Path(image_path)])
        if len(imgs) != 1:
            raise RuntimeError(f"could not load image: {image_path}")
        return self.embed_images(imgs)[0]

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        encoder = getattr(self.model, "text_encoder", None) or getattr(self.model, "text_model", None)
        if encoder is None:
            raise RuntimeError("BLIP text encoder not found on model (expected text_encoder or text_model).")
        text_out = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        cls = text_out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
        return cls.cpu().numpy().astype("float32", copy=False).ravel()


__all__ = ["BLIPEmbedder"]
