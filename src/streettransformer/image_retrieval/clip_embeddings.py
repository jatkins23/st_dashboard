from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

import open_clip  # pip/uv: open-clip-torch

try:  # optional dependency for offline cache checks
    from huggingface_hub import hf_hub_download  # type: ignore
    try:  # huggingface_hub>=0.20 moves the error into utils
        from huggingface_hub.utils import LocalEntryNotFoundError  # type: ignore
    except ImportError:  # pragma: no cover - older versions keep it at top-level
        from huggingface_hub import LocalEntryNotFoundError  # type: ignore
except Exception:  # pragma: no cover - optional
    hf_hub_download = None  # type: ignore
    LocalEntryNotFoundError = FileNotFoundError  # type: ignore

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Device selection                                                            #
# --------------------------------------------------------------------------- #

def pick_device(user: str | None = None) -> str:
    """
    Choose a compute device with sensible fallbacks.
    Priority: explicit arg/env -> MPS (Apple) -> CUDA -> CPU.
    """
    user = (user or os.getenv("DEVICE") or "").strip().lower()
    if user in {"cpu", "mps", "cuda"}:
        if user == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if user == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return user
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Core CLIP embedder                                                          #
# --------------------------------------------------------------------------- #

class CLIPEmbedder:
    """
    Minimal OpenCLIP wrapper for image and text embeddings.

    Defaults:
      - model_name: ViT-B-32
      - pretrained: laion2b_s34b_b79k  (stable cosine behaviour)
      - outputs: L2-normalised np.float32 vectors
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: str | Path | None = None,
        auto_download: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = pick_device(device)
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.auto_download = bool(auto_download)

        if not self._prepare_model_assets():
            raise RuntimeError(
                "OpenCLIP weights missing and auto_download is disabled. "
                "Run 'python scripts/cache_openclip.py' or set OPENCLIP_AUTO_DOWNLOAD=1."
            )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval().to(self.device)
        self.tokenize = open_clip.tokenize
        self.batch_size = max(1, int(batch_size))

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

    def _load_batch(self, paths: Sequence[Path]) -> torch.Tensor:
        imgs: List[torch.Tensor] = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    imgs.append(self.preprocess(im.convert("RGB")))
            except (UnidentifiedImageError, OSError) as exc:
                print(f"[warn] skipping unreadable image: {p} ({exc})")
        if not imgs:
            return torch.empty((0, 3, 224, 224))
        return torch.stack(imgs, dim=0)

    @torch.inference_mode()
    def encode_images(self, tensors: torch.Tensor) -> np.ndarray:
        feats = self.model.encode_image(tensors.to(self.device))
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
        return feats.cpu().numpy().astype("float32", copy=False)

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
            batch = self._load_batch(batch_paths)
            if batch.shape[0] == 0:
                continue
            embs = self.encode_images(batch)
            for idx, path in enumerate(batch_paths[: embs.shape[0]]):
                results.append((str(path), embs[idx]))
            done = min(start + bs, total)
            if done == total or done % progress_every == 0:
                print(f"...processed {done}/{total}")
        return results

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        batch = self._load_batch([Path(image_path)])
        if batch.shape[0] != 1:
            raise RuntimeError(f"could not load image: {image_path}")
        return self.encode_images(batch)[0]

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype="float32")
        tensors = []
        for img in images:
            tensors.append(self.preprocess(img.convert("RGB")))
        batch = torch.stack(tensors, dim=0)
        return self.encode_images(batch)

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenize([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
        return feats.cpu().numpy().astype("float32", copy=False).ravel()

    # ---------------------------------------------------------------- internal
    def _resolve_cache_dir(self, cache_dir: Optional[str | Path]) -> Path:
        if cache_dir is not None:
            return Path(cache_dir).expanduser().resolve()
        env = os.getenv("OPENCLIP_HOME") or os.getenv("OPENCLIP_CACHE")
        if env:
            return Path(env).expanduser().resolve()
        default = Path(__file__).resolve().parent / "artifacts" / "model_cache"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _prepare_model_assets(self) -> bool:
        os.environ.setdefault("OPENCLIP_HOME", str(self.cache_dir))
        os.environ.setdefault("OPENCLIP_CACHE", str(self.cache_dir))
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir / "hf"))

        repo_id = _resolve_hf_repo(self.model_name, self.pretrained)
        if repo_id is None:
            LOGGER.debug(
                "No known HuggingFace repo mapping for %s/%s; skipping pre-download.",
                self.model_name,
                self.pretrained,
            )
            return True

        required_files = (
            "open_clip_model.safetensors",
            "open_clip_pytorch_model.bin",
            "tokenizer.json",
        )
        files = required_files
        local_cache = self.cache_dir / "hf"
        local_cache.mkdir(parents=True, exist_ok=True)

        missing: List[str] = []
        for fname in files:
            if _has_cached_file(local_cache, fname):
                continue
            if hf_hub_download is not None:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        cache_dir=str(local_cache),
                        local_dir=str(local_cache),
                        local_dir_use_symlinks=False,
                        local_files_only=True,
                    )
                    continue
                except LocalEntryNotFoundError:
                    pass
            missing.append(fname)

        if not missing:
            LOGGER.debug("OpenCLIP weights already cached in %s", local_cache)
            return True

        if not self.auto_download:
            LOGGER.warning(
                "OpenCLIP weights missing (%s).",
                ", ".join(missing),
            )
            return False

        if hf_hub_download is None:
            LOGGER.warning(
                "huggingface_hub not available; cannot download OpenCLIP weights automatically.")
            return False

        missing_required: List[str] = []
        for fname in missing:
            try:
                print(f"Downloading OpenCLIP weight '{fname}' from {repo_id} ...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    cache_dir=str(local_cache),
                    local_dir=str(local_cache),
                    local_dir_use_symlinks=False,
                    force_download=False,
                )
                print(f"✓ Cached '{fname}'")
            except Exception as exc:  # pragma: no cover - network failures
                LOGGER.warning(
                    "Failed to download %s from %s (%s)",
                    fname,
                    repo_id,
                    exc,
                )
                missing_required.append(fname)
        if missing_required:
            LOGGER.warning(
                "OpenCLIP required weights missing after download attempt: %s",
                ", ".join(missing_required),
            )
            return False
        return True


def _resolve_hf_repo(model_name: str, pretrained: str) -> Optional[str]:
    """
    Map a (model, pretrained) pair to the HuggingFace repo that open_clip uses.
    Extend as new checkpoints are needed.
    """
    key = (model_name.strip(), pretrained.strip())
    mapping = {
        ("ViT-B-32", "laion2b_s34b_b79k"): "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        ("ViT-L-14", "laion2b_s32b_b82k"): "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ("ViT-H-14", "laion2b_s32b_b79k"): "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    }
    return mapping.get(key)


# --------------------------------------------------------------------------- #
# Script usage                                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Embed images and/or text using OpenCLIP.")
    parser.add_argument("--folder", type=str, help="Folder of images to embed (recursive).")
    parser.add_argument("--image", type=str, help="Single image path to embed.")
    parser.add_argument("--text", type=str, help="Optional text to embed.")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-npz", type=str, help="Optional npz output for folder embeddings.")
    args = parser.parse_args()

    embedder = CLIPEmbedder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )

    if args.folder:
        t0 = time.perf_counter()
        pairs = embedder.embed_folder(args.folder)
        dt = time.perf_counter() - t0
        print(f"embedded {len(pairs)} images in {dt:.2f}s")
        if args.save_npz:
            paths = np.array([p for p, _ in pairs], dtype=object)
            vecs = np.stack([v for _, v in pairs], axis=0).astype("float32", copy=False)
            np.savez_compressed(args.save_npz, paths=paths, vectors=vecs)
            print(f"saved npz: {args.save_npz}")

    if args.image:
        vec = embedder.embed_image(args.image)
        print(f"image embedding shape: {vec.shape}, norm={np.linalg.norm(vec):.6f}")

    if args.text:
        vec = embedder.embed_text(args.text)
        print(f"text embedding shape: {vec.shape}, norm={np.linalg.norm(vec):.6f}")
def _has_cached_file(local_cache: Path, filename: str) -> bool:
    try:
        next(local_cache.rglob(filename))
        return True
    except StopIteration:
        return False
