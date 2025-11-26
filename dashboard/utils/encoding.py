"""Encoding utilities for CLIP text encoding and image base64 conversion."""

import base64
from io import BytesIO
from pathlib import Path
import logging

import numpy as np
from PIL import Image

from ..config import CLIP_MODEL, CLIP_PRETRAINED, MAX_IMAGE_WIDTH

logger = logging.getLogger(__name__)

# Cache for CLIP model (loaded on first text search)
_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None


def load_clip_for_text(model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
    """Load CLIP model for text encoding (cached globally).

    Args:
        model_name: CLIP model architecture
        pretrained: Pre-trained weights

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_DEVICE

    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_DEVICE

    try:
        import open_clip
        import torch
    except ImportError:
        raise ImportError(
            "open_clip_torch is required for text search. "
            "Install with: pip install streettransformer[dashboard]"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP model {model_name} on {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    _CLIP_MODEL = model
    _CLIP_TOKENIZER = tokenizer
    _CLIP_DEVICE = device

    return model, tokenizer, device


def encode_text_query(text: str) -> np.ndarray:
    """Encode text query to embedding vector.

    Args:
        text: Text query

    Returns:
        Normalized embedding vector
    """
    import torch

    model, tokenizer, device = load_clip_for_text()

    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        embedding_np = text_embedding.cpu().numpy()[0]

    return embedding_np


def encode_image_to_base64(image_path: Path, max_width: int = MAX_IMAGE_WIDTH) -> str:
    """Convert image to base64 string for embedding in HTML.

    Args:
        image_path: Path to image file
        max_width: Maximum width for resizing

    Returns:
        Base64 encoded image string or None if error
    """
    try:
        if not Path(image_path).exists():
            return None

        img = Image.open(image_path)

        # Resize if too large
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None
