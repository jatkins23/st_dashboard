"""CLIP text encoding for text-to-image search.

This module provides text encoding using OpenCLIP models.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """CLIP text encoder for converting text queries to embeddings.

    Loads and caches the CLIP model for efficient text encoding.

    Example:
        >>> encoder = CLIPEncoder()  # Load once at startup
        >>> embedding = encoder.encode("street with trees")
        >>> embedding.shape
        (512,)
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """Initialize CLIP encoder.

        Args:
            model_name: CLIP model architecture (default: ViT-B-32)
            pretrained: Pre-trained weights (default: openai)
                Options: 'openai' (standard), 'laion2b_s34b_b79k', 'laion400m_e32'
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """Load CLIP model and tokenizer."""
        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for text search. "
                "Install with: pip install open-clip-torch"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {self.model_name} on {self.device}")

        # Use force_quick_gelu for OpenAI weights to avoid architecture mismatch warning
        force_quick_gelu = (self.pretrained == "openai")

        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            force_quick_gelu=force_quick_gelu
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"CLIP model {self.model_name} loaded successfully")

    def encode(self, text: str) -> np.ndarray:
        """Encode text query to embedding vector.

        Args:
            text: Text query

        Returns:
            Normalized embedding vector as numpy array

        Example:
            >>> encoder = CLIPEncoder()
            >>> embedding = encoder.encode("street with trees")
            >>> embedding.shape
            (512,)
        """
        import torch

        if self.model is None:
            raise RuntimeError("CLIP model not loaded. Call _load_model() first.")

        text_tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            embedding_np = text_embedding.cpu().numpy()[0]

        return embedding_np
