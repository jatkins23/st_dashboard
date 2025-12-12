"""Lazy-loading singleton manager for CLIP/BLIP encoders.

The CLIP model is ~400MB and takes 10-30s to load, so we only load it when needed.
This manager implements a singleton pattern to ensure only one encoder is loaded at a time.
"""
import logging
from typing import Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class EncoderManager:
    """Singleton manager for encoder instances with lazy loading.

    Usage:
        manager = EncoderManager()
        embedding = manager.encode_image(image_path)
        text_embedding = manager.encode_text("a street scene")
    """

    _instance = None
    _encoder = None
    _encoder_type = None

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_clip_encoder(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        """Lazy load CLIP encoder.

        Args:
            model_name: CLIP model name (default: ViT-B-32)
            pretrained: Pretrained weights (default: openai)
            device: Device to use (cpu/cuda/mps, default: auto-detect)
        """
        if self._encoder is not None and self._encoder_type == 'clip':
            logger.debug("CLIP encoder already loaded, reusing")
            return self._encoder

        try:
            from streettransformer.image_retrieval.clip_embeddings import CLIPEmbedder

            logger.info(f"Loading CLIP encoder ({model_name}, {pretrained})...")
            self._encoder = CLIPEmbedder(
                model_name=model_name,
                pretrained=pretrained,
                device=device
            )
            self._encoder_type = 'clip'
            logger.info("CLIP encoder loaded successfully")
            return self._encoder

        except ImportError as e:
            logger.error(f"Failed to import CLIPEmbedder: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP encoder: {e}")
            raise

    def encode_image(
        self,
        image_path: str,
        encoder_type: str = 'clip',
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None
    ) -> np.ndarray:
        """Encode a single image to embedding vector.

        Args:
            image_path: Path to image file
            encoder_type: Type of encoder ('clip', 'blip', etc.)
            model_name: Model name for encoder
            pretrained: Pretrained weights
            device: Device to use

        Returns:
            numpy array embedding vector (float32, normalized)
        """
        if encoder_type != 'clip':
            raise NotImplementedError(f"Encoder type '{encoder_type}' not yet supported. Use 'clip'.")

        encoder = self._load_clip_encoder(model_name, pretrained, device)
        return encoder.embed_image(image_path)

    def encode_text(
        self,
        text: str,
        encoder_type: str = 'clip',
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None
    ) -> np.ndarray:
        """Encode text to embedding vector.

        Args:
            text: Text string to encode
            encoder_type: Type of encoder ('clip', 'blip', etc.)
            model_name: Model name for encoder
            pretrained: Pretrained weights
            device: Device to use

        Returns:
            numpy array embedding vector (float32, normalized)
        """
        if encoder_type != 'clip':
            raise NotImplementedError(f"Encoder type '{encoder_type}' not yet supported. Use 'clip'.")

        encoder = self._load_clip_encoder(model_name, pretrained, device)
        return encoder.encode_text(text)

    def unload_encoder(self):
        """Unload encoder to free memory.

        Call this when encoder is no longer needed to free up ~400MB RAM.
        """
        if self._encoder is not None:
            logger.info(f"Unloading {self._encoder_type} encoder to free memory")
            self._encoder = None
            self._encoder_type = None

    @property
    def is_loaded(self) -> bool:
        """Check if an encoder is currently loaded."""
        return self._encoder is not None

    @property
    def encoder_type(self) -> Optional[str]:
        """Get type of currently loaded encoder."""
        return self._encoder_type
