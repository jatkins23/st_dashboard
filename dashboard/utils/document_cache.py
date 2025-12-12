"""Disk cache for converted PDF document images."""
import hashlib
import logging
from pathlib import Path
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentImgCache:
    """Disk-based cache for PDF-to-image conversions.

    Stores base64-encoded images in temp directory to avoid reprocessing PDFs.
    Cache is invalidated if source PDF is modified.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files. Defaults to system temp dir.
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "st_dashboard_pdf_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentImgCache initialized at: {self.cache_dir}")

    def _get_cache_key(self, pdf_path: Path, max_width: int) -> str:
        """Generate cache key from PDF path and max_width.

        Args:
            pdf_path: Path to PDF file
            max_width: Maximum width for rendering

        Returns:
            MD5 hash as cache key
        """
        cache_str = f"{pdf_path.absolute()}_{max_width}".encode('utf-8')
        return hashlib.md5(cache_str).hexdigest()

    def get(self, pdf_path: Path, max_width: int) -> Optional[str]:
        """Retrieve cached base64 image if available and fresh.

        Args:
            pdf_path: Path to PDF file
            max_width: Maximum width used for rendering

        Returns:
            Base64-encoded image string if cached and fresh, None otherwise
        """
        if not pdf_path.exists():
            return None

        cache_key = self._get_cache_key(pdf_path, max_width)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if not cache_file.exists():
            return None

        # Check if cache is stale (older than source PDF)
        try:
            cache_mtime = cache_file.stat().st_mtime
            pdf_mtime = pdf_path.stat().st_mtime

            if cache_mtime < pdf_mtime:
                logger.debug(f"Cache stale for {pdf_path.name}, will regenerate")
                return None

            # Cache is fresh, return it
            result = cache_file.read_text()
            logger.debug(f"Cache hit: {pdf_path.name}")
            return result

        except Exception as e:
            logger.warning(f"Error reading cache for {pdf_path.name}: {e}")
            return None

    def set(self, pdf_path: Path, max_width: int, base64_data: str) -> bool:
        """Store base64 image in cache.

        Args:
            pdf_path: Path to PDF file
            max_width: Maximum width used for rendering
            base64_data: Base64-encoded image string to cache

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            cache_key = self._get_cache_key(pdf_path, max_width)
            cache_file = self.cache_dir / f"{cache_key}.txt"

            cache_file.write_text(base64_data)
            logger.debug(f"Cached: {pdf_path.name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache {pdf_path.name}: {e}")
            return False

    def clear(self) -> int:
        """Clear all cached images.

        Returns:
            Number of files deleted
        """
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.txt"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cached PDF images")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return count

    def get_cache_size(self) -> int:
        """Get total size of cache in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        try:
            for cache_file in self.cache_dir.glob("*.txt"):
                total_size += cache_file.stat().st_size
            return total_size
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0
