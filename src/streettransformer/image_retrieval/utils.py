from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Tuple

_YEAR_RE = re.compile(r"(^|[/_.-])(20\d{2}|19\d{2})(?=$|[/_.-])")

def extract_year_from_path(path: str) -> Optional[int]:
    """
    Extract a 4-digit year token from a path segment like '/.../2018/...'.
    Returns None if absent.
    """
    m = _YEAR_RE.search(path)
    if not m:
        return None
    y = int(m.group(2))
    # Constrain to a plausible range to avoid false positives.
    if 1990 <= y <= 2100:
        return y
    return None

def split_at_year(path: str) -> Tuple[Optional[int], str]:
    """
    Return (year, suffix_after_year). If year not found -> (None, '').
    Suffix is the part after the year segment, used to derive a stable 'location_id'.
    """
    m = _YEAR_RE.search(path)
    if not m:
        return None, ""
    year = int(m.group(2))
    return year, path[m.end():].lstrip("/")

def location_id_from_path(path: str) -> str:
    """
    Stable location id: the portion of the path AFTER the year segment.
    Example:
      '/root/2014/boro/A/B.png' -> 'boro/A/B.png'
    If no year is present, fall back to the final 2 path components.
    """
    year, suffix = split_at_year(path)
    if year and suffix:
        return suffix
    p = Path(path)
    parts = p.parts[-2:] if len(p.parts) >= 2 else p.parts
    return "/".join(parts)
