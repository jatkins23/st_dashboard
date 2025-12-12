from __future__ import annotations

import hashlib
from pathlib import Path


def strip_year_segment(path: str) -> str:
    """
    Remove the first 4-digit year segment from a path.
    Example: '.../2006/4831.png' -> '.../4831.png'
    """
    parts = Path(path).parts
    out: list[str] = []
    removed = False
    for part in parts:
        if not removed and len(part) == 4 and part.isdigit():
            removed = True
            continue
        out.append(part)
    return str(Path(*out)) if out else path


def location_key(path: str) -> str:
    """
    Immutable key used to group a physical site across years.
    We strip the year folder, keep any remaining subfolders,
    and drop the image extension so that '/root/2012/queens/4831.png'
    becomes 'queens/4831'.
    """
    # stripped = strip_year_segment(path)
    # p = Path(stripped)
    # parts = p.parts
    # if not parts:
    #     return p.stem or stripped
    # stemmed = list(parts[:-1])
    # stemmed.append(Path(parts[-1]).stem)
    # return str(Path(*stemmed))
    return Path(path).stem


def location_hash(loc_key: str, *, digest_size: int = 8) -> int:
    """
    Deterministic BIGINT-safe hash for the location key. Used when we need a numeric id.
    """
    h = hashlib.blake2b(loc_key.encode("utf-8"), digest_size=digest_size)
    value = int.from_bytes(h.digest(), byteorder="big", signed=False)
    return value & ((1 << 63) - 1)
