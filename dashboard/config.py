"""Dashboard configuration and styling."""
from dataclasses import dataclass
from typing import Optional

# Dark mode color scheme
COLORS = {
    'background': '#1e1e1e',
    'card': '#252526',
    'border': '#3e3e42',
    'text': '#cccccc',
    'text-secondary': '#8e8e93',
    'primary': '#0e639c',
    'primary-hover': '#1177bb',
    'success': '#4ec9b0',
    'warning': '#ce9178',
    'error': '#f48771',
    'input-bg': '#3c3c3c',
}

# Default settings
DEFAULT_LIMIT = 20
MAX_IMAGE_WIDTH = 400
DEFAULT_PORT = 8050

# CLIP model settings
CLIP_MODEL = 'ViT-B-32'
CLIP_PRETRAINED = 'openai'

# PostgreSQL Configuration
@dataclass
class PGConfig:
    """PostgreSQL connection configuration."""
    db_name: str = "image_retrieval"
    db_user: str = "postgres"
    db_password: str = ""
    db_host: str = "localhost"
    db_port: int = 5432
    min_connections: int = 1
    max_connections: int = 5
    vector_dimension: int = 512  # Default CLIP ViT-B-32 dimension
    # Field mapping: PostgreSQL field name that maps to DuckDB's location_id
    pg_location_id_field: str = "location_key"  # Currently location_key in PG = location_id in DuckDB
    
    

# Feature Flags
@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout and A/B testing."""
    use_vector_search: bool = False  # Enable PostgreSQL vector search
    use_whitening: bool = False      # Enable whitening transform
    use_faiss_fallback: bool = True  # Fallback to FAISS if PostgreSQL fails
    log_search_metrics: bool = True  # Log latency/performance metrics
    encoder_selection: bool = False  # Allow user to select encoder (CLIP/BLIP/etc)
