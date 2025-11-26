"""Dashboard configuration and styling."""

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
