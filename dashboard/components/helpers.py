"""Shared UI component helpers.

These are standalone rendering functions that can be used by any part of the
application to build consistent UI components.
"""

# TODO: Are all of these just part of Results?

from pathlib import Path
from dash import html
from ..config import COLORS

from ..utils import encode_image_to_base64

def get_similarity_badge(similarity: float):
    """Get styled similarity badge.

    Args:
        similarity: Similarity score (0-1)

    Returns:
        Dash HTML Span with colored badge
    """
    # Color based on similarity threshold
    if similarity >= 0.9:
        color = COLORS['success']
    elif similarity >= 0.7:
        color = COLORS['warning']
    else:
        color = COLORS['text-secondary']

    return html.Span(
        f"{similarity:.3f}",
        style={
            'backgroundColor': color,
            'color': 'white',
            'padding': '4px 8px',
            'borderRadius': '4px',
            'fontSize': 12,
            'fontWeight': 'bold'
        }
    )


def render_image(image_path_or_str, max_width: int = 300):
    """Render image from path.

    Args:
        image_path_or_str: Path to image (Path or str)
        max_width: Maximum width in pixels

    Returns:
        Dash HTML component (Img or error Div)
    """

    if not image_path_or_str:
        return html.Div("No image available",
                      style={'fontStyle': 'italic', 'color': COLORS['text-secondary']})

    image_path = Path(image_path_or_str)
    if not image_path.exists():
        return html.Div(f"Image not found: {image_path.name}",
                      style={'color': COLORS['warning'], 'fontSize': 12})

    img_base64 = encode_image_to_base64(image_path, max_width=max_width)
    if img_base64:
        return html.Img(src=img_base64,
                      style={'maxWidth': '100%', 'borderRadius': '4px',
                            'border': f"1px solid {COLORS['border']}"})

    return html.Div("Error loading image",
                  style={'color': COLORS['error'], 'fontSize': 12})


def render_detail_row(label: str, value: str):
    """Render label-value detail row.

    Args:
        label: Label text
        value: Value text

    Returns:
        Dash HTML Div with formatted detail row
    """
    return html.Div([
        html.Span(f"{label} ",
                 style={'color': COLORS['text-secondary'], 'fontWeight': '500'}),
        html.Span(value,
                 style={'color': COLORS['text']})
    ], style={'marginBottom': 6, 'fontSize': 13})


def render_accordion_header(index: int, location_id: str, similarity: float,
                            street_info: str = None):
    """Render accordion header with consistent styling.

    Args:
        index: Result index (1-based)
        location_id: Location ID
        similarity: Similarity score
        street_info: Optional street name(s)

    Returns:
        Dash HTML Div with accordion header
    """
    components = [
        get_similarity_badge(similarity),
        html.Span(f"#{index}: ", style={'marginLeft': 10, 'fontWeight': 'bold'}),
        html.Span(f"Location {location_id}")
    ]

    if street_info:
        components.append(
            html.Span(f" - {street_info}",
                     style={'marginLeft': 5, 'color': COLORS['text-secondary']})
        )

    return html.Div(components, className='accordion-header')


def render_image_pair(image_path_from, image_path_to, year_from: int, year_to: int,
                     max_width: int = 200):
    """Render before/after image pair for change detection.

    Args:
        image_path_from: Path to "before" image
        image_path_to: Path to "after" image
        year_from: Starting year
        year_to: Ending year
        max_width: Maximum width per image

    Returns:
        Dash HTML Div with side-by-side images
    """
    before_img = html.Div([
        html.Div(f"{year_from}",
                style={'fontSize': 12, 'color': COLORS['text-secondary'],
                       'marginBottom': 4, 'fontWeight': '500'}),
        render_image(image_path_from, max_width=max_width)
    ], style={'flex': 1})

    after_img = html.Div([
        html.Div(f"{year_to}",
                style={'fontSize': 12, 'color': COLORS['text-secondary'],
                       'marginBottom': 4, 'fontWeight': '500'}),
        render_image(image_path_to, max_width=max_width)
    ], style={'flex': 1})

    return html.Div([before_img, after_img],
                   style={'display': 'flex', 'gap': 15})
