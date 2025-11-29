"""Simple location details display module."""

from dash import html
from pathlib import Path
from ...config import COLORS
from ...utils.display import encode_image_to_base64


def render_simple_details(location_id: int, street_name: str, images: list):
    """Render simple location details with street name and images.

    Args:
        location_id: Location ID
        street_name: Street name to display
        images: List of dictionaries with 'image_path' and 'year' keys

    Returns:
        Dash HTML Div with location details
    """
    if not street_name and not images:
        return html.Div([
            html.P("Click on a location to view details",
                  style={'color': COLORS['text-secondary'], 'fontStyle': 'italic',
                         'textAlign': 'center', 'padding': '20px'})
        ])

    components = []

    # Location ID and street name
    components.append(
        html.Div([
            html.H4(f"Location {location_id}",
                   style={'color': COLORS['text'], 'marginTop': 0, 'marginBottom': 10}),
            html.P(street_name or "Unknown location",
                  style={'color': COLORS['text-secondary'], 'fontSize': 16, 'marginBottom': 20})
        ])
    )

    # Images
    if images and len(images) > 0:
        components.append(
            html.H4("Images",
                   style={'color': COLORS['text'], 'fontSize': 15, 'marginBottom': 15,
                          'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': 5})
        )

        for img in images:
            img_path = Path(img['image_path'])
            if img_path.exists():
                img_base64 = encode_image_to_base64(img_path, max_width=380)
                if img_base64:
                    components.append(
                        html.Div([
                            html.Div(f"Year: {img.get('year', 'N/A')}",
                                   style={'fontSize': 13, 'color': COLORS['text'],
                                          'marginBottom': 8, 'fontWeight': 500}),
                            html.Img(src=img_base64,
                                   style={'width': '100%', 'borderRadius': '4px',
                                          'border': f"1px solid {COLORS['border']}"}),
                        ], style={'marginBottom': 20})
                    )
    else:
        components.append(
            html.P("No images available",
                  style={'color': COLORS['text-secondary'], 'fontStyle': 'italic',
                         'textAlign': 'center', 'padding': '20px'})
        )

    return html.Div(components)
