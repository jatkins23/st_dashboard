"""Location details display components."""

from dash import html, dcc
from pathlib import Path
from ...config import COLORS
from ...utils import encode_image_to_base64


def render_location_details(location_data: dict, recent_images: list = None, all_years: list = None) -> html.Div:
    """Render detailed information about a location.

    Args:
        location_data: Dictionary with location details
        recent_images: Optional list of image paths to display
        all_years: Optional list of all available years for this location

    Returns:
        Dash HTML Div with location details
    """
    if location_data is None:
        return html.Div([
            html.P("Location not found",
                  style={'color': COLORS['error'], 'textAlign': 'center', 'padding': '20px'})
        ])

    components = []

    # Header with location name
    #print(location_data)
    #location_name = location_data.get('additional_streets') or location_data.get('location_key', '')
    location_name = location_data.get('title')
    components.append(
        html.H3(location_name,
               style={'color': COLORS['text'], 'marginTop': 0, 'marginBottom': 15,
                      'fontSize': 18, 'fontWeight': 600})
    )

    # Location details section
    components.append(
        html.Div([
            html.H4("Location Information",
                   style={'color': COLORS['text'], 'fontSize': 15, 'marginBottom': 10,
                          'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': 5})
        ])
    )

    details_items = []
    details_items.append(create_detail_row("Location ID:", str(location_data['location_id'])))
    details_items.append(create_detail_row("Location Key:", location_data.get('location_key', 'N/A')))

    if location_data.get('street1') or location_data.get('street2'):
        street_info = []
        if location_data.get('street1'):
            street_info.append(str(location_data['street1']))
        if location_data.get('street2'):
            street_info.append(str(location_data['street2']))
        details_items.append(create_detail_row("Streets:", " & ".join(street_info)))

    if location_data.get('latitude') and location_data.get('longitude'):
        coords = f"{location_data['latitude']:.6f}, {location_data['longitude']:.6f}"
        details_items.append(create_detail_row("Coordinates:", coords))

    components.append(
        html.Div(details_items, style={'marginBottom': 20})
    )

    # Statistics section
    if location_data.get('year_count') or location_data.get('image_count'):
        components.append(
            html.Div([
                html.H4("Statistics",
                       style={'color': COLORS['text'], 'fontSize': 15, 'marginBottom': 10,
                              'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': 5})
            ])
        )

        stats_items = []
        if location_data.get('year_count'):
            stats_items.append(create_detail_row("Years Available:", str(location_data['year_count'])))
            year_range = f"{location_data.get('first_year', 'N/A')} - {location_data.get('last_year', 'N/A')}"
            stats_items.append(create_detail_row("Year Range:", year_range))

        if location_data.get('image_count'):
            stats_items.append(create_detail_row("Total Images:", str(location_data['image_count'])))

        components.append(
            html.Div(stats_items, style={'marginBottom': 20})
        )

    # Image gallery section
    if recent_images and len(recent_images) > 0:
        components.append(
            html.Div([
                html.H4("Image Gallery",
                       style={'color': COLORS['text'], 'fontSize': 15, 'marginBottom': 10,
                              'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': 5})
            ])
        )

        # Create year selector if multiple years available
        years_in_images = sorted(set(img.get('year') for img in recent_images if img.get('year')))

        if len(years_in_images) > 1:
            components.append(
                html.Div([
                    html.Label("Select Year:", style={'fontSize': 13, 'color': COLORS['text-secondary'],
                                                      'marginBottom': 5, 'display': 'block'}),
                    dcc.Dropdown(
                        id='detail-year-selector',
                        options=[{'label': str(year), 'value': year} for year in years_in_images],
                        value=years_in_images[-1] if years_in_images else None,
                        style={'marginBottom': 15}
                    )
                ])
            )

        # Display images
        image_components = []
        for img_info in recent_images:
            image_path = Path(img_info['image_path'])
            if image_path.exists():
                img_base64 = encode_image_to_base64(image_path, max_width=400)
                if img_base64:
                    image_components.append(
                        html.Div([
                            html.Div(f"Year: {img_info.get('year', 'N/A')}",
                                   style={'fontSize': 13, 'color': COLORS['text'],
                                          'marginBottom': 8, 'fontWeight': 500}),
                            html.Img(src=img_base64,
                                   style={'width': '100%', 'borderRadius': '4px',
                                          'border': f"1px solid {COLORS['border']}"}),
                        ], style={'marginBottom': 20})
                    )
            else:
                image_components.append(
                    html.Div([
                        html.Div(f"Year: {img_info.get('year', 'N/A')}",
                               style={'fontSize': 13, 'color': COLORS['text'],
                                      'marginBottom': 8, 'fontWeight': 500}),
                        html.Div(f"Image not found: {image_path.name}",
                               style={'color': COLORS['warning'], 'fontSize': 12,
                                      'padding': 20, 'textAlign': 'center',
                                      'border': f"1px dashed {COLORS['border']}",
                                      'borderRadius': '4px'})
                    ], style={'marginBottom': 20})
                )

        if image_components:
            components.append(html.Div(image_components))
        else:
            components.append(
                html.Div("No images available",
                       style={'color': COLORS['text-secondary'], 'fontStyle': 'italic',
                              'textAlign': 'center', 'padding': 20})
            )

    return html.Div(components)


def create_detail_row(label: str, value: str) -> html.Div:
    """Create a detail row with label and value.

    Args:
        label: Label text
        value: Value text

    Returns:
        Dash HTML Div with detail row
    """
    return html.Div([
        html.Span(label,
                 style={'color': COLORS['text-secondary'], 'fontSize': 14, 'fontWeight': '500'}),
        html.Span(' ' + value,
                 style={'color': COLORS['text'], 'fontSize': 14, 'marginLeft': 8})
    ], style={'marginBottom': 8})
