"""Reusable result formatting components for the dashboard."""

from pathlib import Path
import base64
from io import BytesIO
import logging

from dash import html
import pandas as pd

from ...config import COLORS
from ...utils.display import encode_image_to_base64

logger = logging.getLogger(__name__)

def format_results_accordion(results: pd.DataFrame, show_years: bool = False) -> html.Div:
    """Format search results as accordion with inline images.

    Args:
        results: DataFrame with search results including image_path, similarity, etc.
        show_years: Whether to show year information in headers

    Returns:
        Dash HTML Div with accordion items
    """
    if results.empty:
        return html.Div("No results found.",
                       style={'color': COLORS['text-secondary'], 'fontStyle': 'italic'})

    accordion_items = []

    for i, row in enumerate(results.itertuples(), 1):
        # Header with location info and similarity
        # Safely get location name with explicit type checking to avoid array boolean evaluation
        location_name = None
        if hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                location_name = streets

        if not location_name and hasattr(row, 'location_key'):
            location_name = row.location_key

        if not location_name:
            location_name = f"Location {row.location_id}"

        if hasattr(row, 'year'):
            header_text = f"#{i} - {location_name} (Year {row.year})"
        elif show_years:
            header_text = f"#{i} - {location_name} ({row.year_from} → {row.year_to})"
        else:
            header_text = f"#{i} - {location_name}"

        similarity_badge = html.Span(
            f"{row.similarity:.4f}",
            className='similarity-badge'
        )

        header = html.Div([
            html.Span(header_text),
            similarity_badge
        ], className='accordion-header')

        # Content with image
        content_items = []

        # Location details
        content_items.append(
            html.Div([
                html.Strong("Location ID: "),
                html.Span(str(row.location_id))
            ], className='location-info', style={'marginBottom': '5px'})
        )

        # Show location_key if different from additional_streets
        # Use explicit type checking to avoid array boolean evaluation
        show_location_key = False
        if hasattr(row, 'location_key') and hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                show_location_key = True

        if show_location_key:
            content_items.append(
                html.Div([
                    html.Strong("Location Key: "),
                    html.Span(row.location_key)
                ], className='location-info', style={'marginBottom': '5px'})
            )

        # Image
        if hasattr(row, 'image_path'):
            image_path = Path(row.image_path)
            if image_path.exists():
                img_base64 = encode_image_to_base64(image_path)
                if img_base64:
                    content_items.append(
                        html.Div([
                            html.Img(src=img_base64, style={'maxWidth': '100%'})
                        ], className='image-container')
                    )
                else:
                    content_items.append(
                        html.Div("Image could not be loaded",
                                style={'color': COLORS['warning'], 'marginTop': 10})
                    )
            else:
                content_items.append(
                    html.Div(f"Image not found: {image_path.name}",
                            style={'color': COLORS['warning'], 'marginTop': 10})
                )

        content = html.Div(content_items, className='accordion-content')

        accordion_items.append(
            html.Div([header, content], className='accordion-item')
        )

    return html.Div(accordion_items)


def format_change_results_accordion(results: pd.DataFrame) -> html.Div:
    """Format change search results with paired before/after images side by side.

    Args:
        results: DataFrame with change results including image_path_from and image_path_to

    Returns:
        Dash HTML Div with accordion items showing paired images
    """
    if results.empty:
        return html.Div("No results found.",
                       style={'color': COLORS['text-secondary'], 'fontStyle': 'italic'})

    accordion_items = []

    for i, row in enumerate(results.itertuples(), 1):
        # Header with location info and similarity
        location_name = None
        if hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                location_name = streets

        if not location_name and hasattr(row, 'location_key'):
            location_name = row.location_key

        if not location_name:
            location_name = f"Location {row.location_id}"

        header_text = f"#{i} - {location_name} ({row.year_from} → {row.year_to})"

        similarity_badge = html.Span(
            f"{row.similarity:.4f}",
            className='similarity-badge'
        )

        header = html.Div([
            html.Span(header_text),
            similarity_badge
        ], className='accordion-header')

        # Content with paired images
        content_items = []

        # Location details
        content_items.append(
            html.Div([
                html.Strong("Location ID: "),
                html.Span(str(row.location_id))
            ], className='location-info', style={'marginBottom': '5px'})
        )

        # Show location_key if different from additional_streets
        show_location_key = False
        if hasattr(row, 'location_key') and hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                show_location_key = True

        if show_location_key:
            content_items.append(
                html.Div([
                    html.Strong("Location Key: "),
                    html.Span(row.location_key)
                ], className='location-info', style={'marginBottom': '5px'})
            )

        # Paired images side by side
        if hasattr(row, 'image_path_from') and hasattr(row, 'image_path_to'):
            image_from_path = Path(row.image_path_from) if row.image_path_from else None
            image_to_path = Path(row.image_path_to) if row.image_path_to else None

            # Create side-by-side container
            paired_images = []

            # Before image (left)
            before_container = []
            before_container.append(
                html.Div([
                    html.Strong(f"Before ({row.year_from})"),
                ], style={'marginBottom': '8px', 'textAlign': 'center', 'color': COLORS['text']})
            )

            if image_from_path and image_from_path.exists():
                img_base64 = encode_image_to_base64(image_from_path, max_width=350)
                if img_base64:
                    before_container.append(
                        html.Img(src=img_base64, style={'maxWidth': '100%', 'borderRadius': '4px'})
                    )
                else:
                    before_container.append(
                        html.Div("Could not load image",
                                style={'color': COLORS['warning'], 'fontSize': '12px'})
                    )
            else:
                before_container.append(
                    html.Div("Image not found",
                            style={'color': COLORS['warning'], 'fontSize': '12px'})
                )

            # Show file path
            if image_from_path:
                before_container.append(
                    html.Div(
                        image_from_path.name,
                        style={'fontSize': '11px', 'color': COLORS['text-secondary'],
                               'marginTop': '5px', 'wordBreak': 'break-all'}
                    )
                )

            # After image (right)
            after_container = []
            after_container.append(
                html.Div([
                    html.Strong(f"After ({row.year_to})"),
                ], style={'marginBottom': '8px', 'textAlign': 'center', 'color': COLORS['text']})
            )

            if image_to_path and image_to_path.exists():
                img_base64 = encode_image_to_base64(image_to_path, max_width=350)
                if img_base64:
                    after_container.append(
                        html.Img(src=img_base64, style={'maxWidth': '100%', 'borderRadius': '4px'})
                    )
                else:
                    after_container.append(
                        html.Div("Could not load image",
                                style={'color': COLORS['warning'], 'fontSize': '12px'})
                    )
            else:
                after_container.append(
                    html.Div("Image not found",
                            style={'color': COLORS['warning'], 'fontSize': '12px'})
                )

            # Show file path
            if image_to_path:
                after_container.append(
                    html.Div(
                        image_to_path.name,
                        style={'fontSize': '11px', 'color': COLORS['text-secondary'],
                               'marginTop': '5px', 'wordBreak': 'break-all'}
                    )
                )

            # Combine both sides
            content_items.append(
                html.Div([
                    html.Div(before_container, style={
                        'flex': '1',
                        'padding': '10px',
                        'textAlign': 'center'
                    }),
                    html.Div(after_container, style={
                        'flex': '1',
                        'padding': '10px',
                        'textAlign': 'center'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '20px',
                    'marginTop': '15px',
                    'alignItems': 'flex-start'
                })
            )

        content = html.Div(content_items, className='accordion-content')

        accordion_items.append(
            html.Div([header, content], className='accordion-item')
        )

    return html.Div(accordion_items)
