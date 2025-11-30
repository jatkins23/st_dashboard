"""Reusable result formatting components using Bootstrap."""

from pathlib import Path
import logging

from dash import html
import dash_bootstrap_components as dbc
import pandas as pd

from ...utils.display import encode_image_to_base64

logger = logging.getLogger(__name__)


def create_state_result_card(
    rank: int,
    location_id: int,
    location_name: str,
    similarity: float,
    image_path: str | Path | None = None,
    year: int | None = None,
    location_key: str | None = None,
    additional_info: dict | None = None
) -> dbc.Card:
    """Create a single result card.

    Args:
        rank: Result ranking number
        location_id: Location ID
        location_name: Display name for the location
        similarity: Similarity score
        image_path: Optional path to image
        year: Optional year information
        location_key: Optional location key
        additional_info: Optional dict of additional information to display

    Returns:
        Bootstrap Card with result information
    """
    # Card header with rank and similarity
    header = dbc.CardHeader([
        html.Div([
            html.Span(f"#{rank} - {location_name}", className='fw-bold'),
            dbc.Badge(f"{similarity:.4f}", color='info', className='ms-auto')
        ], className='d-flex align-items-center justify-content-between')
    ])

    # Card body content
    body_content = []

    # Location ID
    body_content.append(
        html.Div([
            html.Small([
                html.Strong("Location ID: "),
                html.Span(str(location_id))
            ], className='text-muted')
        ], className='mb-2')
    )

    # Year if provided
    if year is not None:
        body_content.append(
            html.Div([
                html.Small([
                    html.Strong("Year: "),
                    html.Span(str(year))
                ], className='text-muted')
            ], className='mb-2')
        )

    # Location key if different from name
    if location_key and location_key != location_name:
        body_content.append(
            html.Div([
                html.Small([
                    html.Strong("Location Key: "),
                    html.Span(location_key)
                ], className='text-muted')
            ], className='mb-2')
        )

    # Additional info
    if additional_info:
        for key, value in additional_info.items():
            body_content.append(
                html.Div([
                    html.Small([
                        html.Strong(f"{key}: "),
                        html.Span(str(value))
                    ], className='text-muted')
                ], className='mb-2')
            )

    # Image if provided
    if image_path:
        img_path = Path(image_path) if not isinstance(image_path, Path) else image_path
        if img_path.exists():
            img_base64 = encode_image_to_base64(img_path)
            if img_base64:
                body_content.append(
                    html.Img(
                        src=img_base64,
                        className='img-fluid rounded mt-2',
                        style={'maxWidth': '100%'}
                    )
                )
            else:
                body_content.append(
                    dbc.Alert("Image could not be loaded", color='warning', className='mt-2 small')
                )
        else:
            body_content.append(
                dbc.Alert(f"Image not found", color='warning', className='mt-2 small')
            )

    return dbc.Card([
        header,
        dbc.CardBody(body_content)
    ], className='mb-2')


def create_change_result_card(
    rank: int,
    location_id: int,
    location_name: str,
    similarity: float,
    year_from: int,
    year_to: int,
    image_path_from: str | Path | None = None,
    image_path_to: str | Path | None = None,
    location_key: str | None = None
) -> dbc.Card:
    """Create a single change result card with before/after images.

    Args:
        rank: Result ranking number
        location_id: Location ID
        location_name: Display name for the location
        similarity: Similarity score
        year_from: Starting year
        year_to: Ending year
        image_path_from: Optional path to before image
        image_path_to: Optional path to after image
        location_key: Optional location key

    Returns:
        Bootstrap Card with change result information
    """
    # Card header
    header = dbc.CardHeader([
        html.Div([
            html.Span(f"#{rank} - {location_name} ({year_from} → {year_to})", className='fw-bold'),
            dbc.Badge(f"{similarity:.4f}", color='info', className='ms-auto')
        ], className='d-flex align-items-center justify-content-between')
    ])

    # Card body content
    body_content = []

    # Location info
    body_content.append(
        html.Div([
            html.Small([
                html.Strong("Location ID: "),
                html.Span(str(location_id))
            ], className='text-muted')
        ], className='mb-3')
    )

    # Before/After images in columns
    before_col = []
    after_col = []

    # Before image
    before_col.append(html.H6(f"Before ({year_from})", className='text-center mb-2'))
    if image_path_from:
        img_from_path = Path(image_path_from) if not isinstance(image_path_from, Path) else image_path_from
        if img_from_path.exists():
            img_base64 = encode_image_to_base64(img_from_path, max_width=350)
            if img_base64:
                before_col.append(html.Img(src=img_base64, className='img-fluid rounded'))
            else:
                before_col.append(dbc.Alert("Could not load", color='warning', className='small'))
        else:
            before_col.append(dbc.Alert("Image not found", color='warning', className='small'))
        before_col.append(html.Small(img_from_path.name, className='text-muted d-block mt-1 text-center'))
    else:
        before_col.append(html.Div("No image", className='text-muted small text-center'))

    # After image
    after_col.append(html.H6(f"After ({year_to})", className='text-center mb-2'))
    if image_path_to:
        img_to_path = Path(image_path_to) if not isinstance(image_path_to, Path) else image_path_to
        if img_to_path.exists():
            img_base64 = encode_image_to_base64(img_to_path, max_width=350)
            if img_base64:
                after_col.append(html.Img(src=img_base64, className='img-fluid rounded'))
            else:
                after_col.append(dbc.Alert("Could not load", color='warning', className='small'))
        else:
            after_col.append(dbc.Alert("Image not found", color='warning', className='small'))
        after_col.append(html.Small(img_to_path.name, className='text-muted d-block mt-1 text-center'))
    else:
        after_col.append(html.Div("No image", className='text-muted small text-center'))

    # Add image row
    body_content.append(
        dbc.Row([
            dbc.Col(before_col, width=6),
            dbc.Col(after_col, width=6),
        ])
    )

    return dbc.Card([
        header,
        dbc.CardBody(body_content)
    ], className='mb-2')


def format_results_accordion(results: pd.DataFrame, show_years: bool = False) -> dbc.Accordion:
    """Format search results as Bootstrap accordion with collapsible cards.

    Args:
        results: DataFrame with search results
        show_years: Whether to show year information

    Returns:
        Bootstrap Accordion with collapsible result cards
    """
    if results.empty:
        return html.Div("No results found.", className='text-muted fst-italic')

    accordion_items = []

    for i, row in enumerate(results.itertuples(), 1):
        # Get location name
        location_name = None
        if hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                location_name = streets

        if not location_name and hasattr(row, 'location_key'):
            location_name = row.location_key

        if not location_name:
            location_name = f"Location {row.location_id}"

        # Get optional fields
        year = row.year if hasattr(row, 'year') else None
        location_key = row.location_key if hasattr(row, 'location_key') else None
        image_path = row.image_path if hasattr(row, 'image_path') and row.image_path else None

        # Create accordion item title with badge
        title = html.Div([
            html.Span(f"#{i} - {location_name}", className='me-2'),
            dbc.Badge(f"{row.similarity:.4f}", color='info', pill=True)
        ], className='d-flex align-items-center')

        # Create content
        content = []

        # Location details
        content.append(
            html.Div([
                html.Small([html.Strong("Location ID: "), str(row.location_id)], className='text-muted'),
                html.Br(),
                html.Small([html.Strong("Year: "), str(year)], className='text-muted') if year else None,
            ], className='mb-2')
        )

        # Location key if different
        if location_key and location_key != location_name:
            content.append(
                html.Small([html.Strong("Location Key: "), location_key], className='text-muted d-block mb-2')
            )

        # Image
        if image_path:
            img_path = Path(image_path) if not isinstance(image_path, Path) else image_path
            if img_path.exists():
                img_base64 = encode_image_to_base64(img_path)
                if img_base64:
                    content.append(
                        html.Img(src=img_base64, className='img-fluid rounded mt-2')
                    )
                else:
                    content.append(
                        dbc.Alert("Image could not be loaded", color='warning', className='mt-2 small')
                    )
            else:
                content.append(
                    dbc.Alert("Image not found", color='warning', className='mt-2 small')
                )

        # Add to accordion
        accordion_items.append(
            dbc.AccordionItem(
                content,
                title=title,
                item_id=f"result-{i}"
            )
        )

    return dbc.Accordion(
        accordion_items,
        start_collapsed=True,
        always_open=False,
        flush=True
    )


def format_change_results_accordion(results: pd.DataFrame) -> dbc.Accordion:
    """Format change search results as Bootstrap accordion with collapsible cards.

    Args:
        results: DataFrame with change results

    Returns:
        Bootstrap Accordion with collapsible change result cards
    """
    if results.empty:
        return html.Div("No results found.", className='text-muted fst-italic')

    accordion_items = []

    for i, row in enumerate(results.itertuples(), 1):
        # Get location name
        location_name = None
        if hasattr(row, 'additional_streets'):
            streets = row.additional_streets
            if streets is not None and isinstance(streets, str) and streets.strip():
                location_name = streets

        if not location_name and hasattr(row, 'location_key'):
            location_name = row.location_key

        if not location_name:
            location_name = f"Location {row.location_id}"

        # Create accordion item title with badge
        title = html.Div([
            html.Span(f"#{i} - {location_name} ({row.year_from} → {row.year_to})", className='me-2'),
            dbc.Badge(f"{row.similarity:.4f}", color='info', pill=True)
        ], className='d-flex align-items-center')

        # Create content
        content = []

        # Location info
        content.append(
            html.Small([html.Strong("Location ID: "), str(row.location_id)],
                      className='text-muted d-block mb-3')
        )

        # Before/After images
        image_from = row.image_path_from if hasattr(row, 'image_path_from') and row.image_path_from else None
        image_to = row.image_path_to if hasattr(row, 'image_path_to') and row.image_path_to else None

        before_col = []
        after_col = []

        # Before image
        before_col.append(html.H6(f"Before ({row.year_from})", className='text-center mb-2'))
        if image_from:
            img_from_path = Path(image_from) if not isinstance(image_from, Path) else image_from
            if img_from_path.exists():
                img_base64 = encode_image_to_base64(img_from_path, max_width=350)
                if img_base64:
                    before_col.append(html.Img(src=img_base64, className='img-fluid rounded'))
                else:
                    before_col.append(dbc.Alert("Could not load", color='warning', className='small'))
            else:
                before_col.append(dbc.Alert("Image not found", color='warning', className='small'))
        else:
            before_col.append(html.Div("No image", className='text-muted small text-center'))

        # After image
        after_col.append(html.H6(f"After ({row.year_to})", className='text-center mb-2'))
        if image_to:
            img_to_path = Path(image_to) if not isinstance(image_to, Path) else image_to
            if img_to_path.exists():
                img_base64 = encode_image_to_base64(img_to_path, max_width=350)
                if img_base64:
                    after_col.append(html.Img(src=img_base64, className='img-fluid rounded'))
                else:
                    after_col.append(dbc.Alert("Could not load", color='warning', className='small'))
            else:
                after_col.append(dbc.Alert("Image not found", color='warning', className='small'))
        else:
            after_col.append(html.Div("No image", className='text-muted small text-center'))

        # Add image row
        content.append(
            dbc.Row([
                dbc.Col(before_col, width=6),
                dbc.Col(after_col, width=6),
            ])
        )

        # Add to accordion
        accordion_items.append(
            dbc.AccordionItem(
                content,
                title=title,
                item_id=f"change-result-{i}"
            )
        )

    return dbc.Accordion(
        accordion_items,
        start_collapsed=True,
        always_open=False,
        flush=True
    )