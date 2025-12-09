"""Simple Bootstrap-based dashboard layout."""

import dash
from pathlib import Path
import dash_bootstrap_components as dbc
from .components.dashboard import Dashboard


def create_app():
    """Create Dash app with Bootstrap theme."""
    # Get path to assets folder
    assets_folder = Path(__file__).parent.parent / 'assets'

    return dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=str(assets_folder)
    )


def create_layout(universe_name: str, available_years: list, all_streets: list):
    """Create dashboard layout.

    Args:
        universe_name: Name of the universe being explored
        available_years: List of available years for the search form
        all_streets: List of all unique street names

    Returns:
        Dashboard component layout
    """
    return Dashboard(
        universe_name=universe_name,
        available_years=available_years,
        all_streets=all_streets
    ).layout
