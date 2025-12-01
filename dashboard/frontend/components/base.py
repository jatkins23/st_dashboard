from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from dash import Dash
from dash.development.base_component import Component as DashComponent


class Registerable(Protocol):
    def register_callbacks(self, app: Dash) -> None: ...


@dataclass
class BaseComponent(ABC):
    """Base class for modular Dash UI components."""

    id_prefix: str

    @property
    @abstractmethod
    def layout(self) -> DashComponent:
        """Return the Dash layout for this component."""

    def __call__(self) -> DashComponent:
        """Allow component() syntax as shorthand for component.layout."""
        return self.layout

    @abstractmethod
    def register_callbacks(self, app: Dash) -> None:
        """Register callbacks related to this component."""

    def Id(self, suffix: str) -> str:
        """Helper to create unique IDs for sub-elements."""
        return f"{self.id_prefix}--{suffix}"
