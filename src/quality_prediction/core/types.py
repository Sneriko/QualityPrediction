from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .geometry import BBox


@dataclass
class TextLine:
    text: str
    bbox: Optional[BBox] = None


@dataclass
class PageContent:
    """Ordered lines on a page."""
    lines: List[TextLine]

    def texts(self) -> List[str]:
        return [l.text for l in self.lines]

    def sorted_by_reading_order(self) -> "PageContent":
        if self.lines and all(l.bbox is not None for l in self.lines):
            return PageContent(lines=sorted(self.lines, key=lambda l: l.bbox.sort_key()))  # type: ignore[union-attr]
        return PageContent(lines=list(self.lines))
