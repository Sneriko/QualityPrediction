from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return float(self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return float(self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)

    def overlaps(self, other: "BBox") -> bool:
        return not (
            self.xmax <= other.xmin
            or self.xmin >= other.xmax
            or self.ymax <= other.ymin
            or self.ymin >= other.ymax
        )

    def sort_key(self) -> Tuple[float, float]:
        return (self.ymin, self.xmin)

    @staticmethod
    def from_page_coords(points_str: str) -> "BBox":
        """PAGE XML coords: 'x1,y1 x2,y2 ...' -> min/max bbox."""
        pts: list[tuple[float, float]] = []
        for pair in points_str.strip().split():
            x_str, y_str = pair.split(",")
            pts.append((float(x_str), float(y_str)))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return BBox(min(xs), min(ys), max(xs), max(ys))


def iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.xmin, b.xmin)
    inter_y1 = max(a.ymin, b.ymin)
    inter_x2 = min(a.xmax, b.xmax)
    inter_y2 = min(a.ymax, b.ymax)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)
