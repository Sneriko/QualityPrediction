import logging
import math
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal, Iterable, Sequence

from pagexml.parser import parse_pagexml_file

from htrflow.models.base_model import BaseModel
from htrflow.models.importer import all_models
from htrflow.postprocess import metrics
from htrflow.postprocess.reading_order import order_regions, top_down
from htrflow.postprocess.word_segmentation import simple_word_segmentation
from htrflow.results import Result
from htrflow.serialization import get_serializer, save_collection
from htrflow.utils.imgproc import NumpyImage, binarize, write
from htrflow.utils.layout import estimate_printspace, is_twopage
from htrflow.volume.node import Node
from htrflow.volume.volume import Collection


logger = logging.getLogger(__name__)


@dataclass
class StepMetadata:
    description: str
    settings: dict[str, str]


class PipelineStep:
    """
    Pipeline step base class.

    Pipeline steps are implemented by subclassing this class and
    overriding the `run()` method.
    """

    parent_pipeline = None
    metadata: StepMetadata | None = None

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def run(self, collection: Collection) -> Collection:
        """
        Run the pipeline step.

        Arguments:
            collection: Input collection

        Returns:
            A new collection, updated with the results of the pipeline step.
        """

    def __str__(self):
        return f"{self.__class__.__name__}"


class Inference(PipelineStep):
    """
    Run model inference.

    This is a generic pipeline step for any type of model inference.
    This step always runs the model on the images of the collection's
    leaf nodes.

    Example YAML:
    ```yaml
    - step: Inference
      settings:
        model: DiT
        model_settings:
          model: ...
    ```
    """

    def __init__(self, model_class, model_kwargs, generation_kwargs):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self.model = None

    def _init_model(self):
        self.model = self.model_class(**self.model_kwargs)
        self.metadata = StepMetadata(str(self), self.model.metadata)

    @classmethod
    def from_config(cls, config):
        name = config.pop("model").lower()
        if name not in MODELS:
            model_names = [model.__name__ for model in all_models()]
            msg = f"Model {name} is not supported. The available models are: {', '.join(model_names)}."
            logger.error(msg)
            raise NotImplementedError(msg)
        model = MODELS[name]
        generation_kwargs = config.pop("generation_settings", {})
        init_kwargs = config.pop("model_settings", {}) | config
        return cls(model, init_kwargs, generation_kwargs)

    def run(self, collection):
        if self.model is None:
            self._init_model()
        result = self.model(collection.segments(), **self.generation_kwargs)
        collection.update(result)
        return collection


class ImportSegmentation(PipelineStep):
    """
    Import segmentation from PageXML files.

    This step replicates the line segmentation from PageXML files.
    It can be used to import ground truth segmentation for
    evaluation purposes.

    Example YAML:
    ```yaml
    - step: ImportSegmentation
      settings:
        source: /path/to/pageXMLs
    ```
    """

    def __init__(self, source: str):
        """
        Arguments:
            source: Path to a directory with PageXML files. The XML files
                must have the same names as the input image files (ignoring
                the file extension).
        """
        self.source = source

    def run(self, collection):
        pages = []
        for page in collection:
            try:
                pages.append(parse_pagexml_file(os.path.join(self.source, page.label + ".xml")))
            except ValueError:
                pages.append(None)

        results = []
        for page in pages:
            if page is None:
                results.append(Result())
                continue
            shape = (page.coords.height, page.coords.width)
            polygons = [line.coords.points for line in page.get_lines()]
            results.append(Result.segmentation_result(shape, {}, polygons=polygons))
        collection.update(results)
        return collection


class Segmentation(Inference):
    """
    Run a segmentation model.

    See [Segmentation models](models.md#segmentation-models) for available models.

    Example YAML:
    ```yaml
    - step: Segmentation
      settings:
        model: yolo
        model_settings:
          model: Riksarkivet/yolov9-regions-1
    ```
    """

    pass


class TextRecognition(Inference):
    """
    Run a text recognition model.

    See [Text recognition models](models.md#text-recognition-models) for available models.

    Example YAML:
    ```yaml
    - step: TextRecognition
      settings:
        model: TrOCR
        model_settings:
          model: Riksarkivet/trocr-base-handwritten-hist-swe-2
    ```
    """

    pass

import logging
from dataclasses import asdict
from statistics import median
from typing import Sequence

from htrflow.volume.node import Node
from htrflow.utils.geometry import Bbox

logger = logging.getLogger(__name__)

class OrderRegionsAndLinesAfterMakeRegions(PipelineStep):
    """
    Ordering step for the new pipeline:
      Segmentation (flat lines) -> TextRecognition -> MakeRegionsFromLines -> (this step) -> Export

    What it does:
      1) Orders regions by (column_index, top-to-bottom, left-to-right).
         If column_index is missing, infers a column from x-position.
      2) Orders lines within each region top-to-bottom (and left-to-right for ties).
      3) Optionally gathers any page-level lines into a final region.

    This step is safe after MakeRegionsFromLines where regions are plain Node containers.
    """

    def __init__(
        self,
        region_label: str = "textregion",
        line_label: str = "textline",
        # If True: any page-level lines (unassigned) will be moved into a final region container
        gather_unassigned_lines: bool = True,
        unassigned_region_label: str = "textregion_unassigned",
        # If MakeRegionsFromLines didn't set column_index, infer 2 columns if the x-center split is "strong"
        infer_columns_if_missing: bool = True,
        infer_column_gap_frac: float = 0.12,  # gap fraction of page width to decide a split
    ):
        self.region_label = region_label
        self.line_label = line_label
        self.gather_unassigned_lines = bool(gather_unassigned_lines)
        self.unassigned_region_label = unassigned_region_label

        self.infer_columns_if_missing = bool(infer_columns_if_missing)
        self.infer_column_gap_frac = float(infer_column_gap_frac)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # ---- helpers ----
    def _is_region(self, node: Node) -> bool:
        # In your tree, "regions" are containers with children; MakeRegionsFromLines sets label + data
        if node.children:
            lab = _node_class_label(node)
            if lab is None:
                return False
            # accept either node label or segmentation_label field
            return lab == self.region_label or node.get("segmentation_label") == self.region_label \
                or node._label == self.region_label if hasattr(node, "_label") else False
        return False

    def _is_line(self, node: Node) -> bool:
        lab = _node_class_label(node)
        return lab == self.line_label

    def _node_bbox(self, node: Node) -> Bbox | None:
        return _bbox(node)

    def _infer_region_column_index(self, page: Node, regions: list[Node]) -> dict[Node, int]:
        """
        If regions already have column_index in .data, keep it.
        Otherwise, infer columns from region x-centers.
        """
        page_w = float(page.bbox.width) if hasattr(page, "bbox") else 1.0

        # First check if any region already has a column_index
        has_any = any(isinstance(r.get("column_index", None), int) for r in regions)
        if has_any:
            return {r: int(r.get("column_index", 0)) if isinstance(r.get("column_index", None), int) else 0 for r in regions}

        if not self.infer_columns_if_missing or len(regions) <= 1:
            return {r: 0 for r in regions}

        # infer from x-centers
        xcs = []
        valid_regions = []
        for r in regions:
            b = self._node_bbox(r)
            if b is None:
                continue
            xcs.append((_center_x(b), r))
            valid_regions.append(r)

        if len(xcs) <= 1:
            return {r: 0 for r in regions}

        xcs.sort(key=lambda t: t[0])
        # find largest gap
        max_gap = 0.0
        split_at = None
        for i in range(1, len(xcs)):
            gap = xcs[i][0] - xcs[i - 1][0]
            if gap > max_gap:
                max_gap = gap
                split_at = i

        # decide if it's really two columns
        if split_at is None or max_gap < self.infer_column_gap_frac * page_w:
            return {r: 0 for r in regions}

        left = set(r for _, r in xcs[:split_at])
        return {r: (0 if r in left else 1) for r in regions}

    def _sort_lines(self, lines: list[Node]) -> list[Node]:
        # Top-down, then left-to-right as tie-break
        def key(n: Node):
            b = self._node_bbox(n)
            if b is None:
                return (10**9, 10**9, 10**9)
            return (b.ymin, b.xmin, b.xmax)
        return sorted(lines, key=key)

    # ---- main ----
    def run(self, collection):
        for page in collection:
            if page.is_leaf():
                continue

            # Separate current children into regions and page-level lines/others
            regions: list[Node] = []
            page_level_lines: list[Node] = []
            other: list[Node] = []

            for ch in list(page.children):
                if self._is_region(ch):
                    regions.append(ch)
                elif self._is_line(ch):
                    page_level_lines.append(ch)
                else:
                    other.append(ch)

            # Order lines within each region
            for r in regions:
                r.children = self._sort_lines(list(r.children))

            # Gather unassigned lines into a final region (optional)
            if self.gather_unassigned_lines and page_level_lines:
                # Create a region container at the end
                boxes = [self._node_bbox(ln) for ln in page_level_lines]
                boxes = [b for b in boxes if b is not None]
                if boxes:
                    rb = _bbox_union(boxes)
                else:
                    rb = None

                unreg = Node(parent=page, label=self.unassigned_region_label)
                if rb is not None:
                    unreg.add_data(
                        segmentation_label=self.unassigned_region_label,
                        segmentation_confidence=1.0,
                        bbox=asdict(rb),
                        polygon=str(rb.polygon()),
                        column_index=999,   # after normal columns
                        block_index=999,
                        constructed_from="unassigned_lines",
                    )
                unreg.children = self._sort_lines(page_level_lines)
                for ln in unreg.children:
                    ln.parent = unreg
                regions.append(unreg)
                page_level_lines = []

            # Determine column indices for ordering
            col_index = self._infer_region_column_index(page, regions)

            # Sort regions: column first, then top-down, then left-to-right
            def region_key(r: Node):
                b = self._node_bbox(r)
                if b is None:
                    return (col_index.get(r, 0), 10**9, 10**9)
                return (col_index.get(r, 0), b.ymin, b.xmin)

            regions_sorted = sorted(regions, key=region_key)

            # Rebuild page children: ordered regions + any leftovers
            page.children = []
            for r in regions_sorted:
                r.parent = page
                # refresh column_index if inferred
                if "column_index" not in r.data and r in col_index:
                    r.add_data(column_index=int(col_index[r]))
                page.children.append(r)

            # Keep any non-region, non-line children after regions (rare)
            for ch in other:
                ch.parent = page
                page.children.append(ch)

        collection.relabel()
        return collection

import logging
from dataclasses import asdict
from statistics import median
from typing import Sequence

from htrflow.volume.node import Node
from htrflow.utils.geometry import Bbox

logger = logging.getLogger(__name__)


# -------------------------
# Helpers (reuse-friendly)
# -------------------------
def _node_class_label(node: Node) -> str | None:
    # SegmentNode label comes from Segment.class_label (preferred)
    if hasattr(node, "_segment") and getattr(node._segment, "class_label", None):
        return node._segment.class_label
    # Node label as fallback
    if hasattr(node, "_label") and isinstance(node._label, str):
        return node._label
    v = node.get("segmentation_label", None)
    return v if isinstance(v, str) else None


def _bbox(node: Node) -> Bbox | None:
    try:
        b = getattr(node, "bbox", None)
        return b if isinstance(b, Bbox) else None
    except Exception:
        return None


def _bbox_union(boxes: Sequence[Bbox]) -> Bbox:
    xmin = min(b.xmin for b in boxes)
    ymin = min(b.ymin for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    return Bbox(xmin, ymin, xmax, ymax)


def _center_x(b: Bbox) -> float:
    return (b.xmin + b.xmax) / 2.0


def _x_overlap_ratio(a: Bbox, b: Bbox) -> float:
    inter = a.intersection(b)
    if inter is None:
        return 0.0
    return inter.width / max(1, min(a.width, b.width))


def _sort_top_down(nodes: list[Node]) -> list[Node]:
    def key(n: Node):
        b = _bbox(n)
        if b is None:
            return (10**9, 10**9)
        return (b.ymin, b.xmin)
    return sorted(nodes, key=key)


def _cluster_1d(values: list[float], gap: float) -> list[list[int]]:
    """
    Simple 1D clustering by sorting and splitting where consecutive gap > threshold.
    Returns clusters as lists of original indices.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    cur = [order[0]]
    for i in order[1:]:
        if (values[i] - values[cur[-1]]) > gap:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)
    clusters.append(cur)
    return clusters


def _robust_gap_threshold(gaps: list[float]) -> float:
    """
    Estimate a per-(column/page) vertical gap threshold from the distribution of
    inter-line gaps using median + MAD.

    The idea: typical line spacing clusters around median(gaps).
    Paragraph/block breaks appear as large outliers.
    """
    gaps = [float(g) for g in gaps if g >= 0.0]
    if len(gaps) < 8:
        # Too few observations -> be conservative (avoid over-splitting).
        return float(median(gaps)) if gaps else 0.0

    med = float(median(gaps))
    abs_dev = [abs(g - med) for g in gaps]
    mad = float(median(abs_dev)) if abs_dev else 0.0

    # "outlier-ish" gap threshold
    thr = med + 4.0 * mad

    # Clamp to avoid pathological pages (very noisy or very uniform)
    if med > 0.0:
        thr = max(0.8 * med, min(thr, 6.0 * med))
    else:
        thr = max(thr, 0.0)

    return thr


def _adaptive_min_x_overlap(line_bboxes: list[Bbox], page_width: int) -> float:
    """
    Decide whether x-overlap is reliable on this column, and how strict to be.

    Heuristic:
      - If line widths vary a lot (tight bboxes, indents, short insertions),
        x-overlap is unreliable -> return 0.0 (disabled).
      - If widths are consistent, require a small overlap to prevent cross-column merges.
    """
    if len(line_bboxes) < 5:
        return 0.0

    widths = sorted(float(b.width) for b in line_bboxes)
    w_med = float(median(widths))
    if w_med <= 0:
        return 0.0

    # Robust spread (approx IQR without numpy)
    q1 = widths[int(0.25 * (len(widths) - 1))]
    q3 = widths[int(0.75 * (len(widths) - 1))]
    iqr = float(q3 - q1)

    # Also look at indentation variability (xmin spread)
    xmins = sorted(float(b.xmin) for b in line_bboxes)
    x_med = float(median(xmins))
    x1 = xmins[int(0.25 * (len(xmins) - 1))]
    x3 = xmins[int(0.75 * (len(xmins) - 1))]
    x_iqr = float(x3 - x1)

    width_rel_spread = iqr / max(1.0, w_med)

    # If lots of indentation variability relative to page width, relax
    indent_rel = x_iqr / max(1.0, float(page_width))

    # Very ragged => disable overlap gate
    if width_rel_spread > 0.55 or indent_rel > 0.06:
        return 0.0

    # Moderately ragged => very small gate
    if width_rel_spread > 0.35 or indent_rel > 0.04:
        return 0.01

    # Clean running text => small but useful gate
    return 0.03


# -------------------------
# Step
# -------------------------
class MakeRegionsFromLines(PipelineStep):
    """
    Keep ONLY `line_labels` segments (filters out everything else), then constructs
    artificial regions (blocks) and columns from those lines.

    Changes vs older version:
      1) Vertical grouping threshold is ADAPTIVE per column using median+MAD of gaps
         (no v_gap_factor tuning needed).
      2) X-overlap is ADAPTIVE per column; disabled automatically when unreliable.

    Output structure:
      page -> region(Node) -> line(SegmentNode)
      plus optionally some page-level lines (unassigned).
    """

    def __init__(
        self,
        line_labels: Sequence[str] = ("textline",),
        region_label: str = "textregion",
        rebuild_from_scratch: bool = True,

        # Column clustering (still needs *one* stable rule; this is page-relative not dataset-specific)
        column_gap_frac: float = 0.08,
        merge_small_columns: bool = True,
        min_lines_per_column: int = 5,

        # correction/interlineation fallback
        correction_attach: bool = True,
        # This is relative to *measured* line height (robust); not a dataset-level spacing knob
        correction_max_dy_factor: float = 4.0,

        # leftovers
        keep_unassigned_at_page_level: bool = True,
        unassigned_region_label: str = "textregion_unassigned",

        # debugging
        log_thresholds: bool = True,
    ):
        self.line_labels = set(line_labels)
        self.region_label = region_label
        self.rebuild_from_scratch = rebuild_from_scratch

        self.column_gap_frac = float(column_gap_frac)
        self.merge_small_columns = bool(merge_small_columns)
        self.min_lines_per_column = int(min_lines_per_column)

        self.correction_attach = bool(correction_attach)
        self.correction_max_dy_factor = float(correction_max_dy_factor)

        self.keep_unassigned_at_page_level = bool(keep_unassigned_at_page_level)
        self.unassigned_region_label = unassigned_region_label

        self.log_thresholds = bool(log_thresholds)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def run(self, collection):
        for page in collection:
            if page.is_leaf():
                continue

            page_bbox: Bbox = page.bbox
            page_w = int(page_bbox.width)

            # 1) collect only line nodes (filters out textregion etc.)
            all_nodes = list(page.traverse())
            line_nodes: list[Node] = []
            removed_non_lines = 0

            for n in all_nodes:
                b = _bbox(n)
                if b is None:
                    continue
                lab = _node_class_label(n)
                if lab in self.line_labels:
                    line_nodes.append(n)
                else:
                    removed_non_lines += 1

            if not line_nodes:
                logger.warning(
                    "MakeRegionsFromLines: no line nodes (%s) found on page %s; skipping",
                    sorted(self.line_labels),
                    page.label,
                )
                continue

            # 2) rebuild from scratch: page children become ONLY lines (removes textregion outputs, etc.)
            if self.rebuild_from_scratch:
                for child in list(page.children):
                    child.detach()
                page.children = []
                for ln in line_nodes:
                    ln.detach()
                    ln.parent = page
                    page.children.append(ln)

            # refresh
            line_nodes = [
                n for n in page.children
                if _node_class_label(n) in self.line_labels and _bbox(n) is not None
            ]

            # Robust line height estimate (for correction attach only)
            heights = [float(_bbox(n).height) for n in line_nodes if _bbox(n) is not None]
            med_h = max(1.0, float(median(heights))) if heights else 20.0

            # 3) column clustering by x-centers, using page-relative gap
            xcs = [_center_x(_bbox(n)) for n in line_nodes]
            gap = max(10.0, self.column_gap_frac * float(page_w))
            col_clusters_idx = _cluster_1d(xcs, gap=gap)
            columns: list[list[Node]] = [[line_nodes[i] for i in idxs] for idxs in col_clusters_idx]

            # merge tiny columns into nearest neighbor by mean x-center
            if self.merge_small_columns and len(columns) > 1:
                col_centers = [sum(_center_x(_bbox(n)) for n in col) / max(1, len(col)) for col in columns]

                changed = True
                while changed:
                    changed = False
                    to_merge = [len(col) < self.min_lines_per_column for col in columns]
                    if not any(to_merge):
                        break

                    for idx, col in enumerate(list(columns)):
                        if not to_merge[idx]:
                            continue
                        candidates = [j for j in range(len(columns)) if not to_merge[j] and j != idx]
                        if not candidates:
                            continue
                        j = min(candidates, key=lambda k: abs(col_centers[k] - col_centers[idx]))
                        columns[j].extend(col)
                        columns[idx] = []
                        changed = True

                    if changed:
                        columns = [c for c in columns if c]
                        col_centers = [sum(_center_x(_bbox(n)) for n in c) / max(1, len(c)) for c in columns]

            # sort columns left->right
            columns = sorted(columns, key=lambda col: sum(_center_x(_bbox(n)) for n in col) / max(1, len(col)))

            # 4) build blocks (regions) within each column using adaptive thresholds
            new_page_children: list[Node] = []
            unassigned_lines: list[Node] = []

            for col_idx, col_lines in enumerate(columns):
                col_lines = _sort_top_down(col_lines)

                # --- adaptive vertical gap threshold (per column) ---
                gaps: list[float] = []
                col_bboxes: list[Bbox] = []
                for a, b in zip(col_lines, col_lines[1:]):
                    ba = _bbox(a)
                    bb = _bbox(b)
                    if ba is None or bb is None:
                        continue
                    col_bboxes.append(ba)
                    gaps.append(float(bb.ymin - ba.ymax))
                # include last bbox for x-adaptation
                if col_lines:
                    last_b = _bbox(col_lines[-1])
                    if last_b is not None:
                        col_bboxes.append(last_b)

                gap_thr = _robust_gap_threshold(gaps)

                # --- adaptive x-overlap requirement (per column) ---
                min_x_overlap = _adaptive_min_x_overlap(col_bboxes, page_w)

                if self.log_thresholds:
                    logger.info(
                        "MakeRegionsFromLines: page=%s col=%d lines=%d gap_thr=%.2fpx min_x_overlap=%.3f",
                        page.label,
                        col_idx,
                        len(col_lines),
                        gap_thr,
                        min_x_overlap,
                    )

                region_groups: list[list[Node]] = []
                cur_group: list[Node] = []
                cur_group_bbox: Bbox | None = None

                for ln in col_lines:
                    b = _bbox(ln)
                    if b is None:
                        unassigned_lines.append(ln)
                        continue

                    if not cur_group:
                        cur_group = [ln]
                        cur_group_bbox = b
                        continue

                    # Vertical decision: compare to learned gap threshold
                    v_gap = float(b.ymin - (cur_group_bbox.ymax if cur_group_bbox else b.ymin))
                    ok_v = (v_gap <= gap_thr) if gap_thr > 0 else True

                    # X decision: only enforce if adaptive min_x_overlap > 0
                    ok_x = True
                    if min_x_overlap > 0.0 and cur_group_bbox is not None:
                        ok_x = _x_overlap_ratio(b, cur_group_bbox) >= min_x_overlap

                    if ok_v and ok_x:
                        cur_group.append(ln)
                        cur_group_bbox = _bbox_union([cur_group_bbox, b]) if cur_group_bbox else b
                        continue

                    # correction/interlineation fallback: attach short inserted lines close in y
                    attached = False
                    if self.correction_attach and cur_group_bbox is not None:
                        dy = min(abs(float(b.ymin - cur_group_bbox.ymax)), abs(float(cur_group_bbox.ymin - b.ymax)))
                        if dy <= self.correction_max_dy_factor * med_h:
                            # for corrections we don't require x overlap (too unreliable)
                            cur_group.append(ln)
                            cur_group_bbox = _bbox_union([cur_group_bbox, b])
                            attached = True

                    if attached:
                        continue

                    # start a new block
                    region_groups.append(cur_group)
                    cur_group = [ln]
                    cur_group_bbox = b

                if cur_group:
                    region_groups.append(cur_group)

                # create region containers
                for block_idx, group in enumerate(region_groups):
                    boxes = [bb for bb in (_bbox(n) for n in group) if bb is not None]
                    if not boxes:
                        unassigned_lines.extend(group)
                        continue

                    rb = _bbox_union(boxes)
                    region = Node(parent=page, label=self.region_label)
                    region.add_data(
                        segmentation_label=self.region_label,
                        segmentation_confidence=1.0,
                        bbox=asdict(rb),
                        polygon=str(rb.polygon()),
                        column_index=col_idx,
                        block_index=block_idx,
                        constructed_from="lines_adaptive",
                        gap_threshold_px=float(gap_thr),
                        min_x_overlap=float(min_x_overlap),
                    )
                    region.children = []
                    for ln in group:
                        ln.parent = region
                        region.children.append(ln)
                    new_page_children.append(region)

            # 5) handle leftovers
            if unassigned_lines:
                if self.keep_unassigned_at_page_level:
                    for ln in unassigned_lines:
                        ln.parent = page
                        new_page_children.append(ln)
                else:
                    boxes = [bb for bb in (_bbox(n) for n in unassigned_lines) if bb is not None]
                    if boxes:
                        rb = _bbox_union(boxes)
                        region = Node(parent=page, label=self.unassigned_region_label)
                        region.add_data(
                            segmentation_label=self.unassigned_region_label,
                            segmentation_confidence=1.0,
                            bbox=asdict(rb),
                            polygon=str(rb.polygon()),
                            column_index=-1,
                            block_index=-1,
                            constructed_from="unassigned_lines",
                        )
                        region.children = []
                        for ln in unassigned_lines:
                            ln.parent = region
                            region.children.append(ln)
                        new_page_children.append(region)
                    else:
                        for ln in unassigned_lines:
                            ln.parent = page
                            new_page_children.append(ln)

            page.children = new_page_children

            logger.info(
                "MakeRegionsFromLines: page=%s kept_lines=%d removed_non_lines=%d columns=%d regions=%d",
                page.label,
                len(line_nodes),
                removed_non_lines,
                len(columns),
                sum(1 for ch in page.children if _node_class_label(ch) == self.region_label),
            )

        collection.relabel()
        return collection




class WordSegmentation(PipelineStep):
    """
    Segment lines into words.

    This step segments lines of text into words. It estimates the word
    boundaries from the recognized text, which means that this step
    must be run after a line-based text recognition model.

    See also `<models.huggingface.trocr.WordLevelTrOCR>`, which is a
    version of TrOCR that outputs word-level text directly using a more
    sophisticated method.

    Example YAML:
    ```yaml
    - step: WordSegmentation
    ```
    """

    def run(self, collection):
        results = simple_word_segmentation(collection.active_leaves())
        collection.update(results)
        return collection

class Export(PipelineStep):
    """
    Export results.

    Exports the current state of the collection in the given format.
    This step is typically the last step of a pipeline, however, it can
    be inserted at any pipeline stage. For example, you could put an
    `Export` step before a post processing step in order to save a copy
    without post processing. A pipeline can include as many `Export`
    steps as you like.

    See [Export formats](export-formats.md) or the `<serialization.serialization>`
    module for more details about each export format.

    Example:
    ```yaml
    - step: Export
      settings:
        format: Alto
        dest: alto-outputs
    ```
    """

    def __init__(
        self,
        dest: str,
        format: Literal["alto", "page", "txt", "json"],
        **serializer_kwargs,
    ):
        """
        Arguments:
            dest: Output directory.
            format: Output format as a string.
        """
        self.serializer = get_serializer(format, **serializer_kwargs)
        self.dest = dest

    def run(self, collection):
        metadata = self.parent_pipeline.metadata() if self.parent_pipeline else None
        save_collection(collection, self.serializer, self.dest, processing_steps=metadata)
        return collection


class ReadingOrderMarginalia(PipelineStep):
    """
    Order regions and lines by reading order.

    This step orders the pages' first- and second-level segments
    (corresponding to regions and lines). Both the regions and their
    lines are ordered using `reading_order.order_regions`.
    """

    def __init__(self, two_page: Literal["auto"] | bool = False):
        """
        Arguments:
            two_page: Whether the page is a two-page spread. Three modes:
                - 'auto': determine heuristically for each page using
                    `layout.is_twopage`
                - True: assume all pages are spreads
                - False: assume all pages are single pages
        """
        self.two_page = two_page

    def is_twopage(self, image):
        if self.two_page == "auto":
            return is_twopage(image)
        return self.two_page

    def run(self, collection):
        for page in collection:
            if page.is_leaf():
                continue

            image = page.image
            printspace = estimate_printspace(image)
            page.children = order_regions(page.children, printspace, self.is_twopage(image))

            for region in page:
                region.children = order_regions(region.children, printspace, is_twopage=False)
        collection.relabel()
        return collection


class OrderLines(PipelineStep):
    """
    Order lines top-down.

    This step orders the lines within each region top-down.

    Example YAML:
    ```yaml
    - step: OrderLines
    ```
    """

    def run(self, collection):
        for page in collection:
            for node in page.traverse():
                if node.is_region():
                    order = top_down([child.bbox for child in node])
                    node.children = [node.children[i] for i in order]
        return collection


class ExportImages(PipelineStep):
    """
    Export the collection's images.

    This step writes all existing images (regions, lines, etc.) in the
    collection to disk. The exported images are the images that have
    been passed to previous `Inference` steps and the images that would
    be passed to a following `Inference` step.

    Example YAML:
    ```yaml
    - step: ExportImages
      settings:
        dest: exported_images
    ```
    """

    def __init__(self, dest: str):
        """
        Arguments:
            dest: Destination directory.
        """
        self.dest = dest
        os.makedirs(self.dest, exist_ok=True)

    def run(self, collection):
        for page in collection:
            directory = os.path.join(self.dest, page.get("image_name"))
            extension = page.get("image_path").split(".")[-1]
            os.makedirs(directory, exist_ok=True)
            for node in page.traverse():
                if node.image is None:
                    continue
                write(os.path.join(directory, f"{node.label}.{extension}"), node.image)
        return collection


class Break(PipelineStep):
    """
    Break the pipeline! Used for testing.

    Example YAML:
    ```yaml
    - step: Break
    ```
    """

    def run(self, collection):
        raise Exception


class Prune(PipelineStep):
    """
    Remove nodes based on a given condition.

    This is a generic pruning (filtering) step which removes nodes
    (segments, lines, words) based on the given condition. The
    condition is a function `f` such that `f(node) == True` if `node`
    should be removed from the tree. This step runs `f` on all nodes,
    at all segmentation levels. See the `RemoveLowTextConfidence[Lines|Regions|Pages]`
    steps for examples of how to formulate `condition`.
    """

    def __init__(self, condition: Callable[[Node], bool]):
        """
        Arguments:
            condition: A function `f` such that `f(node) == True` if
                `node` should be removed from the document tree.
        """
        self.condition = condition

    def run(self, collection):
        for page in collection:
            page.prune(self.condition)
        collection.relabel()
        return collection


class RemoveLowTextConfidenceLines(Prune):
    """
    Remove all lines with text confidence score below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidenceLines
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(lambda node: node.is_line() and metrics.line_text_confidence(node) < threshold)


class RemoveLowTextConfidenceRegions(Prune):
    """
    Remove all regions where the average text confidence score is below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidenceRegions
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(lambda node: node.is_region() and metrics.average_text_confidence(node) < threshold)


class RemoveLowTextConfidencePages(Prune):
    """
    Remove all pages where the average text confidence score is below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidencePages
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(
            lambda node: node.parent and node.parent.is_root() and metrics.average_text_confidence(node) < threshold
        )


class FilterRegionsBySize(Prune):
    """
    Filter regions by size.

    Removes all leaf nodes that are smaller or larger than the given size.

    Example YAML:
    ```yaml
    - step: FilterRegionsBySize
      settings:
        min_height: 10
        min_width: 10
        max_height: 100
        max_width: 100
    ```
    """

    def __init__(
        self, min_height: int = 0, min_width: int = 0, max_height: int | None = None, max_width: int | None = None
    ):
        """
        Arguments:
            min_height: Minimum region height in pixels.
            min_width: Minimum region width in pixels.
            max_height: Maximum region height in pixels.
            max_width: Maximum region width in pixels.
        """
        max_height = max_height or math.inf
        max_width = max_width or math.inf

        super().__init__(
            lambda node: node.is_leaf()
            and not ((min_height < node.height < max_height) and (min_width < node.width < max_width))
        )


class FilterRegionsByShape(Prune):
    """
    Filter regions by shape.

    Removes all leaf nodes that are wider or taller than the given aspect ratio(s).
    For example, if we want to filter out all regions that are more than twice as
    tall as they are wide, we set the `min_ratio` to 0.5 (1:2 width-to-height ratio).

    Example YAML:
    ```yaml
    - step: FilterRegionsByShape
      settings:
        min_ratio: 1
        max_ratio: 10
    ```
    """

    def __init__(self, min_ratio: float = 0.0, max_ratio: float = math.inf):
        """
        Arguments:
            min_ratio: Minimum width-to-height ratio.
            max_ratio: Maximum width-to-height ratio.
        """
        super().__init__(lambda node: node.is_leaf() and not (min_ratio < node.width / node.height < max_ratio))


import logging
from typing import Sequence

import cv2
import numpy as np

from htrflow.volume.node import Node
from htrflow.utils.geometry import Polygon, Bbox

logger = logging.getLogger(__name__)


def _node_class_label(node: Node) -> str | None:
    if hasattr(node, "_segment") and getattr(node._segment, "class_label", None):
        return node._segment.class_label
    if hasattr(node, "_label") and isinstance(node._label, str):
        return node._label
    for k in ("segmentation_label", "class_label", "category", "type"):
        v = node.get(k, None)
        if isinstance(v, str):
            return v
    return None


def _node_score(node: Node) -> float | None:
    if hasattr(node, "_segment") and getattr(node._segment, "score", None) is not None:
        return float(node._segment.score)
    for k in ("segmentation_confidence", "score", "confidence", "conf"):
        v = node.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _node_bbox(node: Node) -> Bbox | None:
    if hasattr(node, "bbox"):
        try:
            b = node.bbox
            if isinstance(b, Bbox):
                return b
        except Exception:
            pass
    return None


def _node_poly_np(node: Node) -> np.ndarray | None:
    if hasattr(node, "polygon"):
        try:
            pol = node.polygon
            if isinstance(pol, Polygon):
                arr = pol.as_nparray()
                if arr is not None and arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                    return arr.astype(np.int32)
        except Exception:
            pass

    b = _node_bbox(node)
    if b is None:
        return None
    x1, y1, x2, y2 = b.xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def _poly_area(poly: np.ndarray) -> float:
    return float(cv2.contourArea(poly.astype(np.float32)))


def _point_in_poly(pt: tuple[float, float], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0


class NestRegionsAndLines(PipelineStep):
    """
    Rebuild nested structure (regions -> lines) from flat multi-class segmentation.

    IMPORTANT: unassigned lines are kept at page-level (parent = PageNode),
    because SegmentNode cropping assumes parent has .image and .coord.
    """

    def __init__(
        self,
        region_labels: Sequence[str] = ("textregion",),
        line_labels: Sequence[str] = ("textline",),
        only_if_flat: bool = True,
        tie_break: str = "smallest_region",  # "smallest_region" or "highest_score"
        treat_unknown_as: str = "line",      # "line" or "ignore"
        min_line_overlap: float = 0.7,       # intersection(line, region)/area(line)
    ):
        self.region_labels = set(region_labels)
        self.line_labels = set(line_labels)
        self.only_if_flat = only_if_flat
        self.tie_break = tie_break
        self.treat_unknown_as = treat_unknown_as
        self.min_line_overlap = float(min_line_overlap)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def run(self, collection):
        for page in collection:
            if page.is_leaf():
                continue

            flat = list(page.children)
            if not flat:
                continue

            if self.only_if_flat and not all(ch.is_leaf() for ch in flat):
                continue

            regions: list[Node] = []
            lines: list[Node] = []
            unknown: list[Node] = []

            for n in flat:
                lab = _node_class_label(n)
                if lab in self.region_labels:
                    regions.append(n)
                elif lab in self.line_labels:
                    lines.append(n)
                else:
                    unknown.append(n)

            if not regions and not lines:
                seen = sorted({str(_node_class_label(n)) for n in flat})
                logger.warning(
                    "NestRegionsAndLines: no nodes matched region_labels=%s or line_labels=%s on page %s; skipping. Seen=%s",
                    sorted(self.region_labels),
                    sorted(self.line_labels),
                    page.label,
                    seen,
                )
                continue

            if unknown and self.treat_unknown_as == "line":
                lines.extend(unknown)

            region_recs = []
            for r in regions:
                r_poly = _node_poly_np(r)
                r_bbox = _node_bbox(r)
                if r_poly is None or r_bbox is None:
                    continue
                region_recs.append(
                    {"node": r, "poly": r_poly, "bbox": r_bbox, "area": _poly_area(r_poly), "score": _node_score(r) or 0.0}
                )

            if not region_recs:
                logger.warning(
                    "NestRegionsAndLines: found region nodes but none had usable polygon/bbox on page %s; skipping.",
                    page.label,
                )
                continue

            assignments: dict[Node, list[Node]] = {rec["node"]: [] for rec in region_recs}
            unassigned: list[Node] = []

            for ln in lines:
                ln_bbox = _node_bbox(ln)
                if ln_bbox is None:
                    unassigned.append(ln)
                    continue

                pt = (float(ln_bbox.center.x), float(ln_bbox.center.y))

                candidates = []
                for rec in region_recs:
                    if not _point_in_poly(pt, rec["poly"]):
                        continue

                    inter = rec["bbox"].intersection(ln_bbox)
                    overlap = 0.0 if inter is None else (inter.area / max(1, ln_bbox.area))
                    if overlap >= self.min_line_overlap:
                        candidates.append(rec)

                if not candidates:
                    unassigned.append(ln)
                    continue

                if self.tie_break == "highest_score":
                    best = max(candidates, key=lambda r: r["score"])
                else:
                    best = min(candidates, key=lambda r: r["area"])

                assignments[best["node"]].append(ln)

            # Rebuild: page children are regions + unassigned lines (directly under page)
            page.children = []

            for r in regions:
                r.parent = page
                r.children = []
                page.children.append(r)

                for ln in assignments.get(r, []):
                    ln.parent = r
                    ln.children = []
                    r.children.append(ln)

            # Keep unassigned lines as page-level children (parent has .image/.coord)
            for ln in unassigned:
                ln.parent = page
                ln.children = []
                page.children.append(ln)

        collection.relabel()
        return collection


class ProcessImages(PipelineStep):
    """
    Base for image preprocessing steps.

    This is a base class for all image preprocessing steps. Subclasses
    define their image processing operation by overriding the `op()`
    method. This step does not alter the original image. Instead, a new
    copy of the image is saved in the directory specified by
    `ProcessImages.output_directory`. The `PageNode`'s image path is
    then updated to point to the new processed image.

    Attributes:
        output_directory: Where to write the processed images.
    """

    output_directory: str

    def run(self, collection):
        for page in collection:
            new_image = self.op(page.image)
            _, image_name = os.path.split(page.path)
            dest = os.path.join("processed_images", collection.label, self.output_directory)
            os.makedirs(dest, exist_ok=True)
            page.path = write(os.path.join(dest, image_name), new_image)
        return collection

    def op(self, image: NumpyImage) -> NumpyImage:
        """
        Perform the image processing operation on `image`.

        Arguments:
            image: Input image.

        Returns:
            A processed version of `image`.
        """
        pass


class Binarization(ProcessImages):
    """
    Binarize images.

    Runs image binarization on the collection's images. Saves the
    resulting images in a directory named `binarized`. All subsequent
    pipeline steps will use the binarized images.

    Example YAML:
    ```yaml
    - step: Binarization
    ```
    """

    output_directory = "binarized"

    def op(self, image):
        return binarize(image)


def auto_import(source: list[str] | str, max_size: int | None = None) -> Generator[Collection, Any, Any]:
    """Import collection(s) from `source`

    Arguments:
        source: Import source as a single path or list of paths, where
            each path points to any of the following:
                - a directory of images
                - an image
        max_size: The maximum number of pages in each new collection.

    Yields:
        Collection instances created from the given source.
    """
    paths = []
    for path in source:
        if os.path.isdir(path):
            files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            paths.extend(files)
            logger.info("Found %d files in input directory '%s'", len(files), path)
            continue
        paths.append(path)

    logger.info("Importing %d input images with batch size %d", len(paths), max_size)
    yield from _create_collection_batches(paths, max_size)


def _create_collection_batches(paths: list[str], max_size: int | None) -> Generator[Collection, Any, Any]:
    """Create and yield collection of at most `max_size` pages"""
    if paths:
        max_size = max_size or len(paths)
        for i in range(0, len(paths), max_size):
            yield Collection(paths[i : i + max_size])


def join_collections(collections: list[Collection]) -> Collection:
    """Create a single `Collection` from the given collections."""
    label = os.path.commonprefix([col.label for col in collections])
    base = collections[0]
    for collection in collections[1:]:
        base.pages.append(collection.pages)
    base.label = label
    return base


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


# Mapping class name -> class
# Ex. {segmentation: `steps.Segmentation`}
STEPS: dict[str, PipelineStep] = {cls_.__name__.lower(): cls_ for cls_ in all_subclasses(PipelineStep)}
MODELS: dict[str, BaseModel] = {model.__name__.lower(): model for model in all_models()}


def init_step(step_name: str, step_settings: dict[str, Any]) -> PipelineStep:
    """Initialize a pipeline step

    Arguments:
        step_name: The name of the pipeline step class. Not case sensitive.
        step_settings: A dictionary containing parameters for the step's
            __init__() method.
    """
    return STEPS[step_name.lower()].from_config(step_settings)
