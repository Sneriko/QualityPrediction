from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import List, Optional
from dataclasses import dataclass
from typing import Tuple

from quality_prediction.core.geometry import BBox
from quality_prediction.core.types import PageContent, TextLine

@dataclass
class LineGeom:
    bbox: Optional[BBox]
    polygon: str  # PAGE "points" string, empty for ALTO


class PageXmlParser:
    """Loads GT lines from PAGE XML or ALTO XML, preserving XML order."""

    # -----------------------------
    # Public API
    # -----------------------------
    def parse_line_geoms(self, xml_path: str) -> List[LineGeom]:
        """
        For PAGE XML: returns bbox + polygon points string for each TextLine (Coords/@points).
        For ALTO: returns bbox (union over Strings) and polygon="" (fallback to bbox IoU).
        Preserves XML order like parse().
        """
        root = self._load_root(xml_path)
        if "alto" in root.tag.lower():
            return self._parse_alto_line_geoms(root)
        return self._parse_page_line_geoms(root)

    def _parse_page_line_geoms(self, root: ET.Element) -> List[LineGeom]:
        out: List[LineGeom] = []
        for tl in root.findall(".//{*}TextLine"):
            bbox: Optional[BBox] = None
            poly = ""

            coords_el = tl.find(".//{*}Coords")
            if coords_el is not None:
                points_str = coords_el.get("points") or ""
                poly = points_str.strip()
                if poly:
                    bbox = BBox.from_page_coords(poly)

            out.append(LineGeom(bbox=bbox, polygon=poly))
        return out

    def _parse_alto_line_geoms(self, root: ET.Element) -> List[LineGeom]:
        out: List[LineGeom] = []
        for tl in root.findall(".//{*}TextLine"):
            string_elems = tl.findall(".//{*}String")
            if not string_elems:
                continue
            bbox = self._alto_union_bbox(string_elems)
            out.append(LineGeom(bbox=bbox, polygon=""))  # ALTO has no line polygon here
        return out

    def parse_region_bboxes(self, xml_path: str) -> List[BBox]:
        root = self._load_root(xml_path)
        tag_lower = root.tag.lower()

        # ----- ALTO -----
        if "alto" in tag_lower:
            boxes: List[BBox] = []

            # Some ALTO variants can use ComposedBlock as well; keep TextBlock as primary.
            for tag in ("TextBlock", "ComposedBlock"):
                for blk in root.findall(f".//{{*}}{tag}"):
                    try:
                        x = float(blk.get("HPOS"))
                        y = float(blk.get("VPOS"))
                        w = float(blk.get("WIDTH"))
                        h = float(blk.get("HEIGHT"))
                        boxes.append(BBox(x, y, x + w, y + h))
                    except (TypeError, ValueError):
                        continue

            return boxes

        # ----- PAGE -----
        boxes: List[BBox] = []
        for tr in root.findall(".//{*}TextRegion"):
            coords_el = tr.find(".//{*}Coords")
            if coords_el is None:
                continue
            points_str = coords_el.get("points")
            if not points_str:
                continue
            boxes.append(BBox.from_page_coords(points_str))
        return boxes

    def parse(self, xml_path: str) -> PageContent:
        root = self._load_root(xml_path)
        if "alto" in root.tag.lower():
            return self._parse_alto(root)
        return self._parse_page_xml(root)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _load_root(self, xml_path: str) -> ET.Element:
        try:
            tree = ET.parse(xml_path)
        except ParseError as e:
            raise ParseError(f"Failed to parse XML '{xml_path}': {e}")
        return tree.getroot()

    # -----------------------------
    # PAGE parsing
    # -----------------------------
    def _parse_page_xml(self, root: ET.Element) -> PageContent:
        out: List[TextLine] = []
        for tl in root.findall(".//{*}TextLine"):
            text = self._extract_page_textline_text(tl)
            bbox: Optional[BBox] = None

            coords_el = tl.find(".//{*}Coords")
            if coords_el is not None:
                points_str = coords_el.get("points")
                if points_str:
                    bbox = BBox.from_page_coords(points_str)

            out.append(TextLine(text=text, bbox=bbox))
        return PageContent(lines=out)

    def _extract_page_textline_text(self, tl: ET.Element) -> str:
        best_text = None
        best_conf = -1.0

        # Prefer the TextEquiv with the highest conf, if present.
        for te in tl.findall("./{*}TextEquiv"):
            unicode_el = te.find("./{*}Unicode")
            if unicode_el is None or unicode_el.text is None:
                continue
            conf_attr = te.get("conf")
            try:
                conf = float(conf_attr) if conf_attr is not None else 1.0
            except ValueError:
                conf = 1.0
            if conf > best_conf:
                best_conf = conf
                best_text = unicode_el.text

        if best_text is not None:
            return best_text.strip()

        # Fallback: join first TextEquiv per Word
        word_texts = []
        for w in tl.findall(".//{*}Word"):
            te = w.findall("./{*}TextEquiv")
            if not te:
                continue
            unicode_el = te[0].find("./{*}Unicode")
            if unicode_el is not None and unicode_el.text:
                word_texts.append(unicode_el.text)
        if word_texts:
            return " ".join(word_texts).strip()

        # Final fallback: any Unicode descendant
        unicode_el = tl.find(".//{*}Unicode")
        if unicode_el is not None and unicode_el.text is not None:
            return unicode_el.text.strip()
        return ""

    # -----------------------------
    # ALTO parsing
    # -----------------------------
    def _parse_alto(self, root: ET.Element) -> PageContent:
        out: List[TextLine] = []
        for tl in root.findall(".//{*}TextLine"):
            string_elems = tl.findall(".//{*}String")
            if not string_elems:
                continue

            # Text: prefer SUBS_CONTENT when present (helps hyphenated words)
            words: List[str] = []
            for se in string_elems:
                token = (se.get("SUBS_CONTENT") or se.get("CONTENT") or "").strip()
                if token:
                    words.append(token)

            line_text = " ".join(words).strip()
            if not line_text:
                continue

            # BBox: union over all String boxes (more robust than first+last)
            bbox = self._alto_union_bbox(string_elems)

            out.append(TextLine(text=line_text, bbox=bbox))

        return PageContent(lines=out)

    def _alto_union_bbox(self, string_elems: List[ET.Element]) -> Optional[BBox]:
        xs1: List[float] = []
        ys1: List[float] = []
        xs2: List[float] = []
        ys2: List[float] = []

        for se in string_elems:
            try:
                x = float(se.get("HPOS"))
                y = float(se.get("VPOS"))
                w = float(se.get("WIDTH"))
                h = float(se.get("HEIGHT"))
            except (TypeError, ValueError):
                continue

            xs1.append(x)
            ys1.append(y)
            xs2.append(x + w)
            ys2.append(y + h)

        if not xs1:
            return None

        return BBox(min(xs1), min(ys1), max(xs2), max(ys2))
