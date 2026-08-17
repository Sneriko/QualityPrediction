import json

from quality_prediction.io.htr_json import (
    HtrJsonDetectionsParser,
    HtrJsonParser,
    load_page_document,
)


def _write_page(tmp_path, line_polygon):
    page = {
        "polygon": "0,0 1000,0 1000,800 0,800",
        "regions": [
            {
                "polygon": "100,200 500,200 500,600 100,600",
                "annotations": {"segmentation_label": "textregion"},
                "regions": [
                    {
                        "polygon": line_polygon,
                        "annotations": {"segmentation_label": "textline"},
                        "transcription": [{"text": "test", "confidence": 0.9}],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "page.json"
    path.write_text(json.dumps(page), encoding="utf-8")
    return path


def test_crop_local_lines_are_converted_to_page_coordinates(tmp_path):
    path = _write_page(tmp_path, "10,20 390,20 390,60 10,60")

    page = load_page_document(str(path))
    parsed_line = HtrJsonParser().parse(str(path)).lines[0]
    _, detected_lines = HtrJsonDetectionsParser().parse(str(path))

    assert page.all_lines[0].bbox == parsed_line.bbox == detected_lines[0].bbox
    assert detected_lines[0].bbox.xmin == 110
    assert detected_lines[0].bbox.ymin == 220
    assert detected_lines[0].polygon.startswith("110,220 ")


def test_absolute_nested_lines_are_not_translated_twice(tmp_path):
    path = _write_page(tmp_path, "110,220 490,220 490,260 110,260")

    page = load_page_document(str(path))
    _, detected_lines = HtrJsonDetectionsParser().parse(str(path))

    assert page.all_lines[0].bbox == detected_lines[0].bbox
    assert detected_lines[0].bbox.xmin == 110
    assert detected_lines[0].bbox.ymin == 220
