import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HTRFLOW_SRC = Path(__file__).parents[1] / "src" / "htrflow" / "src"
sys.path.insert(0, str(HTRFLOW_SRC))

from htrflow.document import Region, Text  # noqa: E402
from htrflow.pipeline.steps import QualityPrediction  # noqa: E402
from htrflow.serialization import get_serializer  # noqa: E402
from htrflow.utils.geometry import Polygon  # noqa: E402
from quality_prediction.inference import (  # noqa: E402
    XGBoostQualityPredictor,
    page_document_from_htrflow,
)


def _polygon(xmin, ymin, xmax, ymax):
    return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])


def _document():
    document = SimpleNamespace(
        image_name="page",
        _image_path="page.jpg",
        polygon=_polygon(0, 0, 1000, 800),
        regions=[],
        transcription=[],
        annotations={},
    )
    region = Region(
        _polygon(100, 200, 500, 600),
        segmentation_label="textregion",
        segmentation_confidence=0.8,
    )
    line = Region(
        _polygon(10, 20, 390, 60),
        transcription=[
            Text("recognized text", confidence=0.9, token_scores=[("recognized", 0.7)])
        ],
        segmentation_label="textline",
        segmentation_confidence=0.75,
    )
    line.attach(region)
    region.attach(document)
    return document


def test_adapter_reads_document_tree_and_translates_crop_coordinates():
    page = page_document_from_htrflow(_document())

    assert page.regions[0].segmentation_confidence == 0.8
    assert page.all_lines[0].full_text == "recognized text"
    assert page.all_lines[0].bbox.xmin == 110
    assert page.all_lines[0].bbox.ymin == 220
    assert page.all_lines[0].token_confidences == [0.7]


def test_adapter_does_not_translate_absolute_coordinates_twice():
    document = _document()
    document.regions[0].regions[0].polygon = _polygon(110, 220, 490, 260)

    page = page_document_from_htrflow(document)

    assert page.all_lines[0].bbox.xmin == 110
    assert page.all_lines[0].bbox.ymin == 220


def test_step_annotation_is_included_in_json_export():
    document = _document()
    step = QualityPrediction.__new__(QualityPrediction)
    step.target = "target_bow_f1"
    step.predictor = SimpleNamespace(predict=lambda current: 0.625)

    returned = step.run(document)
    output = json.loads(get_serializer("json").serialize(document))

    assert returned is document
    assert output["annotations"]["quality_prediction"]["target_bow_f1"] == 0.625


def test_predictor_reads_numpy_feature_names():
    predictor = XGBoostQualityPredictor.__new__(XGBoostQualityPredictor)
    predictor.model = SimpleNamespace(feature_names_in_=np.array(["first", "second"]))

    assert predictor._model_feature_names() == ["first", "second"]
