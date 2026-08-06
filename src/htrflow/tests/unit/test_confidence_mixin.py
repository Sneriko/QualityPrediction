from types import SimpleNamespace

import pytest
import torch

from htrflow.models.huggingface.mixins import ConfidenceMixin


class CharacterTokenizer:
    characters = {1: "", 2: "a", 3: " ", 4: "b", 5: ""}

    def decode(self, token_ids, skip_special_tokens=False):
        assert isinstance(token_ids, list)
        assert len(token_ids) == 1
        return self.characters[token_ids[0]]


class CharacterConfidenceModel(ConfidenceMixin):
    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=CharacterTokenizer())

    def _compute_transition_scores(self, outputs):
        return torch.log(torch.tensor([[0.9, 0.8, 0.7, 1.0]]))


def test_confidence_scores_are_aligned_with_generated_character_tokens():
    outputs = SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 4, 5]]))

    scores = CharacterConfidenceModel().compute_confidence_per_token(outputs)

    assert [token for token, _ in scores[0]] == ["a", " ", "b"]
    assert [score for _, score in scores[0]] == pytest.approx([0.9, 0.8, 0.7])
