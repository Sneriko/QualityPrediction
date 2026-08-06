from __future__ import annotations

import math
import re
from typing import Dict, List, Protocol, Set, Tuple

from ngram.ngrammodel import NgramModel  # your existing module


class NgramResource:
    def __init__(self, ngram_sets: Dict[int, Set[str]]):
        self.ngram_sets = ngram_sets

    def ratio_present(self, text: str, n: int) -> float:
        text = " ".join(text.replace("\n", " ").split())
        allowed = self.ngram_sets.get(n)
        if not allowed:
            return float("nan")
        grams = [text[i : i + n] for i in range(len(text) - n + 1)]
        if not grams:
            return float("nan")
        present = sum(1 for g in grams if g in allowed)
        return present / len(grams)


class LMPerplexityScorer(Protocol):
    def page_ppl(self, text: str) -> float: ...
    def line_ppl(self, line: str) -> float: ...


class NgramLMPerplexityScorer:
    def __init__(self, model: NgramModel, smoothing_k: float = 1.0):
        self.model = model
        self.smoothing_k = smoothing_k
        self.n = model.n
        self.level = model.level
        self._token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        if self.level == "char":
            return list(text)
        return self._token_pattern.findall(text)

    def _sequence_logprob_and_length(self, text: str) -> Tuple[float, int]:
        tokens = [t for t in self._tokenize(text) if t.strip()]
        if not tokens:
            return 0.0, 0

        if self.n == 1:
            logp = 0.0
            for tok in tokens:
                p = self.model.prob((), tok, k=self.smoothing_k)
                logp += math.log(max(p, 1e-12))
            return logp, len(tokens)

        if len(tokens) < self.n:
            return 0.0, 0

        logp = 0.0
        count = 0
        for i in range(len(tokens) - self.n + 1):
            ctx = tuple(tokens[i : i + self.n - 1])
            tok = tokens[i + self.n - 1]
            p = self.model.prob(ctx, tok, k=self.smoothing_k)
            logp += math.log(max(p, 1e-12))
            count += 1
        return logp, count

    def _perplexity(self, logp: float, length: int) -> float:
        if length == 0:
            return float("nan")
        return float(math.exp(-logp / float(length)))

    def page_ppl(self, text: str) -> float:
        text = text.strip()
        if not text:
            return float("nan")
        logp, length = self._sequence_logprob_and_length(text)
        return self._perplexity(logp, length)

    def line_ppl(self, line: str) -> float:
        line = line.strip()
        if not line:
            return float("nan")
        logp, length = self._sequence_logprob_and_length(line)
        return self._perplexity(logp, length)


class Lexicon:
    def __init__(self, words: Set[str]):
        self.words = words

    def contains(self, w: str) -> bool:
        return w in self.words
