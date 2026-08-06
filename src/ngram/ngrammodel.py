# ngram_model.py
from collections import Counter, defaultdict
from typing import Dict, Tuple, Iterable, List, Literal
import re
import pickle


Level = Literal["word", "char"]


class NgramModel:
    """
    Simple maximum-likelihood n-gram model with add-k smoothing.
    
    Stores:
        context_counts[context][token] = count
        total_context_counts[context] = total count of tokens after that context
    """
    def __init__(self, n: int, level: Level = "word") -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.level: Level = level
        self.context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.total_context_counts: Counter = Counter()
        self._token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        if self.level == "char":
            return list(text)
        # word-level: keep punctuation as separate tokens
        return self._token_pattern.findall(text)

    def update(self, text: str) -> None:
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return

        # For n == 1, context is empty tuple
        if self.n == 1:
            context = ()
            for tok in tokens:
                self.context_counts[context][tok] += 1
                self.total_context_counts[context] += 1
            return

        if len(tokens) < self.n:
            return

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            tok = tokens[i + self.n - 1]
            self.context_counts[context][tok] += 1
            self.total_context_counts[context] += 1

    def vocab(self) -> Iterable[str]:
        seen = set()
        for ctx, counter in self.context_counts.items():
            for tok in counter.keys():
                if tok not in seen:
                    seen.add(tok)
                    yield tok

    def prob(self, context: Tuple[str, ...], token: str, k: float = 1.0) -> float:
        """
        P(token | context) with add-k (Laplace when k=1.0) smoothing.
        For n == 1, context must be an empty tuple.
        """
        ctx = tuple(context)
        counts = self.context_counts.get(ctx, Counter())
        count_ct = counts.get(token, 0)
        total_ct = self.total_context_counts.get(ctx, 0)

        # Vocabulary size for this context (backoff-style; you could also use global vocab)
        vocab_size = len(counts) if counts else 1
        return (count_ct + k) / (total_ct + k * vocab_size)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "level": self.level,
                    "context_counts": self.context_counts,
                    "total_context_counts": self.total_context_counts,
                },
                f,
            )

    @classmethod
    def load(cls, path) -> "NgramModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["n"], data["level"])
        model.context_counts = data["context_counts"]
        model.total_context_counts = data["total_context_counts"]
        return model
