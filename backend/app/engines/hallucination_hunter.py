"""
HallucinationHunter: Quantifies factual drift between LLM output and ground truth
using multiple lexical and statistical scoring methods.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Dict, List, Tuple


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _precision_recall_f1(
    response_tokens: List[str], truth_tokens: List[str]
) -> Tuple[float, float, float]:
    """Token-level precision, recall, and F1 (micro)."""
    resp_c = Counter(response_tokens)
    truth_c = Counter(truth_tokens)

    common = sum((resp_c & truth_c).values())
    if not response_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0

    precision = common / len(response_tokens)
    recall = common / len(truth_tokens)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


def _bigram_overlap(response_tokens: List[str], truth_tokens: List[str]) -> float:
    """Bigram overlap ratio (similar to ROUGE-2 recall)."""
    resp_bg = _ngrams(response_tokens, 2)
    truth_bg = _ngrams(truth_tokens, 2)
    if not truth_bg:
        return 1.0 if not resp_bg else 0.0
    common = sum((resp_bg & truth_bg).values())
    return common / sum(truth_bg.values())


def _novel_token_ratio(response_tokens: List[str], truth_tokens: List[str]) -> float:
    """Fraction of response tokens not found in the ground truth vocabulary."""
    if not response_tokens:
        return 0.0
    truth_vocab = set(truth_tokens)
    novel = [t for t in response_tokens if t not in truth_vocab]
    return len(novel) / len(response_tokens)


def _length_ratio(response_tokens: List[str], truth_tokens: List[str]) -> float:
    """Response/truth length ratio, capped at a penalty in [0, 1]."""
    if not truth_tokens:
        return 0.0
    ratio = len(response_tokens) / len(truth_tokens)
    # Penalise very short (<0.5x) or very long (>2x) responses
    if 0.5 <= ratio <= 2.0:
        return 1.0
    return max(0.0, 1.0 - abs(ratio - 1.0) / 4.0)


class HallucinationHunter:
    """
    Quantifies factual drift between an LLM response and a ground-truth string.

    Scoring is a weighted combination of:
      - Token F1          (token-level recall/precision balance)
      - Jaccard similarity (unigram set overlap)
      - Bigram overlap    (consecutive word-pair coverage)
      - Novel token ratio (fraction of response words absent from ground truth)
      - Length ratio      (response length appropriateness)

    The final ``drift_score`` is in [0, 1].  A score of 0 means the response
    is identical to the ground truth; a score of 1 means total divergence.
    ``hallucination_risk`` is a categorical label: "low", "medium", or "high".
    """

    # Thresholds for the categorical risk label
    RISK_LOW = 0.35
    RISK_MEDIUM = 0.65

    def __init__(
        self,
        weight_f1: float = 0.35,
        weight_jaccard: float = 0.25,
        weight_bigram: float = 0.20,
        weight_novel: float = 0.15,
        weight_length: float = 0.05,
    ) -> None:
        total = weight_f1 + weight_jaccard + weight_bigram + weight_novel + weight_length
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        self.w_f1 = weight_f1
        self.w_jaccard = weight_jaccard
        self.w_bigram = weight_bigram
        self.w_novel = weight_novel
        self.w_length = weight_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, response: str, ground_truth: str) -> Dict:
        """
        Analyse a single LLM response against a ground-truth string.

        Parameters
        ----------
        response:     The text produced by the LLM.
        ground_truth: The expected / reference answer.

        Returns
        -------
        A dict with the following keys:
          drift_score         – float in [0, 1]; higher = more drift / hallucination
          hallucination_risk  – "low" | "medium" | "high"
          token_f1            – float in [0, 1]
          jaccard_similarity  – float in [0, 1]
          bigram_overlap      – float in [0, 1]
          novel_token_ratio   – float in [0, 1]
          length_ratio        – float in [0, 1]
          response_tokens     – int
          truth_tokens        – int
        """
        resp_tok = _tokenize(response)
        truth_tok = _tokenize(ground_truth)

        precision, recall, f1 = _precision_recall_f1(resp_tok, truth_tok)
        jaccard = _jaccard(set(resp_tok), set(truth_tok))
        bigram = _bigram_overlap(resp_tok, truth_tok)
        novel = _novel_token_ratio(resp_tok, truth_tok)
        length = _length_ratio(resp_tok, truth_tok)

        # Similarity composite (higher = more similar = less drift)
        similarity = (
            self.w_f1 * f1
            + self.w_jaccard * jaccard
            + self.w_bigram * bigram
            + self.w_length * length
            # novel ratio contributes negatively (more novel = less similar)
            - self.w_novel * novel
        )
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        drift_score = round(1.0 - similarity, 4)

        risk = self._risk_label(drift_score)

        return {
            "drift_score": drift_score,
            "hallucination_risk": risk,
            "token_f1": round(f1, 4),
            "jaccard_similarity": round(jaccard, 4),
            "bigram_overlap": round(bigram, 4),
            "novel_token_ratio": round(novel, 4),
            "length_ratio": round(length, 4),
            "response_tokens": len(resp_tok),
            "truth_tokens": len(truth_tok),
        }

    def batch_analyze(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Analyse a list of (response, ground_truth) pairs.

        Parameters
        ----------
        pairs: list of (response, ground_truth) 2-tuples

        Returns
        -------
        List of result dicts in the same order as the input.
        """
        return [self.analyze(resp, gt) for resp, gt in pairs]

    def summary(self, results: List[Dict]) -> Dict:
        """
        Compute aggregate statistics over a list of analyze() results.

        Returns
        -------
        Dict with mean/min/max drift_score and per-risk counts.
        """
        if not results:
            return {}

        scores = [r["drift_score"] for r in results]
        risk_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for r in results:
            risk_counts[r["hallucination_risk"]] += 1

        return {
            "n": len(results),
            "mean_drift_score": round(sum(scores) / len(scores), 4),
            "min_drift_score": round(min(scores), 4),
            "max_drift_score": round(max(scores), 4),
            "risk_counts": risk_counts,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _risk_label(self, drift_score: float) -> str:
        if drift_score < self.RISK_LOW:
            return "low"
        if drift_score < self.RISK_MEDIUM:
            return "medium"
        return "high"
