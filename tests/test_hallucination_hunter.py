"""
Unit tests for HallucinationHunter.
Run with: python -m pytest tests/ -v
"""

import pytest
from backend.app.engines.hallucination_hunter import (
    HallucinationHunter,
    _normalize,
    _tokenize,
    _jaccard,
    _bigram_overlap,
    _novel_token_ratio,
    _precision_recall_f1,
)


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize_lowercases(self):
        assert _normalize("Hello, World!") == "hello world"

    def test_normalize_strips_punctuation(self):
        assert _normalize("it's a test.") == "its a test"

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_novel_token_ratio_no_overlap(self):
        assert _novel_token_ratio(["foo", "bar"], ["baz"]) == 1.0

    def test_novel_token_ratio_full_overlap(self):
        assert _novel_token_ratio(["foo", "bar"], ["foo", "bar"]) == 0.0

    def test_bigram_overlap_identical(self):
        tokens = ["the", "quick", "brown", "fox"]
        assert _bigram_overlap(tokens, tokens) == 1.0

    def test_precision_recall_f1_identical(self):
        t = ["the", "cat", "sat"]
        p, r, f = _precision_recall_f1(t, t)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    def test_precision_recall_f1_empty_response(self):
        p, r, f = _precision_recall_f1([], ["some", "tokens"])
        assert p == 0.0 and r == 0.0 and f == 0.0


# ---------------------------------------------------------------------------
# HallucinationHunter.analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    @pytest.fixture
    def hunter(self):
        return HallucinationHunter()

    def test_identical_text_low_drift(self, hunter):
        text = "The capital of France is Paris."
        result = hunter.analyze(text, text)
        assert result["drift_score"] < 0.2
        assert result["hallucination_risk"] == "low"

    def test_completely_different_text_high_drift(self, hunter):
        result = hunter.analyze(
            "The moon is made of purple marshmallows orbiting Jupiter.",
            "The capital of France is Paris, located along the Seine river.",
        )
        assert result["drift_score"] > 0.5

    def test_result_keys_present(self, hunter):
        result = hunter.analyze("some response", "some ground truth")
        expected_keys = {
            "drift_score", "hallucination_risk",
            "token_f1", "jaccard_similarity", "bigram_overlap",
            "novel_token_ratio", "length_ratio",
            "response_tokens", "truth_tokens",
        }
        assert expected_keys == set(result.keys())

    def test_drift_score_range(self, hunter):
        result = hunter.analyze("foo bar baz", "alpha beta gamma delta")
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_risk_label_values(self, hunter):
        result = hunter.analyze("test", "test")
        assert result["hallucination_risk"] in ("low", "medium", "high")

    def test_empty_strings(self, hunter):
        result = hunter.analyze("", "")
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_token_counts(self, hunter):
        result = hunter.analyze("one two three", "alpha beta")
        assert result["response_tokens"] == 3
        assert result["truth_tokens"] == 2

    def test_partial_overlap_medium_drift(self, hunter):
        result = hunter.analyze(
            "Paris is the capital and largest city of France.",
            "Paris is the capital of France and a major European metropolis.",
        )
        assert result["drift_score"] < 0.6

    def test_hallucinated_content_raises_risk(self, hunter):
        result = hunter.analyze(
            "The Eiffel Tower was built in 1750 by Napoleon Bonaparte.",
            "The Eiffel Tower was completed in 1889 by engineer Gustave Eiffel.",
        )
        assert result["drift_score"] > 0.2


# ---------------------------------------------------------------------------
# HallucinationHunter.batch_analyze
# ---------------------------------------------------------------------------

class TestBatchAnalyze:
    @pytest.fixture
    def hunter(self):
        return HallucinationHunter()

    def test_returns_correct_count(self, hunter):
        pairs = [
            ("response one", "truth one"),
            ("response two", "truth two"),
            ("response three", "truth three"),
        ]
        results = hunter.batch_analyze(pairs)
        assert len(results) == 3

    def test_empty_batch(self, hunter):
        assert hunter.batch_analyze([]) == []

    def test_each_result_has_drift_score(self, hunter):
        pairs = [("a b c", "a b"), ("x y z", "a b c")]
        for result in hunter.batch_analyze(pairs):
            assert "drift_score" in result


# ---------------------------------------------------------------------------
# HallucinationHunter.summary
# ---------------------------------------------------------------------------

class TestSummary:
    @pytest.fixture
    def hunter(self):
        return HallucinationHunter()

    def test_summary_keys(self, hunter):
        results = hunter.batch_analyze([("a b", "a b"), ("x y z", "a b c")])
        s = hunter.summary(results)
        assert "n" in s
        assert "mean_drift_score" in s
        assert "risk_counts" in s

    def test_summary_n(self, hunter):
        pairs = [("a", "a")] * 5
        results = hunter.batch_analyze(pairs)
        assert hunter.summary(results)["n"] == 5

    def test_empty_summary(self, hunter):
        assert hunter.summary([]) == {}

    def test_risk_counts_sum(self, hunter):
        pairs = [("a b", "a b"), ("x y z", "p q r"), ("hello world", "hello world")]
        results = hunter.batch_analyze(pairs)
        s = hunter.summary(results)
        rc = s["risk_counts"]
        assert rc["low"] + rc["medium"] + rc["high"] == len(pairs)


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            HallucinationHunter(
                weight_f1=0.5,
                weight_jaccard=0.5,
                weight_bigram=0.5,
                weight_novel=0.5,
                weight_length=0.5,
            )

    def test_custom_valid_weights(self):
        h = HallucinationHunter(
            weight_f1=0.4,
            weight_jaccard=0.3,
            weight_bigram=0.15,
            weight_novel=0.10,
            weight_length=0.05,
        )
        result = h.analyze("Paris is in France", "Paris is the capital of France")
        assert 0.0 <= result["drift_score"] <= 1.0
