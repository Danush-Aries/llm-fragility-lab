from backend.app.engines.hallucination_hunter import HallucinationHunter
import pytest

def test_hallucination_hunter():
    hunter = HallucinationHunter()
    res = hunter.analyze("this is a test", "this is a test")
    assert res['drift_score'] == 0.0

def test_hallucination_hunter_empty_response():
    hunter = HallucinationHunter()
    res = hunter.analyze("", "this is a test")
    assert res['drift_score'] == 1.0
