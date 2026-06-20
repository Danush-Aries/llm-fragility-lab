# LLM Fragility Lab

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)

A research toolkit for quantifying LLM hallucinations through multi-metric semantic drift analysis. Given an LLM response and a ground-truth string, **LLM Fragility Lab** produces a `drift_score` in [0, 1] and a categorical risk label (`low` / `medium` / `high`) by combining token-level F1, Jaccard similarity, bigram overlap, novel-token ratio, and length appropriateness.

---

## What it does

Large language models occasionally "hallucinate" — generating plausible-sounding text that deviates from factual ground truth. LLM Fragility Lab gives you a lightweight, dependency-free way to measure that deviation.

The core `HallucinationHunter` class computes five complementary signals and blends them into a single drift score:

| Signal | Description |
|---|---|
| Token F1 | Harmonic mean of token-level precision and recall |
| Jaccard similarity | Unigram set overlap between response and ground truth |
| Bigram overlap | Consecutive two-word pair coverage (similar to ROUGE-2) |
| Novel token ratio | Fraction of response words absent from the ground truth vocabulary |
| Length ratio | Penalty for responses that are far shorter or longer than expected |

A `drift_score` near **0** means the response closely matches the ground truth. A score near **1** indicates high divergence and likely hallucination.

---

## Features

- Zero third-party runtime dependencies — uses only the Python standard library
- Single-sample analysis with a rich result dictionary
- Batch analysis over lists of (response, ground_truth) pairs
- Aggregate summary statistics over a batch
- Configurable signal weights to tune sensitivity for your use case
- CLI entry point with demo, single, batch, and JSON output modes
- 29-test suite covering helpers, analysis, batch, summary, and weight validation

---

## Installation

```bash
git clone https://github.com/Dhanush-Aries/llm-fragility-lab.git
cd llm-fragility-lab
pip install -r requirements.txt
```

Python 3.9 or later is required. No external packages are needed at runtime; `numpy` and `pytest` are listed for optional numerical extensions and testing.

---

## Usage

### Run the built-in demo

```bash
python3 main.py
```

### Analyse a single response

```bash
python3 main.py \
  --response "The Eiffel Tower was built in 1750 by Napoleon." \
  --truth    "The Eiffel Tower was completed in 1889 by Gustave Eiffel."
```

### Batch evaluation from a JSON file

```bash
python3 main.py --batch samples.json
```

The JSON file must be an array of objects with `response` and `ground_truth` keys (see `samples.json` for an example).

### JSON output (pipe-friendly)

```bash
python3 main.py --batch samples.json --json | jq '.summary'
```

---

## Python API

```python
from backend.app.engines.hallucination_hunter import HallucinationHunter

hunter = HallucinationHunter()

# Single analysis
result = hunter.analyze(
    response="The Eiffel Tower was built in 1750 by Napoleon.",
    ground_truth="The Eiffel Tower was completed in 1889 by Gustave Eiffel.",
)
print(result)
# {
#   'drift_score': 0.7143,
#   'hallucination_risk': 'high',
#   'token_f1': 0.5455,
#   'jaccard_similarity': 0.375,
#   'bigram_overlap': 0.2857,
#   'novel_token_ratio': 0.25,
#   'length_ratio': 1.0,
#   'response_tokens': 9,
#   'truth_tokens': 10
# }

# Batch analysis
pairs = [
    ("Paris is in France.", "Paris is the capital of France."),
    ("Water boils at 50 C.", "Water boils at 100 degrees Celsius."),
]
results = hunter.batch_analyze(pairs)
summary = hunter.summary(results)
print(summary)
# {'n': 2, 'mean_drift_score': ..., 'min_drift_score': ...,
#  'max_drift_score': ..., 'risk_counts': {'low': 1, 'medium': 1, 'high': 0}}
```

### Custom weights

```python
hunter = HallucinationHunter(
    weight_f1=0.40,
    weight_jaccard=0.30,
    weight_bigram=0.15,
    weight_novel=0.10,
    weight_length=0.05,  # weights must sum to 1.0
)
```

---

## Running the tests

```bash
python3 -m pytest tests/ -v
```

---

## Project structure

```
llm-fragility-lab/
├── backend/
│   └── app/
│       └── engines/
│           ├── __init__.py
│           └── hallucination_hunter.py   # core scoring engine
├── tests/
│   └── test_hallucination_hunter.py      # 29 unit tests
├── main.py                               # CLI entry point
├── samples.json                          # example batch input
├── requirements.txt
└── README.md
```

---

## Tech stack

- **Language**: Python 3.9+
- **Core dependencies**: Python standard library only (`re`, `math`, `collections`)
- **Optional**: `numpy>=1.24` (available for future vector-based extensions)
- **Testing**: `pytest>=7.4`

---

## License

MIT License. See [LICENSE](LICENSE) for details.
