# LLM Fragility Lab — Measure how much your model is making things up

> **A zero-dependency toolkit that turns any (response, ground-truth) pair into a drift score in [0, 1] and a categorical hallucination risk — using five complementary lexical signals blended into one number.**

<p align="center"><img src="assets/hero.gif" alt="LLM Fragility Lab drift score demo" width="720"></p>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/Danush-Aries/llm-fragility-lab/ci.yml?branch=main&style=flat-square" alt="build">
  <img src="https://img.shields.io/badge/license-MIT-00ff41?style=flat-square" alt="license">
  <img src="https://img.shields.io/badge/made%20with-Python%203.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/tests-29%20passing-brightgreen?style=flat-square" alt="tests">
  <img src="https://img.shields.io/badge/runtime%20deps-0-informational?style=flat-square" alt="deps">
</p>

## Why this exists

You want to catch a hallucination the moment it happens, but every off-the-shelf "LLM eval" package pulls in `transformers`, `torch`, and 4 GB of weights just to run cosine on some embeddings. LLM Fragility Lab does the opposite: pure standard library, five cheap lexical signals (token F1, Jaccard, bigram overlap, novel-token ratio, length penalty), a single blended `drift_score`, and a categorical `low / medium / high` risk label. Runs on a Raspberry Pi. Fits in a CI check that adds 200 ms to your test suite.

## Try it in 60 seconds

```bash
git clone https://github.com/Danush-Aries/llm-fragility-lab.git
cd llm-fragility-lab
pip install -r requirements.txt         # only pytest — no runtime deps

python3 main.py                         # built-in demo
python3 main.py --batch samples.json    # score a batch
python3 main.py \
  --response "The Eiffel Tower was built in 1750 by Napoleon." \
  --truth    "The Eiffel Tower was completed in 1889 by Gustave Eiffel."
# → drift_score: 0.71, risk: high
```

## How it works

- **`HallucinationHunter.analyze()`** tokenises both strings, then computes five signals in parallel: token F1 (harmonic mean of precision/recall), Jaccard (unigram set overlap), bigram overlap (ROUGE-2-ish), novel-token ratio (response words missing from truth), and length ratio (penalty for too short / too long).
- **Weighted blend** — default weights `[0.35, 0.25, 0.20, 0.15, 0.05]` sum to 1.0; override via constructor kwargs to tune sensitivity per domain (factual QA vs summarisation vs code gen).
- **Risk buckets** — `drift < 0.25 → low`, `< 0.5 → medium`, else `high`. Tunable via subclass.
- **Batch + summary** — `batch_analyze(pairs)` runs the pipeline over a list; `summary(results)` gives you mean / min / max drift and a `risk_counts` histogram, ready to plot.
- **CLI** — `--response`/`--truth` for single, `--batch <json>` for many, `--json` for pipe-friendly output. Backend package importable directly: `from backend.app.engines.hallucination_hunter import HallucinationHunter`.

## Screenshots

| CLI demo | Batch summary | Custom weights |
|---|---|---|
| ![](assets/screenshot-1.png) | ![](assets/screenshot-2.png) | ![](assets/screenshot-3.png) |

## Signals

| Signal | Description |
|---|---|
| Token F1 | Harmonic mean of token-level precision and recall |
| Jaccard similarity | Unigram set overlap between response and ground truth |
| Bigram overlap | Consecutive two-word pair coverage (similar to ROUGE-2) |
| Novel token ratio | Fraction of response words absent from the ground truth vocabulary |
| Length ratio | Penalty for responses that are far shorter or longer than expected |

A `drift_score` near **0** means the response closely matches the ground truth. A score near **1** indicates high divergence and likely hallucination.

## Python API

```python
from backend.app.engines.hallucination_hunter import HallucinationHunter

hunter = HallucinationHunter()

# Single analysis
result = hunter.analyze(
    response="The Eiffel Tower was built in 1750 by Napoleon.",
    ground_truth="The Eiffel Tower was completed in 1889 by Gustave Eiffel.",
)

# Batch
pairs = [
    ("Paris is in France.", "Paris is the capital of France."),
    ("Water boils at 50 C.", "Water boils at 100 degrees Celsius."),
]
results = hunter.batch_analyze(pairs)
summary = hunter.summary(results)
# {'n': 2, 'mean_drift_score': ..., 'risk_counts': {'low': 1, 'medium': 1, 'high': 0}}
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

## Running the tests

```bash
python3 -m pytest tests/ -v
```

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

## Stack

Python 3.9+ stdlib only (`re`, `math`, `collections`) · `pytest` for the 29-test suite · optional `numpy>=1.24` if you extend with vector metrics.

## Contributing

PRs welcome. New signals should implement a `_score_<name>(resp_tokens, truth_tokens) -> float in [0, 1]` method on `HallucinationHunter` and register in the weight vector — the blend arithmetic will pick them up automatically.

## License

MIT — see [LICENSE](LICENSE).

---

### More from Danush

- [ponytail-for-python](https://github.com/Danush-Aries/ponytail-for-python) — code intelligence for Python codebases
- [Agentic_Systems](https://github.com/Danush-Aries/Agentic_Systems) — reference implementations of agent patterns
- [autonomous-coding-agent](https://github.com/Danush-Aries/autonomous-coding-agent) — full-auto engineering agent
- [computer-use-agent](https://github.com/Danush-Aries/computer-use-agent) — Claude drives your desktop via VNC
- [browser-automation-agent](https://github.com/Danush-Aries/browser-automation-agent) — Claude drives Playwright
- [blinkchat](https://github.com/Danush-Aries/blinkchat) — realtime chat with vibes
