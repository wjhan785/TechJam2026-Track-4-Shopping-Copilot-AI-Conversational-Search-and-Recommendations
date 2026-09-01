# TechJam Shopping Copilot — Submission

Conversational e-commerce search agent for the TechJam Conversational E-Commerce
Search Challenge. The agent holds a dialogue with a simulated customer, asks
clarifying questions, and recommends catalog products until it places the
customer's hidden target product within at most 10 turns.

This bundle is self-contained and runs **fully offline**: no API keys, no
external services, and no network access are required for official scoring.

**Model is not included in Github due to large file size and must hence be downloaded. Read the setup below.**

## Contents

```text
agent.py                          Agent entry re-export
requirements.txt                  dependency manifest
DATA_ATTRIBUTION.md               data provenance and terms
REPORT.docx                       method, model choice, results, limitations
results.json                      latest results
```

## Requirements

- Python 3.10 or later (developed and verified on Python 3.14.7).
- Dependencies (installed by `pip` from `requirements.txt`):
  - `sentence-transformers` (used only for the reranker)
  - `numpy`
  - `optree` (pin avoids a known stale-wheel incompatibility on Python 3.14)
- The agent core (indexing, retrieval, dialogue state) uses only the Python
  standard library (`sqlite3` FTS5, `json`, `re`) and works even if the
  reranker dependencies are missing.
- Install `torch` to use GPU for LLM Reranker

## Setup

```bash
# 1. Create and activate an environment
python -m venv .venv && source .venv/bin/activate   # or conda create -n <env> python=3.14

# 2. Install dependencies
pip install -r requirements.txt
```

**Place the LLM reranker and the catalogue at where the agent can find them.**
1. Catalogue MUST be downloaded to data/catalog.jsonl.
2. LLM Reranker MUST be at assets/model/Qwen3-Reranker-0.6B.
Download the asset folder from this link: https://drive.google.com/file/d/1kJFmS0nLtYqNbjBJdxPPKW22FuvaNYaJ/view?usp=sharing.

Alternatively, you may download from Huggingface: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B and place the model at assets/model/.

If the model is missing the agent degrades silently to lexical-only ranking.

## Run the agent in the official harness

Run your evaluator (based on your path. This is just an example):

```bash
python -m evaluator.local_evaluator
```

Ensure that evaluator imports agent.py from the right path.

This runs the official 200-session public evaluation, imports
agent.py, and writes per-session results and
aggregate metrics to `results.json` in the current directory.

Expected aggregate results (verified on the frozen public set):

| Metric         | Value    |
| -------------- | -------- |
| Hit Rate@10    | 0.99     |
| MRR            | 0.663343 |
| MTTC           | 2.005    |
| Efficiency     | 0.8995   |
| TechnicalScore | 0.873903 |

Runtime is dominated by the lazy local reranker (loaded on first `respond`
call); measured 32 min 23 s for the full 200-session evaluation on a single
CPU (≈3.8 s per reranked turn). It would be faster if using GPU. However, 
if using CPU, expect runtime of at least ~2 hours for 800 sessions.

## Environment variables

| Variable             | Required | Meaning                                                                                        |
| -------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `RERANKER_PATH`      | no       | Directory of the Qwen3-Reranker-0.6B checkpoint; checked first                                 |
| `SHOPCOPILOT_DEVICE` | no       | `auto` (default) / `cpu` / `cuda` — reranker device; `auto` uses CUDA when available, else CPU |
| `SHOPCOPILOT_LOG`    | no       | Set to `0` to silence the agent's stderr progress lines (default `1`)                          |

No other environment variables, secrets, or API credentials are used.

## Live progress output

During evaluation the agent prints one compact progress line per turn to
**stderr**, so judges can watch sessions live, for example:

```text
[ShopCopilot] session public_2aca4 | t2 | cat=Accessories Belts | slots=[100% Leather, Buckle closure, leather] | pool=16 | asked=other | rec=10 | override
```

Each line shows the session, turn, parsed category, disclosed constraint
slots, live candidate pool size, the attribute asked, and recommendation
count, plus `override` / `drained` / `FINAL` flags when applicable. This
output is purely informational. Set `SHOPCOPILOT_LOG=0` to disable.

## Offline statement

- The submission makes **no network calls** at any point (development or
  scoring). All models are local; no live credentials exist.
- If the reranker checkpoint is unavailable, the agent still runs end-to-end
  using lexical retrieval and exact-slot ranking only.

## Data attribution

Derived from Amazon Reviews 2023 (McAuley Lab, UCSD); see
`DATA_ATTRIBUTION.md` for terms. Reranker weights: Qwen3-Reranker-0.6B,
Apache-2.0, Alibaba Cloud.
