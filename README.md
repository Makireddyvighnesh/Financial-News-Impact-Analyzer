# Financial News Impact Analyzer 

## Project Overview

A multi-agent system using LLMs to extract financial sentiment and predict market impact from news articles. Built using `pydantic-ai`, OpenAI-compatible LLMs (DeepSeek), and evaluated using both rule-based and LLM-based methods.

---

## Agents

- **News Event Agent:** Extracts `tickers` and `events` from the article.
- **Sentiment Agent:** Analyzes sentiment and assigns confidence.
- **Impact Agent:** Predicts stock direction and % change.
- **Evaluation Agent:** Uses an LLM to assess prediction quality (bonus).

---

## Evaluation Metrics

1. **Direction Accuracy:** % of predictions with correct `up/down/neutral` labels.
2. **Average Model Confidence**
3. **LLM Evaluation Score** (optional)

---

## Folder Structure

```

agents/         # Each agent module
evaluation/     # eval.py and LLM evaluator
data/           # Input articles
outputs/        # Generated predictions

```

---

## What Didn’t Work

- Some LLM responses were not perfectly JSON formatted (needed regex cleaning).
- ByteDance article lacked a stock ticker → used `overall` as fallback.
- LLM sometimes returned structured data as lists of dicts → validation fixes added.

---


