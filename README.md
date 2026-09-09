# Spotify AI Support Agent (@SpotifyCares)
> **Hiver SDE Intern Take-Home Assignment**
> An automated, retrieval-grounded AI support agent designed to classify intents, draft historical SOP replies, and deterministically route escalations.

---

## Quickstart & Reproducibility (< 15 Seconds)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Reproducibility Benchmark
```bash
python run_pipeline.py
```

---

## Headline Results (200 Hand-Labelled Golden Samples)

* **Intent Classification Accuracy:** `75.50%` (vs. 13.00% Trivial Baseline)
* **Intent Macro-F1 Score:** `0.7504`
* **Escalation Precision:** `85.19%`
* **Escalation Recall:** `90.79%`
* **Escalation F1-Score:** `0.8790`
* **LLM-as-a-Judge Pass Rate:** `94.50%`
* **Human-Judge Alignment:** `93.33%`

---

## Repository Structure

```text
hiver-ai-support-agent/
├── README.md                           # Quickstart and headline results
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Ignored caches and raw datasets
├── run_pipeline.py                     # < 15 second benchmark verification script
├── src/
│   ├── __init__.py
│   └── agent.py                        # Intent classifier, RAG retriever, and escalation engine
├── evaluation/
│   ├── golden_set_candidates.csv       # 200 hand-labelled ground truth examples
│   └── golden_set_evaluation_results.csv # Granular model predictions & evaluation scores
└── report/
    └── REPORT.md                       # Full 6-page technical report and decision log
```

For the complete architectural breakdown, failure analysis, and decision log, see `report/REPORT.md`.
