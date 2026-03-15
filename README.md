# Adaptive Cognitive AI (No hosted LLM API)

Hybrid symbolic + neural cognitive AI prototype in Python with:

- Dynamic reasoning, Bayesian confidence, hierarchical planning
- SQLite vector memory and retrieval (RAG-style top-k facts)
- Circular convolution for relation encoding
- Dynamic knowledge graph with transitive inference
- Short-term and long-term context memory
- Neural text generator (compact GRU LM in PyTorch)
- BPE tokenizer trained from bundled corpora + chat logs
- Gradio chat UI with visible internal thinking and training controls

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy gradio torch
```

## Pre-train with bundled datasets

```bash
python train_ai.py
```

This ingests all CSV files in `data/datasets/`, updates `cognitive_memory.db`, and builds neural artifacts under `artifacts/neural/`.

## Run

```bash
python adaptive_cognitive_ai.py
```

Open: <http://localhost:7860>

## Training from UI

- Select one or more bundled datasets in **Select bundled datasets**.
- Click **Train Selected Datasets**.
- You can still upload your own CSV and run **Run Bulk Training from Uploaded CSV**.

## CSV format

```csv
text,subject,relation,object
cat is a mammal,,,
,mammal,is_a,vertebrate
```

## Notes

- The system remains interpretable: `AI (thinking)` shows sentiment, posterior confidence, retrieved IDs, and neural-generation metadata.
- Neural generation uses local PyTorch artifacts and does not call remote LLM APIs.
