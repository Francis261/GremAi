# Adaptive Cognitive AI (No LLM)

A fully dynamic, interpretable, CPU-friendly cognitive AI prototype in Python with:

- Dynamic reasoning + response generation (no static response templates)
- Bayesian confidence updates
- Hierarchical planning for multi-step tasks
- SQLite vector memory and retrieval (RAG-style top-k facts)
- Circular convolution for relation encoding
- Dynamic knowledge graph + transitive inference
- Short-term and long-term context memory
- Interactive Gradio chat UI with visible internal thinking
- Feedback-based self-learning and bulk CSV training

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy gradio
```

## Pre-train with bundled datasets

```bash
python train_ai.py
```

This consumes all CSV files in `data/datasets/` (20 bundled datasets) and writes a warmed-up `cognitive_memory.db` so hosted deployments can chat immediately.

## Run

```bash
python adaptive_cognitive_ai.py
```

Open: <http://localhost:7860>

## Training from UI

- Select one or more bundled datasets in **Select bundled datasets**.
- Click **Train Selected Datasets**.
- You can still upload your own CSV and run **Run Bulk Training from Uploaded CSV**.

## CSV training format

Use headers like:

```csv
text,subject,relation,object
cat is a mammal,,,
,mammal,is_a,vertebrate
```

## Notes

- The system is intentionally interpretable and does not use pretrained LLMs.
- Embeddings are deterministic hash-projection vectors stored in SQLite.
- `cognitive_memory.db` is committed after offline training for instant startup memory.
