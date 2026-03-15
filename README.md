# Adaptive Cognitive AI (No hosted LLM API)

Hybrid symbolic + neural cognitive AI prototype in Python with:

- Dynamic reasoning, Bayesian confidence, hierarchical planning
- SQLite vector memory and retrieval (RAG-style top-k facts)
- Pluggable encoder interface (`encode(text)`) with fallback hash encoder
- Local trainable sentence encoder (co-occurrence + SVD, CPU-friendly)
- MMR/diversity-aware retrieval + optional recency weighting
- Circular convolution for relation encoding
- Persistent SQLite knowledge graph (`entities`, `relations`) with confidence updates
- Multi-hop graph reasoning (direct/transitive/causal/requirement chains) + path explanation
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

This ingests all CSV files in `data/datasets/`, updates `cognitive_memory.db`, trains local encoder artifacts under `artifacts/encoder/`, and builds neural artifacts under `artifacts/neural/`.

## Run

```bash
python adaptive_cognitive_ai.py
```

Open: <http://localhost:7860>

## Re-embed existing memory DB after encoder changes

```bash
python scripts/reembed_memory.py --db cognitive_memory.db --encoder local_svd
```

This recomputes all vector embeddings and updates `metadata.encoder_version` for migration safety.

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


## Knowledge graph reasoning

The graph layer now persists in SQLite tables and supports:

- direct query
- transitive query (`is_a`)
- causal chains (`causes`)
- requirement chains (`requires`)
- path explanation via `explain_path(...)` integrated into `AI (thinking)`


## Render deployment fix

If deploying on Render:

- Install dependencies from `requirements.txt` (includes CPU PyTorch).
- Use start command:

```bash
python adaptive_cognitive_ai.py
```

Do **not** use `python adaptive_cognitive_ai.py && python train_ai.py` as a start command, because the web process must stay alive and bind to `$PORT`.

A ready `render.yaml` is included with build/start commands.

If your Render dashboard has custom commands, ensure they are exactly:

- Build command: `pip install -r requirements.txt && python train_ai.py`
- Start command: `python adaptive_cognitive_ai.py`

Also pin Python to 3.12 for torch wheel compatibility (this repo includes `runtime.txt`).

If you see `No matching distribution found for requirements.txt`, your build command is wrong; use `pip install -r requirements.txt` (note `-r`).

The app now degrades gracefully when `torch` is unavailable (symbolic response fallback), but for best quality keep CPU torch installed from `requirements.txt`.
