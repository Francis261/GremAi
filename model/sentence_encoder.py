import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


class HashEncoder:
    """Deterministic fallback encoder with hash-projected token vectors."""

    def __init__(self, dim: int = 128, version: str = "hash-v1"):
        self.dim = dim
        self.version = version

    def encode(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            seed = abs(hash((token, self.version))) % (2**32)
            rng = np.random.default_rng(seed)
            vec += rng.normal(0.0, 1.0, self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class LocalSentenceEncoder:
    """
    Trainable local sentence encoder using co-occurrence + truncated SVD.

    - CPU-friendly
    - Fully local, no remote API
    - Persisted artifacts for deterministic deployment
    """

    def __init__(self, artifact_dir: str = "artifacts/encoder", dim: int = 128, min_freq: int = 1):
        self.artifact_dir = Path(artifact_dir)
        self.dim = dim
        self.min_freq = min_freq
        self.version = "local-svd-v1"
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: List[str] = []
        self.token_matrix: np.ndarray = np.zeros((1, self.dim), dtype=np.float32)
        self.trained = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9']+", text.lower()) if len(t) > 1]

    def fit(self, texts: List[str], window: int = 2) -> Dict[str, float]:
        corpus = [self._tokenize(t) for t in texts if t and t.strip()]
        corpus = [x for x in corpus if x]
        if not corpus:
            self.trained = False
            return {"tokens": 0.0, "dim": float(self.dim)}

        counts = Counter(tok for sent in corpus for tok in sent)
        vocab = [tok for tok, c in counts.items() if c >= self.min_freq]
        if not vocab:
            self.trained = False
            return {"tokens": 0.0, "dim": float(self.dim)}

        self.token_to_idx = {t: i for i, t in enumerate(vocab)}
        self.idx_to_token = vocab

        n = len(vocab)
        cooc = np.zeros((n, n), dtype=np.float32)
        for sent in corpus:
            idxs = [self.token_to_idx[t] for t in sent if t in self.token_to_idx]
            for i, center in enumerate(idxs):
                lo = max(0, i - window)
                hi = min(len(idxs), i + window + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    cooc[center, idxs[j]] += 1.0

        # PPMI transformation
        total = float(cooc.sum()) + 1e-9
        row_sum = cooc.sum(axis=1, keepdims=True) + 1e-9
        col_sum = cooc.sum(axis=0, keepdims=True) + 1e-9
        pmi = np.log((cooc * total + 1e-9) / (row_sum @ col_sum))
        ppmi = np.maximum(pmi, 0.0)

        target_dim = min(self.dim, max(8, min(ppmi.shape) - 1))
        try:
            u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
            emb = (u[:, :target_dim] * np.sqrt(s[:target_dim])).astype(np.float32)
        except np.linalg.LinAlgError:
            emb = np.random.normal(0, 1, (n, target_dim)).astype(np.float32)

        if target_dim < self.dim:
            pad = np.zeros((n, self.dim - target_dim), dtype=np.float32)
            emb = np.hstack([emb, pad])

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        self.token_matrix = emb / norms
        self.trained = True
        return {"tokens": float(n), "dim": float(self.dim)}

    def encode(self, text: str) -> np.ndarray:
        if not self.trained:
            return np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        vecs = []
        for t in tokens:
            idx = self.token_to_idx.get(t)
            if idx is not None:
                vecs.append(self.token_matrix[idx])
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = np.linalg.norm(v)
        return (v / norm).astype(np.float32) if norm > 0 else v.astype(np.float32)

    def save(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.artifact_dir / "token_matrix.npy", self.token_matrix)
        payload = {
            "dim": self.dim,
            "min_freq": self.min_freq,
            "version": self.version,
            "idx_to_token": self.idx_to_token,
            "trained": self.trained,
        }
        (self.artifact_dir / "encoder.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> bool:
        meta_path = self.artifact_dir / "encoder.json"
        matrix_path = self.artifact_dir / "token_matrix.npy"
        if not (meta_path.exists() and matrix_path.exists()):
            return False
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dim = int(payload.get("dim", self.dim))
        self.min_freq = int(payload.get("min_freq", self.min_freq))
        self.version = payload.get("version", self.version)
        self.idx_to_token = list(payload.get("idx_to_token", []))
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
        self.token_matrix = np.load(matrix_path).astype(np.float32)
        self.trained = bool(payload.get("trained", False))
        return self.trained
