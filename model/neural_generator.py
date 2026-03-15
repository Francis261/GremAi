import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPETokenizer:
    """Small BPE tokenizer with on-disk save/load support."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, vocab_size: int = 600):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

    @staticmethod
    def _word_to_symbols(word: str) -> List[str]:
        return list(word) + ["</w>"]

    def train(self, texts: List[str]) -> None:
        words = []
        for text in texts:
            for w in text.lower().split():
                if w:
                    words.append(tuple(self._word_to_symbols(w)))

        vocab = Counter(words)
        while len(self.merges) < max(self.vocab_size - 256, 0):
            pair_counts = Counter()
            for token_seq, freq in vocab.items():
                for i in range(len(token_seq) - 1):
                    pair_counts[(token_seq[i], token_seq[i + 1])] += freq
            if not pair_counts:
                break
            (a, b), best = pair_counts.most_common(1)[0]
            if best < 2:
                break
            self.merges.append((a, b))

            new_vocab = Counter()
            for token_seq, freq in vocab.items():
                merged = []
                i = 0
                while i < len(token_seq):
                    if i < len(token_seq) - 1 and token_seq[i] == a and token_seq[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(token_seq[i])
                        i += 1
                new_vocab[tuple(merged)] += freq
            vocab = new_vocab

        symbols = {self.PAD, self.BOS, self.EOS, self.UNK}
        for token_seq in vocab:
            symbols.update(token_seq)

        self.id_to_token = sorted(symbols)
        if len(self.id_to_token) > self.vocab_size:
            self.id_to_token = self.id_to_token[: self.vocab_size]
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    def encode(self, text: str) -> List[int]:
        toks = [self.token_to_id.get(self.BOS, 0)]
        for word in text.lower().split():
            symbols = self._word_to_symbols(word)
            for a, b in self.merges:
                merged = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged
            for sym in symbols:
                toks.append(self.token_to_id.get(sym, self.token_to_id.get(self.UNK, 0)))
        toks.append(self.token_to_id.get(self.EOS, 0))
        return toks

    def decode(self, ids: List[int]) -> str:
        parts = []
        for idx in ids:
            tok = self.id_to_token[idx] if 0 <= idx < len(self.id_to_token) else self.UNK
            if tok in {self.PAD, self.BOS, self.EOS}:
                continue
            parts.append(tok)
        text = "".join(parts).replace("</w>", " ").strip()
        return " ".join(text.split())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocab_size": self.vocab_size, "merges": self.merges, "id_to_token": self.id_to_token}))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(path.read_text())
        tok = cls(vocab_size=payload["vocab_size"])
        tok.merges = [tuple(x) for x in payload["merges"]]
        tok.id_to_token = payload["id_to_token"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        e = self.embed(x)
        out, h = self.gru(e, h)
        logits = self.proj(out)
        return logits, h


class NeuralGenerator:
    """Compact neural language model + tokenizer for next-token training and generation."""

    def __init__(self, artifact_dir: str = "artifacts/neural", device: str = "cpu"):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device)
        self.tokenizer_path = self.artifact_dir / "tokenizer.json"
        self.model_path = self.artifact_dir / "gru_lm.pt"
        self.meta_path = self.artifact_dir / "meta.json"
        self.tokenizer: Optional[BPETokenizer] = None
        self.model: Optional[GRULM] = None

    def is_ready(self) -> bool:
        return self.tokenizer_path.exists() and self.model_path.exists() and self.meta_path.exists()

    def load(self) -> None:
        self.tokenizer = BPETokenizer.load(self.tokenizer_path)
        meta = json.loads(self.meta_path.read_text())
        self.model = GRULM(len(self.tokenizer.id_to_token), meta["embed_dim"], meta["hidden_dim"]).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def train_from_texts(
        self,
        texts: List[str],
        vocab_size: int = 700,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        epochs: int = 3,
        lr: float = 2e-3,
    ) -> Dict[str, float]:
        clean = [t.strip() for t in texts if t and t.strip()]
        if len(clean) < 8:
            return {"loss": 0.0, "perplexity": float("inf"), "samples": len(clean)}

        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(clean)
        encoded = [tokenizer.encode(t) for t in clean]
        pairs = []
        for seq in encoded:
            if len(seq) >= 3:
                pairs.append((seq[:-1], seq[1:]))

        model = GRULM(len(tokenizer.id_to_token), embed_dim=embed_dim, hidden_dim=hidden_dim).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        total_tokens = 0
        for _ in range(epochs):
            model.train()
            for inp, tgt in pairs:
                x = torch.tensor(inp, dtype=torch.long, device=self.device).unsqueeze(0)
                y = torch.tensor(tgt, dtype=torch.long, device=self.device).unsqueeze(0)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item()) * y.numel()
                total_tokens += int(y.numel())

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(self.tokenizer_path)
        torch.save(model.state_dict(), self.model_path)
        self.meta_path.write_text(json.dumps({"embed_dim": embed_dim, "hidden_dim": hidden_dim, "epochs": epochs}))

        self.tokenizer = tokenizer
        self.model = model.eval()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        return {"loss": avg_loss, "perplexity": ppl, "samples": len(clean)}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
    ) -> Tuple[str, Dict[str, float]]:
        if self.model is None or self.tokenizer is None:
            if not self.is_ready():
                return "", {"avg_entropy": 0.0, "steps": 0}
            self.load()

        assert self.model is not None and self.tokenizer is not None
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        generated = list(ids)
        h = None
        entropies = []

        for _ in range(max_new_tokens):
            x = torch.tensor([generated[-1]], dtype=torch.long, device=self.device).view(1, 1)
            logits, h = self.model(x, h)
            logits = logits[0, -1, :] / max(temperature, 1e-6)

            # repetition penalty
            for gid in set(generated[-40:]):
                logits[gid] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            entropy = float(-(probs * torch.log(probs + 1e-9)).sum().item())
            entropies.append(entropy)

            if top_k > 0:
                vals, idxs = torch.topk(probs, min(top_k, probs.numel()))
                mask = torch.zeros_like(probs)
                mask[idxs] = vals
                probs = mask / mask.sum()

            if 0 < top_p < 1.0:
                s_probs, s_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(s_probs, dim=0)
                keep = cumsum <= top_p
                keep[0] = True
                filtered = torch.zeros_like(probs)
                filtered[s_idx[keep]] = probs[s_idx[keep]]
                probs = filtered / filtered.sum()

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            if next_id == self.tokenizer.token_to_id.get(BPETokenizer.EOS, -1):
                break

        text = self.tokenizer.decode(generated[len(ids) :]).strip()
        return text, {"avg_entropy": float(sum(entropies) / max(len(entropies), 1)), "steps": float(len(entropies))}
