import csv
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np


@dataclass
class CognitiveResult:
    """Container for all model outputs for a turn."""

    response: str
    thinking: str
    confidence: float
    retrieved_facts: List[str]
    sentiment: float


class VectorMemorySQLite:
    """
    Lightweight vector memory backed by SQLite.

    Stores concepts, entities, and facts as embeddings and supports
    relation composition via circular convolution.
    """

    def __init__(self, db_path: str = "cognitive_memory.db", embedding_dim: int = 128):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT UNIQUE,
                kind TEXT,
                embedding BLOB,
                usage_count INTEGER DEFAULT 0,
                reward REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self.conn.commit()

    def _token_to_seed(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Interpretable hash-based embedding (no external model).

        Uses deterministic random projections per token to stay CPU-friendly and
        incrementally adjustable through memory updates.
        """
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            rng = np.random.default_rng(self._token_to_seed(token))
            token_vec = rng.normal(0.0, 1.0, self.embedding_dim).astype(np.float32)
            vec += token_vec

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compose relation vectors using circular convolution."""
        fa = np.fft.fft(a)
        fb = np.fft.fft(b)
        conv = np.fft.ifft(fa * fb).real.astype(np.float32)
        norm = np.linalg.norm(conv)
        return conv / norm if norm > 0 else conv

    def upsert_memory(self, text: str, kind: str = "fact", reward: float = 0.0, metadata: Optional[dict] = None) -> None:
        metadata = metadata or {}
        emb = self.embed_text(text).tobytes()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO vectors (text, kind, embedding, usage_count, reward, metadata)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(text) DO UPDATE SET
                usage_count = usage_count + 1,
                reward = reward + excluded.reward,
                metadata = excluded.metadata
            """,
            (text, kind, emb, reward, json.dumps(metadata)),
        )
        self.conn.commit()

    def fetch_top_k(self, query: str, k: int = 5, kind: Optional[str] = None) -> List[Tuple[str, float, dict]]:
        query_vec = self.embed_text(query)
        cur = self.conn.cursor()
        if kind:
            cur.execute("SELECT text, embedding, reward, usage_count, metadata FROM vectors WHERE kind=?", (kind,))
        else:
            cur.execute("SELECT text, embedding, reward, usage_count, metadata FROM vectors")

        scored = []
        for text, emb_blob, reward, usage_count, metadata in cur.fetchall():
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            sim = float(np.dot(query_vec, emb)) if emb.size == query_vec.size else 0.0
            adaptive_score = sim + 0.05 * math.tanh(reward) + 0.02 * math.log1p(max(usage_count, 1))
            scored.append((text, adaptive_score, json.loads(metadata or "{}")))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class KnowledgeGraph:
    """Dynamic entity relation graph with transitive inference."""

    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str, float]]] = {}

    def add_relation(self, subject: str, relation: str, obj: str, confidence: float = 0.8) -> None:
        subject = subject.strip().lower()
        obj = obj.strip().lower()
        relation = relation.strip().lower()
        self.edges.setdefault(subject, []).append((relation, obj, confidence))

    def query(self, subject: str) -> List[Tuple[str, str, float]]:
        return self.edges.get(subject.strip().lower(), [])

    def transitive_inference(self, relation_type: str = "is_a") -> List[Tuple[str, str, float]]:
        derived = []
        for a, rels in self.edges.items():
            for rel1, b, c1 in rels:
                if rel1 != relation_type:
                    continue
                for rel2, c, c2 in self.edges.get(b, []):
                    if rel2 == relation_type and a != c:
                        conf = c1 * c2
                        derived.append((a, c, conf))
        return derived

    def from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract simple relation triples from natural text."""
        patterns = [
            (r"(.+)\s+is a\s+(.+)", "is_a"),
            (r"(.+)\s+part of\s+(.+)", "part_of"),
            (r"(.+)\s+causes\s+(.+)", "causes"),
            (r"(.+)\s+requires\s+(.+)", "requires"),
        ]
        triples = []
        low = text.lower().strip(" .!?")
        for p, rel in patterns:
            m = re.match(p, low)
            if m:
                triples.append((m.group(1).strip(), rel, m.group(2).strip()))
        return triples


class ContextMemory:
    """Tracks short-term and long-term context memory."""

    def __init__(self, short_limit: int = 8):
        self.short_limit = short_limit
        self.short_term: List[Tuple[str, str]] = []
        self.long_term: Dict[str, float] = {}

    def add_interaction(self, user_text: str, ai_text: str, sentiment: float) -> None:
        self.short_term.append((user_text, ai_text))
        self.short_term = self.short_term[-self.short_limit :]

        for token in re.findall(r"[a-zA-Z0-9']+", (user_text + " " + ai_text).lower()):
            self.long_term[token] = self.long_term.get(token, 0.0) + (0.5 + sentiment * 0.5)

    def get_recent_context(self) -> str:
        merged = []
        for u, a in self.short_term[-4:]:
            merged.append(f"user:{u}")
            merged.append(f"ai:{a}")
        return " | ".join(merged)

    def salient_tokens(self, n: int = 6) -> List[str]:
        return [w for w, _ in sorted(self.long_term.items(), key=lambda x: x[1], reverse=True)[:n]]


class ResponseGeneratorDynamic:
    """Builds responses from reasoning signals, memory, and feedback-learned patterns."""

    positive_words = {"great", "good", "helpful", "thanks", "excellent", "love", "happy", "awesome"}
    negative_words = {"bad", "wrong", "hate", "angry", "terrible", "sad", "upset", "frustrated"}

    def __init__(self):
        self.pattern_memory: Dict[str, float] = {}

    def detect_sentiment(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in self.positive_words)
        neg = sum(1 for t in tokens if t in self.negative_words)
        return (pos - neg) / max(len(tokens), 1)

    def bayesian_confidence(self, evidence_score: float, prior: float = 0.5) -> float:
        likelihood = min(max(0.5 + 0.5 * evidence_score, 0.01), 0.99)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        return float(posterior)

    def hierarchical_plan(self, text: str) -> List[str]:
        words = [w for w in re.findall(r"[a-zA-Z0-9']+", text.lower()) if len(w) > 2]
        if not words:
            return []
        goal = " ".join(words[:4])
        return [
            f"Define goal: {goal}",
            f"Collect resources: {', '.join(words[:3])}",
            "Execute subtasks in order",
            "Evaluate outcome and adapt",
        ]

    def update_pattern(self, response: str, reward: float) -> None:
        key = " ".join(re.findall(r"[a-zA-Z0-9']+", response.lower())[:14])
        if not key:
            return
        self.pattern_memory[key] = self.pattern_memory.get(key, 0.0) + reward

    def _style_vector(self, sentiment: float) -> Dict[str, float]:
        warm = max(0.0, sentiment)
        careful = max(0.0, -sentiment)
        return {"warm": warm, "careful": careful, "neutral": 1 - abs(sentiment)}

    def _extract_focus(self, user_text: str, facts: List[Tuple[str, float, dict]], salients: List[str]) -> List[str]:
        user_tokens = [t for t in re.findall(r"[a-zA-Z0-9']+", user_text.lower()) if len(t) > 2]
        token_weights: Dict[str, float] = {}
        for tok in user_tokens:
            token_weights[tok] = token_weights.get(tok, 0.0) + 2.0
        for fact, score, _ in facts:
            for tok in re.findall(r"[a-zA-Z0-9']+", fact.lower()):
                if len(tok) > 2:
                    token_weights[tok] = token_weights.get(tok, 0.0) + max(0.1, score)
        for tok in salients:
            if len(tok) > 2:
                token_weights[tok] = token_weights.get(tok, 0.0) + 0.8
        for pattern, reward in self.pattern_memory.items():
            if reward <= 0:
                continue
            for tok in pattern.split():
                if len(tok) > 2:
                    token_weights[tok] = token_weights.get(tok, 0.0) + 0.2 * reward
        return [t for t, _ in sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:8]]

    def generate(
        self,
        user_text: str,
        facts: List[Tuple[str, float, dict]],
        context: str,
        salients: List[str],
        inferred_relations: List[Tuple[str, str, float]],
    ) -> CognitiveResult:
        sentiment = self.detect_sentiment(user_text)
        evidence = float(np.mean([score for _, score, _ in facts])) if facts else 0.0
        confidence = self.bayesian_confidence(evidence)
        is_planning = any(w in user_text.lower() for w in ["make", "build", "plan", "organize", "prepare"])
        plan = self.hierarchical_plan(user_text) if is_planning else []
        style = self._style_vector(sentiment)
        focus_terms = self._extract_focus(user_text, facts, salients)

        greeting = "Hello." if re.search(r"\b(hi|hello|hey)\b", user_text.lower()) else ""
        tone_sentence = ""
        if style["warm"] > 0.2:
            tone_sentence = "I can feel the positive tone in your message."
        elif style["careful"] > 0.2:
            tone_sentence = "I notice some frustration, so I will keep this clear and careful."

        anchor_fact = facts[0][0] if facts else ""
        anchor_sentence = f"Most relevant memory right now: {anchor_fact}." if anchor_fact else ""

        inference_sentence = ""
        if inferred_relations:
            a, c, conf = inferred_relations[0]
            inference_sentence = f"Graph inference suggests {a} -> {c} (confidence {conf:.2f})."

        focus_sentence = f"Key focus terms: {', '.join(focus_terms)}." if focus_terms else ""

        plan_sentence = ""
        if plan:
            numbered = " ".join([f"{idx + 1}) {step}." for idx, step in enumerate(plan)])
            plan_sentence = f"Proposed plan: {numbered}"

        context_sentence = ""
        if context:
            ctx_tokens = re.findall(r"[a-zA-Z0-9']+", context.lower())[:8]
            if ctx_tokens:
                context_sentence = f"I am using recent context tokens: {', '.join(ctx_tokens)}."

        segments = [greeting, tone_sentence, anchor_sentence, inference_sentence, focus_sentence, plan_sentence, context_sentence]
        response = " ".join([s for s in segments if s]).strip()
        if not response:
            response = "I have limited evidence right now, but I can learn quickly from more examples and feedback."
        response += f" Overall confidence: {confidence:.2f}."

        context_token_count = len(re.findall(r"[a-zA-Z0-9']+", context))
        thinking = (
            f"sentiment={sentiment:.2f}; evidence={evidence:.2f}; posterior={confidence:.2f}; "
            f"facts_used={len(facts)}; context_tokens={context_token_count}; "
            f"plan_steps={len(plan)}; focus_terms={focus_terms[:4]}"
        )

        return CognitiveResult(
            response=response,
            thinking=thinking,
            confidence=confidence,
            retrieved_facts=[f for f, _, _ in facts],
            sentiment=sentiment,
        )


class AdaptiveCognitiveAI:
    """Main orchestrator combining memory, graph reasoning, planning, and feedback learning."""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self.vector_memory = VectorMemorySQLite(db_path=db_path)
        self.knowledge_graph = KnowledgeGraph()
        self.context_memory = ContextMemory()
        self.generator = ResponseGeneratorDynamic()
        self.last_response: Optional[str] = None

    def _ingest_text(self, text: str) -> None:
        self.vector_memory.upsert_memory(text, kind="fact", reward=0.01)
        for s, rel, o in self.knowledge_graph.from_text(text):
            self.knowledge_graph.add_relation(s, rel, o)
            s_vec = self.vector_memory.embed_text(s)
            o_vec = self.vector_memory.embed_text(o)
            rel_vec = self.vector_memory.circular_convolution(s_vec, o_vec)
            self.vector_memory.upsert_memory(
                f"{s} {rel} {o}", kind="relation", reward=0.02, metadata={"relation": rel, "conv_norm": float(np.linalg.norm(rel_vec))}
            )

    def chat(self, user_text: str) -> CognitiveResult:
        self._ingest_text(user_text)
        facts = self.vector_memory.fetch_top_k(user_text, k=5)
        inferred = self.knowledge_graph.transitive_inference("is_a")
        context = self.context_memory.get_recent_context()
        salients = self.context_memory.salient_tokens()
        result = self.generator.generate(user_text, facts, context, salients, inferred)

        self.context_memory.add_interaction(user_text, result.response, result.sentiment)
        self.vector_memory.upsert_memory(result.response, kind="response", reward=0.01)
        self.generator.update_pattern(result.response, reward=0.02)
        self.last_response = result.response
        return result

    def apply_feedback(self, positive: bool) -> str:
        if not self.last_response:
            return "No response available for feedback yet."
        reward = 0.5 if positive else -0.3
        self.vector_memory.upsert_memory(self.last_response, kind="response", reward=reward)
        self.generator.update_pattern(self.last_response, reward=reward)
        return f"Feedback learned with reward {reward:.2f}."

    def bulk_train_csv(self, file_path: str) -> str:
        trained = 0
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or row.get("input") or ""
                relation = row.get("relation")
                subject = row.get("subject")
                obj = row.get("object")
                if text:
                    self._ingest_text(text)
                    trained += 1
                if relation and subject and obj:
                    self.knowledge_graph.add_relation(subject, relation, obj, confidence=0.9)
                    trained += 1
        return f"Bulk training complete. {trained} records ingested."


class ChatUI:
    """Gradio interface for interactive chat and training."""

    def __init__(self):
        self.ai = AdaptiveCognitiveAI()
        self.dataset_dir = Path("data/datasets")

    def _available_datasets(self) -> List[str]:
        return [p.name for p in sorted(self.dataset_dir.glob("*.csv"))]

    def _handle_message(self, message: str, history: List[dict]):
        history = history or []
        result = self.ai.chat(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"AI (thinking): {result.thinking}\nAI: {result.response}"})
        facts_str = "\n".join(f"- {x}" for x in result.retrieved_facts[:5]) if result.retrieved_facts else "- none"
        return history, history, facts_str

    def _feedback(self, positive: bool):
        return self.ai.apply_feedback(positive)

    def _bulk_train(self, file_obj):
        if file_obj is None:
            return "Upload a CSV file first."
        return self.ai.bulk_train_csv(file_obj.name)

    def _train_selected(self, selected_files: List[str]) -> str:
        if not selected_files:
            return "Select at least one dataset."
        trained = []
        for name in selected_files:
            path = self.dataset_dir / name
            if path.exists():
                trained.append(f"{name}: {self.ai.bulk_train_csv(str(path))}")
        return "\n".join(trained) if trained else "No valid dataset paths were selected."

    def launch(self):
        datasets = self._available_datasets()
        with gr.Blocks(title="Adaptive Cognitive AI") as demo:
            gr.Markdown("# Adaptive Cognitive AI\nDynamic interpretable cognition with memory, graph inference, and training.")
            chatbot = gr.Chatbot(label="Conversation", height=420)
            state = gr.State([])
            with gr.Row():
                msg = gr.Textbox(label="Message", placeholder="Ask anything, plan tasks, or teach new facts")
                send = gr.Button("Send")
            retrieved = gr.Markdown("### Retrieved Facts\n- none")

            with gr.Row():
                up = gr.Button("👍 Helpful")
                down = gr.Button("👎 Not helpful")
                feedback_status = gr.Textbox(label="Feedback status")

            gr.Markdown("## Training Interface")
            with gr.Row():
                dataset_select = gr.Dropdown(choices=datasets, multiselect=True, label="Select bundled datasets")
                train_selected_btn = gr.Button("Train Selected Datasets")
            file_in = gr.File(label="Upload CSV (columns: text, subject, relation, object)")
            train_btn = gr.Button("Run Bulk Training from Uploaded CSV")
            train_out = gr.Textbox(label="Training output", lines=8)

            send.click(self._handle_message, inputs=[msg, state], outputs=[chatbot, state, retrieved])
            msg.submit(self._handle_message, inputs=[msg, state], outputs=[chatbot, state, retrieved])
            up.click(lambda: self._feedback(True), outputs=feedback_status)
            down.click(lambda: self._feedback(False), outputs=feedback_status)
            train_btn.click(self._bulk_train, inputs=file_in, outputs=train_out)
            train_selected_btn.click(self._train_selected, inputs=dataset_select, outputs=train_out)

            gr.Markdown(
                """
### Demo ideas
- Greetings: *hello there, I feel great today*
- Q&A: *what do you know about photosynthesis?*
- Planning: *make coffee for two people quickly*
- Probabilistic reasoning: *is it likely to rain if clouds are dark?*
- Teach facts: *cat is a mammal* then *mammal is a vertebrate*
- Feedback loop: vote up/down to reinforce response patterns
                """
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    ChatUI().launch()
