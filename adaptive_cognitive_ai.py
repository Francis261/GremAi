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

from model.neural_generator import NeuralGenerator


@dataclass
class CognitiveResult:
    """Container for all model outputs for a turn."""

    response: str
    thinking: str
    confidence: float
    retrieved_facts: List[str]
    sentiment: float


class VectorMemorySQLite:
    """SQLite-backed vector memory for facts, responses and relation embeddings."""

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
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            rng = np.random.default_rng(self._token_to_seed(token))
            vec += rng.normal(0.0, 1.0, self.embedding_dim).astype(np.float32)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def circular_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        conv = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real.astype(np.float32)
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

    def fetch_top_k(self, query: str, k: int = 5, kind: Optional[str] = None) -> List[Tuple[int, str, float, dict]]:
        query_vec = self.embed_text(query)
        cur = self.conn.cursor()
        if kind:
            cur.execute("SELECT id, text, embedding, reward, usage_count, metadata FROM vectors WHERE kind=?", (kind,))
        else:
            cur.execute("SELECT id, text, embedding, reward, usage_count, metadata FROM vectors")

        scored = []
        for rid, text, emb_blob, reward, usage_count, metadata in cur.fetchall():
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            sim = float(np.dot(query_vec, emb)) if emb.size == query_vec.size else 0.0
            adaptive_score = sim + 0.05 * math.tanh(reward) + 0.02 * math.log1p(max(usage_count, 1))
            scored.append((rid, text, adaptive_score, json.loads(metadata or "{}")))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

    def all_texts(self, kinds: Optional[List[str]] = None) -> List[str]:
        cur = self.conn.cursor()
        if kinds:
            placeholders = ",".join(["?"] * len(kinds))
            cur.execute(f"SELECT text FROM vectors WHERE kind IN ({placeholders})", tuple(kinds))
        else:
            cur.execute("SELECT text FROM vectors")
        return [r[0] for r in cur.fetchall()]


class KnowledgeGraph:
    """Dynamic entity-relation store with transitive inference."""

    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str, float]]] = {}

    def add_relation(self, subject: str, relation: str, obj: str, confidence: float = 0.8) -> None:
        self.edges.setdefault(subject.strip().lower(), []).append((relation.strip().lower(), obj.strip().lower(), confidence))

    def transitive_inference(self, relation_type: str = "is_a") -> List[Tuple[str, str, float]]:
        derived = []
        for a, rels in self.edges.items():
            for rel1, b, c1 in rels:
                if rel1 != relation_type:
                    continue
                for rel2, c, c2 in self.edges.get(b, []):
                    if rel2 == relation_type and a != c:
                        derived.append((a, c, c1 * c2))
        return derived

    def from_text(self, text: str) -> List[Tuple[str, str, str]]:
        patterns = [
            (r"(.+)\s+is a\s+(.+)", "is_a"),
            (r"(.+)\s+part of\s+(.+)", "part_of"),
            (r"(.+)\s+causes\s+(.+)", "causes"),
            (r"(.+)\s+requires\s+(.+)", "requires"),
        ]
        low = text.lower().strip(" .!?")
        out = []
        for p, rel in patterns:
            m = re.match(p, low)
            if m:
                out.append((m.group(1).strip(), rel, m.group(2).strip()))
        return out


class ContextMemory:
    """Short-term dialogue memory and long-term salience map."""

    def __init__(self, short_limit: int = 8):
        self.short_limit = short_limit
        self.short_term: List[Tuple[str, str]] = []
        self.long_term: Dict[str, float] = {}

    def add_interaction(self, user_text: str, ai_text: str, sentiment: float) -> None:
        self.short_term.append((user_text, ai_text))
        self.short_term = self.short_term[-self.short_limit :]
        for token in re.findall(r"[a-zA-Z0-9']+", f"{user_text} {ai_text}".lower()):
            if len(token) > 2 and token not in {"confidence", "memory", "context"}:
                self.long_term[token] = self.long_term.get(token, 0.0) + (0.4 + sentiment * 0.4)

    def get_recent_context(self) -> str:
        rows = []
        for u, a in self.short_term[-4:]:
            rows.append(f"user:{u}")
            rows.append(f"assistant:{a}")
        return " | ".join(rows)

    def salient_tokens(self, n: int = 8) -> List[str]:
        return [w for w, _ in sorted(self.long_term.items(), key=lambda x: x[1], reverse=True)[:n]]


class ResponseGeneratorDynamic:
    """Reasoning signals + planning + sentiment used for conditioning the neural generator."""

    positive_words = {"great", "good", "helpful", "thanks", "excellent", "love", "happy", "awesome"}
    negative_words = {"bad", "wrong", "hate", "angry", "terrible", "sad", "upset", "frustrated"}

    def detect_sentiment(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in self.positive_words)
        neg = sum(1 for t in tokens if t in self.negative_words)
        return (pos - neg) / max(len(tokens), 1)

    def bayesian_confidence(self, evidence_score: float, prior: float = 0.5) -> float:
        likelihood = min(max(0.5 + 0.5 * evidence_score, 0.01), 0.99)
        return float((likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior)))

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


class NeuralResponseBridge:
    """Connects symbolic signals to neural text generation and rationale metadata."""

    def __init__(self, artifact_dir: str = "artifacts/neural"):
        self.neural = NeuralGenerator(artifact_dir=artifact_dir)

    def _build_training_texts(self, dataset_dir: Path, vector_memory: VectorMemorySQLite, chat_log_path: Path) -> List[str]:
        texts = []
        for csv_file in sorted(dataset_dir.glob("*.csv")):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("text"):
                        texts.append(row["text"])
                    if row.get("subject") and row.get("relation") and row.get("object"):
                        texts.append(f"{row['subject']} {row['relation']} {row['object']}")

        texts.extend(vector_memory.all_texts(["fact", "response"]))
        if chat_log_path.exists():
            for line in chat_log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    texts.append(line.strip())
        return texts

    def ensure_ready(self, dataset_dir: Path, vector_memory: VectorMemorySQLite, chat_log_path: Path) -> Dict[str, float]:
        if self.neural.is_ready():
            self.neural.load()
            return {"status": 1.0}
        texts = self._build_training_texts(dataset_dir, vector_memory, chat_log_path)
        return self.neural.train_from_texts(texts, epochs=3)

    def generate(
        self,
        user_text: str,
        facts: List[Tuple[int, str, float, dict]],
        plan: List[str],
        context: str,
        salients: List[str],
        sentiment: float,
        confidence: float,
    ) -> Tuple[str, Dict[str, float], str]:
        fact_ids = [str(rid) for rid, *_ in facts]
        fact_texts = [t for _, t, _, _ in facts[:3]]

        condition = [
            f"user: {user_text}",
            f"sentiment: {sentiment:.2f}",
            f"confidence_prior: {confidence:.2f}",
            f"facts: {' | '.join(fact_texts) if fact_texts else 'none'}",
            f"plan: {' | '.join(plan) if plan else 'none'}",
            f"salient: {', '.join(salients[:6]) if salients else 'none'}",
            f"context: {context[:260] if context else 'none'}",
            "assistant:",
        ]
        prompt = "\n".join(condition)
        generated, meta = self.neural.generate(prompt, temperature=0.75, top_k=40, top_p=0.9, repetition_penalty=1.12)

        response = generated.strip()
        if not response:
            response = "I can help with that. Based on current memory, please share one more detail so I can refine the answer."

        rationale = f"retrieval_ids={fact_ids}; neural_entropy={meta.get('avg_entropy', 0.0):.3f}; neural_steps={int(meta.get('steps', 0))}"
        return response, meta, rationale


class AdaptiveCognitiveAI:
    """Main orchestrator for memory, graph reasoning, and neural response generation."""

    def __init__(self, db_path: str = "cognitive_memory.db"):
        self.vector_memory = VectorMemorySQLite(db_path=db_path)
        self.knowledge_graph = KnowledgeGraph()
        self.context_memory = ContextMemory()
        self.rule_generator = ResponseGeneratorDynamic()
        self.neural_bridge = NeuralResponseBridge()
        self.dataset_dir = Path("data/datasets")
        self.chat_log_path = Path("data/chat_logs.txt")
        self.last_response: Optional[str] = None
        self._boot_info = self.neural_bridge.ensure_ready(self.dataset_dir, self.vector_memory, self.chat_log_path)

    def _ingest_text(self, text: str) -> None:
        self.vector_memory.upsert_memory(text, kind="fact", reward=0.01)
        for s, rel, o in self.knowledge_graph.from_text(text):
            self.knowledge_graph.add_relation(s, rel, o)
            rel_vec = self.vector_memory.circular_convolution(self.vector_memory.embed_text(s), self.vector_memory.embed_text(o))
            self.vector_memory.upsert_memory(
                f"{s} {rel} {o}",
                kind="relation",
                reward=0.02,
                metadata={"relation": rel, "conv_norm": float(np.linalg.norm(rel_vec))},
            )

    def _append_chat_log(self, user_text: str, response: str) -> None:
        self.chat_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chat_log_path, "a", encoding="utf-8") as f:
            f.write(f"user: {user_text}\nassistant: {response}\n")

    def chat(self, user_text: str) -> CognitiveResult:
        self._ingest_text(user_text)
        facts = self.vector_memory.fetch_top_k(user_text, k=5)
        inferred = self.knowledge_graph.transitive_inference("is_a")
        context = self.context_memory.get_recent_context()
        salients = self.context_memory.salient_tokens()

        sentiment = self.rule_generator.detect_sentiment(user_text)
        evidence = float(np.mean([score for _, _, score, _ in facts])) if facts else 0.0
        confidence = self.rule_generator.bayesian_confidence(evidence)
        plan = self.rule_generator.hierarchical_plan(user_text) if any(
            w in user_text.lower() for w in ["make", "build", "plan", "organize", "prepare"]
        ) else []

        response, meta, rationale = self.neural_bridge.generate(
            user_text=user_text,
            facts=facts,
            plan=plan,
            context=context,
            salients=salients,
            sentiment=sentiment,
            confidence=confidence,
        )

        if inferred and "inference" not in response.lower():
            a, c, conf = inferred[0]
            response = f"{response} Inference hint: {a} -> {c} ({conf:.2f})."

        self.context_memory.add_interaction(user_text, response, sentiment)
        self.vector_memory.upsert_memory(response, kind="response", reward=0.01)
        self.last_response = response
        self._append_chat_log(user_text, response)

        facts_only = [x[1] for x in facts]
        thinking = (
            f"sentiment={sentiment:.2f}; evidence={evidence:.2f}; posterior={confidence:.2f}; "
            f"facts_used={len(facts)}; plan_steps={len(plan)}; {rationale}; "
            f"boot_status={self._boot_info}"
        )

        return CognitiveResult(
            response=response,
            thinking=thinking,
            confidence=confidence,
            retrieved_facts=facts_only,
            sentiment=sentiment,
        )

    def apply_feedback(self, positive: bool) -> str:
        if not self.last_response:
            return "No response available for feedback yet."
        reward = 0.5 if positive else -0.3
        self.vector_memory.upsert_memory(self.last_response, kind="response", reward=reward)
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
            gr.Markdown("# Adaptive Cognitive AI\nSymbolic + neural hybrid with memory, graph inference, and training.")
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

        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    ChatUI().launch()
