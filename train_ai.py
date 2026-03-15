"""Offline trainer for Adaptive Cognitive AI including neural generator artifacts."""

from pathlib import Path

from adaptive_cognitive_ai import AdaptiveCognitiveAI


DATASET_DIR = Path("data/datasets")
DB_PATH = "cognitive_memory.db"


def main() -> None:
    ai = AdaptiveCognitiveAI(db_path=DB_PATH)

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit("No datasets found in data/datasets")

    total = 0
    for csv_file in csv_files:
        msg = ai.bulk_train_csv(str(csv_file))
        print(f"[train] {csv_file.name}: {msg}")
        total += 1

    synthetic_turns = [
        "hello, I feel great today",
        "what do you know about photosynthesis",
        "cat is a mammal",
        "mammal is an animal",
        "animal is an organism",
        "make coffee for two people quickly",
        "I am upset because the answer was bad",
        "is rain likely with dark clouds and high humidity",
        "organize a meeting for tomorrow",
        "thanks, this was helpful",
    ]

    for i, prompt in enumerate(synthetic_turns, start=1):
        result = ai.chat(prompt)
        reward_positive = result.confidence >= 0.6 and result.sentiment >= -0.2
        ai.apply_feedback(reward_positive)
        print(f"[chat-{i}] conf={result.confidence:.2f} sentiment={result.sentiment:.2f} reward={'+' if reward_positive else '-'}")

    # train/update local sentence encoder and refresh embeddings
    enc_metrics = ai.vector_memory.train_local_encoder(ai._encoder_corpus())
    reembed_count = ai.vector_memory.reembed_all(target_encoder="local_svd")
    print(f"[encoder] metrics={enc_metrics} reembedded={reembed_count}")

    # retrain neural model with newest chat logs + memory corpus
    metrics = ai.neural_bridge.ensure_ready(ai.dataset_dir, ai.vector_memory, ai.chat_log_path)
    print(f"[neural] metrics={metrics}")
    print(f"Training complete using {total} datasets. DB written to {DB_PATH}")


if __name__ == "__main__":
    main()
