"""Re-embed all SQLite vector rows with the current configured encoder.

Usage:
  python scripts/reembed_memory.py --db cognitive_memory.db --encoder local_svd
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_cognitive_ai import VectorMemorySQLite


def build_corpus(dataset_dir: Path, chat_log: Path, vm: VectorMemorySQLite):
    texts = []
    for csv_file in sorted(dataset_dir.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text"):
                    texts.append(row["text"])
                if row.get("subject") and row.get("relation") and row.get("object"):
                    texts.append(f"{row['subject']} {row['relation']} {row['object']}")
    if chat_log.exists():
        texts.extend([x.strip() for x in chat_log.read_text(encoding="utf-8").splitlines() if x.strip()])
    texts.extend(vm.all_texts())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cognitive_memory.db")
    parser.add_argument("--encoder", choices=["local_svd", "hash"], default="local_svd")
    parser.add_argument("--datasets", default="data/datasets")
    parser.add_argument("--chat-log", default="data/chat_logs.txt")
    args = parser.parse_args()

    vm = VectorMemorySQLite(db_path=args.db, encoder_backend=args.encoder)
    if args.encoder == "local_svd":
        corpus = build_corpus(Path(args.datasets), Path(args.chat_log), vm)
        metrics = vm.train_local_encoder(corpus)
        print(f"[encoder] trained local_svd metrics={metrics}")

    updated = vm.reembed_all(target_encoder=args.encoder)
    print(f"[reembed] updated_rows={updated} encoder_version={vm.encoder_version}")


if __name__ == "__main__":
    main()
