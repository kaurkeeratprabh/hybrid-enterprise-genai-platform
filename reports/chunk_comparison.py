
from pathlib import Path
import json

from backend.ingestion.pipeline import ingest_directory

configs = [
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 700, "overlap": 100},
]

corpus = Path("data/corpus")
results = []

if not any(corpus.glob("*.md")):
    print("No Markdown documents found.")
    print("Add the 48 policy documents to data/corpus before comparison.")
else:
    for config in configs:
        docs, chunks = ingest_directory(
            str(corpus),
            config["chunk_size"],
            config["overlap"],
            manifest_path=None
        )

        results.append({
            **config,
            "documents": len(docs),
            "chunks": len(chunks)
        })

    Path("reports").mkdir(exist_ok=True)

    with open("reports/chunk_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))
