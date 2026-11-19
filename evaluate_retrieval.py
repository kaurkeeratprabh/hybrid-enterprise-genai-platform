
"""Evaluate enterprise retrieval using the available QA dataset."""

import argparse
import pandas as pd
from pathlib import Path
from backend.retrieval.index import KnowledgeIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/evaluation/rag_qa_ground_truth.csv"
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {
        "query_id", "question",
        "expected_department", "answerable"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    index = KnowledgeIndex()

    if index.collection.count() == 0:
        raise RuntimeError(
            "No documents indexed. Add the knowledge corpus first."
        )

    rows = []

    for _, row in df.iterrows():
        results = index.hybrid_search(
            row["question"],
            args.k,
            allowed_roles=["employee", "manager", "admin"],
            department=row["expected_department"]
        )

        rows.append({
            "query_id": row["query_id"],
            "expected_department": row["expected_department"],
            "answerable": bool(row["answerable"]),
            "retrieved_count": len(results),
            "retrieval_success": len(results) > 0,
            "retrieved_ids": ", ".join(
                str(item.get("id", "")) for item in results
            )
        })

    output = pd.DataFrame(rows)

    Path("reports").mkdir(exist_ok=True)
    output.to_csv(
        "reports/retrieval_metrics.csv",
        index=False
    )

    print("\nRETRIEVAL EVALUATION")
    print("--------------------")
    print("Total questions:", len(output))
    print("Questions with results:", output["retrieval_success"].sum())
    print(
        "Average retrieved results:",
        round(output["retrieved_count"].mean(), 2)
    )
    print(
        "Answerable questions:",
        output["answerable"].sum()
    )
    print(
        "Unanswerable questions:",
        (~output["answerable"]).sum()
    )

    print("\nNOTE: Recall@K and MRR require gold relevant document IDs.")
    print("Current results measure retrieval coverage, not answer quality.")


if __name__ == "__main__":
    main()
