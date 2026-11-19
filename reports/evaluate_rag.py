
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

from backend.retrieval.index import KnowledgeIndex
from backend.rag.service import answer_with_local_model


def normalize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def token_f1(prediction, reference):
    pred = normalize(prediction)
    ref = normalize(reference)

    if not pred or not ref:
        return 0.0

    common = sum((pd.Series(pred).value_counts()
                  .combine(pd.Series(ref).value_counts(), min, fill_value=0)))

    precision = common / len(pred)
    recall = common / len(ref)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=240)
    args = parser.parse_args()

    data = pd.read_csv(
        "data/evaluation/rag_qa_ground_truth_240.csv"
    ).head(args.limit)

    index = KnowledgeIndex()

    try:
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    except Exception:
        reranker = None

    rows = []

    for number, item in data.iterrows():
        question = str(item["question"])
        expected = str(item["expected_source_document_id"])
        answerable = str(item["answerable"]).lower() == "true"

        print(f"[{number + 1}/{len(data)}] {question}")

        start = time.time()

        try:
            results = index.hybrid_search(
                question,
                top_k=5,
                allowed_roles=["employee"],
                department=item["department"],
                reranker=reranker
            )

            response = answer_with_local_model(question, results)

            answer = str(response.get("answer", ""))
            citations = response.get("citations", [])
            abstained = bool(response.get("abstained", False))

            if isinstance(citations, str):
                citations = [citations]

            retrieved_ids = [
                str(r.get("metadata", {}).get("document_id", ""))
                for r in results
            ]

            citation_hit = expected in [str(c) for c in citations]
            retrieval_hit = expected in retrieved_ids

            rows.append({
                "question_id": item["question_id"],
                "question": question,
                "reference_answer": item["reference_answer"],
                "generated_answer": answer,
                "answerable": answerable,
                "abstained": abstained,
                "expected_source": expected,
                "retrieved_sources": ", ".join(retrieved_ids),
                "citations": ", ".join(map(str, citations)),
                "retrieval_hit": retrieval_hit,
                "citation_hit": citation_hit,
                "token_f1": (
                    token_f1(answer, item["reference_answer"])
                    if answerable and not abstained else 0.0
                ),
                "latency_seconds": round(time.time() - start, 2),
                "error": ""
            })

        except Exception as e:
            rows.append({
                "question_id": item["question_id"],
                "question": question,
                "reference_answer": item["reference_answer"],
                "generated_answer": "",
                "answerable": answerable,
                "abstained": False,
                "expected_source": expected,
                "retrieved_sources": "",
                "citations": "",
                "retrieval_hit": False,
                "citation_hit": False,
                "token_f1": 0.0,
                "latency_seconds": round(time.time() - start, 2),
                "error": str(e)
            })
            print("ERROR:", e)

    output = pd.DataFrame(rows)

    Path("reports").mkdir(exist_ok=True)
    output.to_csv("reports/rag_evaluation_results.csv", index=False)

    answerable_rows = output[output["answerable"] == True]
    unanswerable_rows = output[output["answerable"] == False]

    summary = {
        "questions_evaluated": len(output),
        "successful_runs": int((output["error"] == "").sum()),
        "errors": int((output["error"] != "").sum()),
        "retrieval_hit_rate": round(
            output["retrieval_hit"].mean(), 4
        ),
        "citation_hit_rate": round(
            answerable_rows["citation_hit"].mean()
            if len(answerable_rows) else 0, 4
        ),
        "average_token_f1": round(
            answerable_rows["token_f1"].mean()
            if len(answerable_rows) else 0, 4
        ),
        "unanswerable_abstention_rate": round(
            unanswerable_rows["abstained"].mean()
            if len(unanswerable_rows) else 0, 4
        ),
        "average_latency_seconds": round(
            output["latency_seconds"].mean(), 2
        )
    }

    with open("reports/rag_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=4))
    print("\nSaved results in reports/")


if __name__ == "__main__":
    main()
