# Data Inventory — Project 03
| Resource | Count | Format | Main use |
|---|---:|---|---|
| Enterprise knowledge documents | 48 | Markdown | Ingestion, chunking, versioning, retrieval |
| Corpus manifest | 48 rows | CSV | Metadata and document inventory |
| RAG QA ground truth | 240 | CSV | Retrieval and grounded answer evaluation |
| Adversarial prompt tests | 120 | CSV | Prompt injection and safety evaluation |
| Agent workflow cases | 100 | CSV | Tool orchestration and approval tests |
| Observability events | 5,000 | CSV | Monitoring, dashboard and reliability analysis |

## Observability schema
`event_id`, `timestamp`, `environment`, `route`, `model`, `status`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `retrieved_chunks`, `citation_count`, `guardrail_triggered`, `user_feedback`, `trace_id`.

## Limitations
All datasets are synthetic. Observability records are simulated events, not actual production telemetry. Statistical patterns are intentionally useful for engineering exercises but should not be represented as real-world performance evidence.
