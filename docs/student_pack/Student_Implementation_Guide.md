# Student Implementation Guide
## Project 03 — Hybrid Enterprise GenAI & Local Model Intelligence Platform

### Objective
Build a production-oriented enterprise GenAI platform that runs local language models, ingests internal knowledge, retrieves grounded evidence, orchestrates controlled tools, and measures quality, safety, latency, and reliability.

### Non-negotiable requirements
- Local model inference must be demonstrated. Cloud-only inference does not satisfy the project.
- Retrieval answers must cite source documents and must abstain when the corpus does not support an answer.
- Retrieved documents and user inputs must be treated as untrusted data.
- Sensitive actions must be permission-checked and routed for human approval where required.
- Evaluation must be repeatable and include failures, not only successful examples.
- The application must run using documented setup instructions and Docker Compose or an equivalent reproducible deployment.

### Module 1 — AI Architecture & Local LLM Engineering (Days 1–4)
Deliver: architecture diagram, local model runtime, model gateway/provider interface, configuration, benchmark report, hardware and model details.
Demonstrate: offline inference after model download, streaming or response handling, timeout/error handling, and a comparison of at least two feasible local model configurations.

### Module 2 — Knowledge Engineering & Advanced RAG (Days 5–10)
Use all 48 Markdown documents. Implement ingestion, parsing, chunking, metadata, document versioning, deduplication, indexing, hybrid retrieval and optional reranking.
Deliver: ingestion pipeline, index schema, retrieval API, citation-aware answers, retrieval evaluation on the 240-question dataset, and analysis of unanswerable questions.

### Module 3 — Agentic AI, Evaluation & Guardrails (Days 11–17)
Implement bounded tool use, workflow state, role/permission checks, approval gates, prompt-injection defenses and audit logs.
Deliver: agent workflow tests using 100 scenarios, adversarial tests using 120 cases, safety evaluation, failure analysis, and evidence of approval enforcement.

### Module 4 — Hybrid AI Orchestration & Observability (Days 18–24)
Implement routing between local models and any optional approved provider, fallbacks, retries, timeouts, rate limits and observability.
Use all 5,000 telemetry records to build dashboards and calculate latency percentiles, error rates, token usage, route/model comparisons, guardrail events and feedback trends.
Deliver: dashboard, metric definitions, alert rules, reliability analysis, and a report distinguishing synthetic-data findings from live-system measurements.

### Module 5 — Production AI Engineering & Final Defence (Days 25–30)
Package the platform, add tests, secure configuration, document deployment, run an end-to-end demonstration and defend design decisions.
Deliver: source repository, Docker setup, architecture and threat model, test evidence, evaluation report, runbook, demo and live technical defence.

### Assessment expectations
A working system matters more than a polished slide deck. Students must explain design trade-offs, show source code, reproduce selected results, debug an unseen issue, and explain limitations honestly.
