import pandas as pd
import json
from pathlib import Path

source = Path("data/observability/ai_observability_events_5000.csv")
out = Path("reports")
out.mkdir(exist_ok=True)

df = pd.read_csv(source)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

summary = {
    "total_events": int(len(df)),
    "status_distribution": df["status"].value_counts().to_dict(),
    "average_latency_ms": round(float(df["latency_ms"].mean()), 2),
    "p95_latency_ms": round(float(df["latency_ms"].quantile(0.95)), 2),
    "guardrail_triggered": int(df["guardrail_triggered"].astype(str).str.lower().isin(["true", "1"]).sum()),
    "average_retrieved_chunks": round(float(df["retrieved_chunks"].mean()), 2),
    "average_citations": round(float(df["citation_count"].mean()), 2),
    "feedback_distribution": df["user_feedback"].value_counts().to_dict(),
    "route_distribution": df["route"].value_counts().to_dict(),
    "model_distribution": df["model"].value_counts().to_dict()
}

with open(out / "observability_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

df.groupby(["route", "status"]).size().reset_index(name="event_count").to_csv(
    out / "observability_by_route.csv", index=False
)

print("OBSERVABILITY ANALYSIS COMPLETE")
print(json.dumps(summary, indent=4))
print("\nReports saved in reports/")
