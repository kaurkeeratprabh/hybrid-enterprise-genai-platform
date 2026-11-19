import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("data/evaluation/adversarial_prompt_tests_120.csv")
out = Path("reports")
out.mkdir(exist_ok=True)

summary = {
    "total_test_cases": int(len(df)),
    "attack_categories": df["attack_category"].value_counts().to_dict(),
    "severity_distribution": df["severity"].value_counts().to_dict(),
    "should_block": int(df["should_block"].astype(str).str.lower().isin(["true", "1"]).sum()),
    "should_not_block": int(df["should_block"].astype(str).str.lower().isin(["false", "0"]).sum())
}

df.groupby(["attack_category", "severity"]).size().reset_index(
    name="test_count"
).to_csv(out / "adversarial_test_summary.csv", index=False)

with open(out / "adversarial_test_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("ADVERSARIAL TEST DATASET ANALYSIS")
print(json.dumps(summary, indent=4))
print("Report saved in reports/")
