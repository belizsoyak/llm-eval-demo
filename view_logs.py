#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


def load_logs_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"logs.jsonl not found: {path}")

    rows = []
    # JSONL may contain blank lines; ignore them.
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if "evaluation" not in df.columns:
        df["evaluation"] = None
    return df


def main() -> None:
    log_path = Path(__file__).with_name("logs.jsonl")
    df = load_logs_jsonl(log_path)

    evaluation = df.get("evaluation", pd.Series([{}] * len(df)))
    df["correctness"] = evaluation.apply(lambda e: e.get("correctness") if isinstance(e, dict) else None)
    df["hallucination_risk"] = evaluation.apply(
        lambda e: e.get("hallucination_risk") if isinstance(e, dict) else None
    )
    df["confidence"] = evaluation.apply(lambda e: e.get("confidence") if isinstance(e, dict) else None)

    table = df[["question", "correctness", "hallucination_risk", "confidence"]].copy()
    print(table.to_string(index=False))

    # Basic stats: counts of hallucination risk levels.
    counts = (
        table["hallucination_risk"]
        .value_counts()
        .reindex(["high", "medium", "low"], fill_value=0)
    )
    print()
    print("Hallucination risk counts:")
    for level in ["high", "medium", "low"]:
        print(f"  {level}: {int(counts[level])}")


if __name__ == "__main__":
    main()

