import json
from statistics import mean
from collections import defaultdict

# ===== CONFIG =====
TRAIN_FILE = "train.jsonl"
OUTPUT_FILE = "train_aggregated.jsonl"
# ==================


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def aggregate_annotations(records):
    """Aggregate multiple annotations for the same comment into one record."""
    if len(records) == 1:
        return records[0]

    aggregated = {
        "comment_id": records[0]["comment_id"],
        "text": records[0]["text"],
        "overall": {},
        "facets": {},
        "targets": {},
    }

    # ---- Aggregate overall ----
    hate_scores = [r["overall"]["hate_speech_score"] for r in records]
    avg_hate_score = mean(hate_scores)

    if avg_hate_score > 0.5:
        label = "hateful"
    elif avg_hate_score < -1.0:
        label = "supportive"
    else:
        label = "neutral"

    aggregated["overall"] = {"label": label, "hate_speech_score": avg_hate_score}

    # ---- Aggregate facets (mean then round) ----
    facet_keys = records[0]["facets"].keys()
    for key in facet_keys:
        values = [r["facets"][key] for r in records]
        aggregated["facets"][key] = round(mean(values))

    # ---- Aggregate targets (OR logic) ----
    target_keys = records[0]["targets"].keys()
    for key in target_keys:
        aggregated["targets"][key] = any(r["targets"][key] for r in records)

    return aggregated


def main():
    print(f"📥 Loading {TRAIN_FILE} ...")
    data = load_jsonl(TRAIN_FILE)
    print(f"Loaded {len(data)} records")

    # Group by comment_id
    grouped = defaultdict(list)
    for record in data:
        grouped[record["comment_id"]].append(record)
    print(f"Grouped into {len(grouped)} unique comment_ids")

    # Aggregate each group
    aggregated_data = [aggregate_annotations(records) for records in grouped.values()]

    # Save results
    save_jsonl(aggregated_data, OUTPUT_FILE)
    print(f"✅ Aggregated data saved to {OUTPUT_FILE}")
    print(f"Total aggregated records: {len(aggregated_data)}")


if __name__ == "__main__":
    main()
