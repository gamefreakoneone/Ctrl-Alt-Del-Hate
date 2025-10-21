import json
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "C:\\Users\\amogh\\Desktop\\Ctrl-Alt-Del-Hate\\gold_benchmark_dataset.jsonl"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
TEST_FILE = "test.jsonl"

def load_data(path):
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
    
    # Start with the first record as template
    aggregated = {
        "comment_id": records[0]["comment_id"],
        "text": records[0]["text"],
        "overall": {},
        "facets": {},
        "targets": {}
    }
    
    # Aggregate hate_speech_score (mean)
    hate_scores = [r["overall"]["hate_speech_score"] for r in records]
    avg_hate_score = sum(hate_scores) / len(hate_scores)
    
    # Classify based on averaged score
    if avg_hate_score > 0.5:
        label = "hateful"
    elif avg_hate_score < -1.0:
        label = "supportive"
    else:
        label = "neutral"
    
    aggregated["overall"] = {
        "label": label,
        "hate_speech_score": avg_hate_score
    }
    
    # Aggregate facets (mean then round)
    facet_keys = records[0]["facets"].keys()
    for key in facet_keys:
        values = [r["facets"][key] for r in records]
        aggregated["facets"][key] = round(sum(values) / len(values))
    
    # Aggregate targets (OR logic)
    target_keys = records[0]["targets"].keys()
    for key in target_keys:
        aggregated["targets"][key] = any(r["targets"][key] for r in records)
    
    return aggregated

def main():
    # Load all data
    data = load_data(INPUT_FILE)
    print(f"Loaded {len(data)} total records")
    
    # Group by comment_id
    comment_groups = {}
    for record in data:
        comment_id = record["comment_id"]
        if comment_id not in comment_groups:
            comment_groups[comment_id] = []
        comment_groups[comment_id].append(record)
    
    unique_comment_ids = list(comment_groups.keys())
    print(f"Found {len(unique_comment_ids)} unique comments")
    
    # Split comment_ids: 80% train, 10% val, 10% test
    train_ids, temp_ids = train_test_split(unique_comment_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    # Create training set (keep all annotations)
    train_data = []
    for comment_id in train_ids:
        train_data.extend(comment_groups[comment_id])
    
    # Create val/test sets (aggregate annotations)
    val_data = [aggregate_annotations(comment_groups[comment_id]) for comment_id in val_ids]
    test_data = [aggregate_annotations(comment_groups[comment_id]) for comment_id in test_ids]
    
    # Save datasets
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data, VAL_FILE)
    save_jsonl(test_data, TEST_FILE)
    
    print(f"\n✅ Data split complete:")
    print(f"Train: {len(train_data)} records ({len(train_ids)} unique comments)")
    print(f"Val: {len(val_data)} records ({len(val_ids)} unique comments)")
    print(f"Test: {len(test_data)} records ({len(test_ids)} unique comments)")
    
    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/Val overlap detected!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/Test overlap detected!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/Test overlap detected!"
    print("\n✅ Verified: No comment_id overlap between splits")

if __name__ == "__main__":
    main()