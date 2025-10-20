import json
from sklearn.model_selection import train_test_split

INPUT_FILE = "gold_benchmark_dataset.jsonl"
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


def main():
    data = load_data(INPUT_FILE)
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    save_jsonl(train_df, TRAIN_FILE)
    save_jsonl(val_df, VAL_FILE)
    save_jsonl(test_df, TEST_FILE)

    print(f"✅ Data split complete:")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()
