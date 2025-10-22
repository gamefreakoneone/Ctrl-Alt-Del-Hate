import json
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from scipy.stats import spearmanr

TEST_FILE = "../data/test.jsonl"
OUTPUT_FILE = "./llama_outputs/llama_baseline_outputs_validated.jsonl"
OUTPUT_EVAL = "./llama_outputs/llama_baseline_eval.txt"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    gold = load_jsonl(TEST_FILE)
    preds = load_jsonl(OUTPUT_FILE)

    gold_data, model_outputs = [], []
    for p in preds:
        if p["prediction"] is not None:
            gold_entry = next((d for d in gold if d["comment_id"] == p["id"]), None)
            if gold_entry:
                gold_data.append(gold_entry)
                model_outputs.append(p["prediction"])

    print(f"Evaluating {len(model_outputs)} valid responses...")
    
    if len(model_outputs) == 0:
        print("No valid predictions found! Check your inference results.")
        print("This usually means the model failed to generate valid JSON responses.")
        return

    lines = []

    # === OVERALL ===
    y_true = [x["overall"]["hate_speech_score"] for x in gold_data]
    y_pred = [
        x["overall"].get("hate_speech_score", x["overall"].get("score"))
        for x in model_outputs
    ]
    mae = mean_absolute_error(y_true, y_pred)
    corr = spearmanr(y_true, y_pred).correlation

    lines.append("=== OVERALL ===")
    lines.append(f"MAE: {mae:.4f}")
    lines.append(f"Spearman: {corr:.4f}\n")

    # === FACETS ===
    facet_names = gold_data[0]["facets"].keys()
    facet_mae, facet_corr = {}, {}
    for f in facet_names:
        y_true = [x["facets"][f] for x in gold_data]
        y_pred = [x["facets"][f] for x in model_outputs]
        facet_mae[f] = mean_absolute_error(y_true, y_pred)
        try:
            corr = spearmanr(y_true, y_pred).correlation
            print(f"\nFacet {f}:")
            print(f"Gold values (first 5): {y_true[:5]}")
            print(f"Pred values (first 5): {y_pred[:5]}")
            print(f"Correlation: {corr}")
            facet_corr[f] = 0.0 if np.isnan(corr) else corr
        except Exception as e:
            print(f"Error calculating correlation for {f}: {e}")
            facet_corr[f] = 0.0

    lines.append("=== FACETS ===")
    lines.append(f"Mean MAE: {np.mean(list(facet_mae.values())):.4f}")
    lines.append(f"Mean Spearman: {np.mean(list(facet_corr.values())):.4f}\n")

    # === TARGETS ===
    target_names = gold_data[0]["targets"].keys()
    y_true = np.array(
        [[x["targets"][t] for t in target_names] for x in gold_data], dtype=int
    )
    y_pred = np.array(
        [[x["targets"][t] for t in target_names] for x in model_outputs], dtype=int
    )

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1_targets = f1_score(y_true, y_pred, average="macro", zero_division=0)

    lines.append("=== TARGETS ===")
    lines.append(f"Micro F1: {micro_f1:.4f}")
    lines.append(f"Macro F1: {macro_f1_targets:.4f}\n")

    lines.append("✅ Evaluation complete.\n")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"Metrics written to {OUTPUT_EVAL}")


if __name__ == "__main__":
    main()
