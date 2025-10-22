import json

# ========== CONFIG ==========
INPUT_FILE = "./baseline_data/gemma_baseline_outputs.jsonl"
OUTPUT_FILE = "./baseline_data/gemma_baseline_outputs_validated.jsonl"
# =============================


def clamp_int(value, min_val=0, max_val=4):
    """Round floats, clamp to range [0, 4], and ensure integer."""
    try:
        value = round(float(value))
    except Exception:
        value = 0
    return max(min_val, min(max_val, int(value)))


def validate_schema(entry):
    """Validate one prediction entry."""
    if "prediction" not in entry or entry["prediction"] is None:
        return entry

    pred = entry["prediction"]

    # ========== OVERALL ==========
    overall = pred.get("overall", {})
    score = overall.get("score", 0.0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    overall["score"] = score

    # Keep label if exists (no need to modify)
    pred["overall"] = overall

    # ========== FACETS ==========
    facets_schema = [
        "sentiment",
        "respect",
        "insult",
        "humiliate",
        "status",
        "dehumanize",
        "violence",
        "genocide",
        "attack_defend",
        "hatespeech",
    ]
    facets = pred.get("facets", {})
    validated_facets = {}
    for key in facets_schema:
        validated_facets[key] = clamp_int(facets.get(key, 0))
    pred["facets"] = validated_facets

    # ========== TARGETS ==========
    # All targets default to False
    targets_schema = [
        "target_race_asian",
        "target_race_black",
        "target_race_latinx",
        "target_race_middle_eastern",
        "target_race_native_american",
        "target_race_pacific_islander",
        "target_race_white",
        "target_race_other",
        "target_religion_atheist",
        "target_religion_buddhist",
        "target_religion_christian",
        "target_religion_hindu",
        "target_religion_jewish",
        "target_religion_mormon",
        "target_religion_muslim",
        "target_religion_other",
        "target_origin_immigrant",
        "target_origin_migrant_worker",
        "target_origin_specific_country",
        "target_origin_undocumented",
        "target_origin_other",
        "target_gender_men",
        "target_gender_non_binary",
        "target_gender_transgender_men",
        "target_gender_transgender_unspecified",
        "target_gender_transgender_women",
        "target_gender_women",
        "target_gender_other",
        "target_sexuality_bisexual",
        "target_sexuality_gay",
        "target_sexuality_lesbian",
        "target_sexuality_straight",
        "target_sexuality_other",
        "target_age_children",
        "target_age_teenagers",
        "target_age_young_adults",
        "target_age_middle_aged",
        "target_age_seniors",
        "target_age_other",
        "target_disability_physical",
        "target_disability_cognitive",
        "target_disability_neurological",
        "target_disability_visually_impaired",
        "target_disability_hearing_impaired",
        "target_disability_unspecific",
        "target_disability_other",
    ]
    targets = pred.get("targets", {})
    validated_targets = {}
    for key in targets_schema:
        val = targets.get(key, False)
        validated_targets[key] = bool(val) if isinstance(val, bool) else False
    pred["targets"] = validated_targets

    entry["prediction"] = pred
    return entry


def validate_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                validated = validate_schema(entry)
                outfile.write(json.dumps(validated) + "\n")
            except Exception as e:
                print(f"⚠️ Error on line {line_num}: {e}")
                continue
    print(f"✅ Validation complete. Saved to {output_path}")


if __name__ == "__main__":
    validate_file(INPUT_FILE, OUTPUT_FILE)
