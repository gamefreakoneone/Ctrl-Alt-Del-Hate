import json, os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# ==============================
# CONFIG
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEST_FILE = "../data/test.jsonl"
OUTPUT_FILE = "./llama_outputs/llama_baseline_outputs.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
SAMPLE_LIMIT = None  # Set to None to process all samples
# ==============================

SYSTEM_PROMPT = """You are an expert hate speech analyst. Your task is to analyze the provided text and return ONLY a valid JSON object that strictly adheres to the schema below. Do not include explanations, markdown, or any other text outside of the JSON object.

=========================
IMPORTANT INSTRUCTIONS
=========================
Return ONLY a valid JSON object. 
You must output the result in strict JSON syntax — 
use curly braces `{}` and double quotes `"` for all keys and string values.
Do NOT use YAML-like syntax, colons without quotes, triple dashes, or markdown.

1. The output must be **valid JSON**, with no markdown or extra commentary.
2. Use the exact field names and types described below.
3. Each value must follow its correct type:
   - Floats → only for `score`
   - Integers → only for all values in `facets` (must be whole numbers, not floats)
   - Booleans → only for all values in `targets` (`true` or `false`, lowercase)

=========================
OVERALL SCORE
=========================
Produce a single signed float named `"score"` inside `"overall"`. Example:
- NEGATIVE float for supportive content → e.g. `-1.35`
- POSITIVE float for hateful content → e.g. `1.47`
- NEAR ZERO float for neutral content → e.g. `0.12`, `-0.08`
Must be a **standard JSON number**, not a string (e.g. `0.32`, not `"0.32"`).

=========================
FACETS (0-4 SCALE)
=========================
Each facet must be an **integer** from 0 to 4 (no decimals).  
Use this scale strictly:
- 0 = Absent  
- 1 = Mild  
- 2 = Clear  
- 3 = Severe  
- 4 = Extreme  

Example: `"insult": 2`
Do NOT output `"insult": 2.0` or `"insult": "2"`

=========================
TARGETS (BOOLEAN FLAGS)
=========================
Each target field must be a **boolean** (`true` or `false`).
Set a target to `true` ONLY when:
1. The text explicitly mentions or refers to that specific group
2. The text expresses hate, bias, or negative sentiment toward that group
3. The group is the target of the hate speech, not just mentioned neutrally

=========================
JSON SCHEMA (MUST MATCH EXACTLY)
=========================
{
  "overall": {
    "score": 0.00
  },
  "facets": {
    "sentiment": 0,
    "respect": 0,
    "insult": 0,
    "humiliate": 0,
    "status": 0,
    "dehumanize": 0,
    "violence": 0,
    "genocide": 0,
    "attack_defend": 0,
    "hatespeech": 0
  },
  "targets": {
    "target_race_asian": false,
    "target_race_black": false,
    "target_race_latinx": false,
    "target_race_middle_eastern": false,
    "target_race_native_american": false,
    "target_race_pacific_islander": false,
    "target_race_white": false,
    "target_race_other": false,
    "target_religion_atheist": false,
    "target_religion_buddhist": false,
    "target_religion_christian": false,
    "target_religion_hindu": false,
    "target_religion_jewish": false,
    "target_religion_mormon": false,
    "target_religion_muslim": false,
    "target_religion_other": false,
    "target_origin_immigrant": false,
    "target_origin_migrant_worker": false,
    "target_origin_specific_country": false,
    "target_origin_undocumented": false,
    "target_origin_other": false,
    "target_gender_men": false,
    "target_gender_non_binary": false,
    "target_gender_transgender_men": false,
    "target_gender_transgender_unspecified": false,
    "target_gender_transgender_women": false,
    "target_gender_women": false,
    "target_gender_other": false,
    "target_sexuality_bisexual": false,
    "target_sexuality_gay": false,
    "target_sexuality_lesbian": false,
    "target_sexuality_straight": false,
    "target_sexuality_other": false,
    "target_age_children": false,
    "target_age_teenagers": false,
    "target_age_young_adults": false,
    "target_age_middle_aged": false,
    "target_age_seniors": false,
    "target_age_other": false,
    "target_disability_physical": false,
    "target_disability_cognitive": false,
    "target_disability_neurological": false,
    "target_disability_visually_impaired": false,
    "target_disability_hearing_impaired": false,
    "target_disability_unspecific": false,
    "target_disability_other": false
  }
}

=========================
TEXT TO ANALYZE
=========================
{text}

Return ONLY the JSON object. Do not say anything else."""

# --------------------------
# Load Model & Tokenizer
# --------------------------
print("Loading Llama model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
print(f"Device: {model.device}")


# --------------------------
# Load Data
# --------------------------
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------
# Inference Function
# --------------------------
def extract_json(text: str):
    """
    Extract the first valid JSON object from model output text.
    Handles extra tokens after or before JSON.
    """
    matches = list(re.finditer(r"\{", text))
    if not matches:
        return None

    start = matches[0].start()
    stack = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                end = i + 1
                break
    if end is None:
        return None

    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def analyze(entry):
    # Build a chat conversation for the instruction-tuned model
    chat = [
        {"role": "system", "content": "You are an expert hate speech analyst."},
        {"role": "user", "content": SYSTEM_PROMPT.replace("{text}", entry["text"])},
    ]

    # Apply the chat template to format the conversation
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and run inference
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated text
    text_output = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    # Handle possible markdown/code formatting
    if text_output.startswith("```"):
        text_output = text_output.strip("`")
        if text_output.startswith("json"):
            text_output = text_output[len("json"):].strip()
        text_output = text_output.replace("```", "").strip()

    prediction = extract_json(text_output)

    # Get scores for display
    gold_score = entry.get("overall", {}).get("hate_speech_score", "N/A")
    if prediction is None:
        print(f"\nSample {entry['comment_id']}: Failed to parse JSON")
        print(f"Gold score: {gold_score:>6} | Generated: N/A")
    else:
        generated_score = prediction.get("overall", {}).get("score", "N/A")
        print(f"\nSample {entry['comment_id']}:")
        print(f"Gold score: {gold_score:>6} | Generated: {generated_score:>6}")

    return {"id": entry["comment_id"], "prediction": prediction}


# --------------------------
# Run Inference
# --------------------------
def run_inference(entries):
    results = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(entries), BATCH_SIZE), desc="Running inference"):
            batch = entries[i : i + BATCH_SIZE]
            for entry in batch:
                result = analyze(entry)
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure writes in case of interruption
    return results


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    test_data = load_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")
    
    # Apply sample limit if set
    if SAMPLE_LIMIT is not None:
        test_data = test_data[:SAMPLE_LIMIT]
        print(f"Processing first {len(test_data)} samples (SAMPLE_LIMIT={SAMPLE_LIMIT})")
    
    run_inference(test_data)
    print(f"Inference complete. Results written to {OUTPUT_FILE}")