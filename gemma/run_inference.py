import json, os, asyncio
from tqdm import tqdm
from google import genai
import google.generativeai as genai
from dotenv import load_dotenv

# ========== CONFIG ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemma-3-1b-it")

TEST_FILE = "../data/test.jsonl"
OUTPUT_FILE = "./baseline_data/gemma_baseline_outputs.jsonl"
BATCH_SIZE = 1
SLEEP_TIME = 2
# =============================

SYSTEM_PROMPT = """You are an expert hate speech analyst. Your task is to analyze the provided text and return ONLY a valid JSON object that strictly follows the schema below. 
Do not include any explanations, markdown formatting, or text outside of the JSON object.

=========================
IMPORTANT INSTRUCTIONS
=========================
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
Each target field must be a **boolean** (`true` or `false`) and only set to true if that group is explicitly targeted.

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
"""


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_json(text: str):
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return None
    candidate = cleaned[first_brace : last_brace + 1]

    stack = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                try:
                    return json.loads(candidate[: i + 1])
                except json.JSONDecodeError:
                    continue
    try:
        return json.loads(candidate)
    except Exception:
        return None


async def analyze(entry):
    prompt = SYSTEM_PROMPT.replace("{text}", entry["text"])
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        output = response.text.strip()

        parsed = extract_json(output)
        if parsed is None:
            print(f"⚠️ Failed to parse JSON for {entry['comment_id']}")
        try:
            score = parsed["overall"]["score"]
            if score > 0.5:
                label = "hateful"
            elif score < -1.0:
                label = "supportive"
            else:
                label = "neutral"
            parsed["overall"]["label"] = label
        except Exception as e:
            print(f"⚠️ Label generation error for {entry['comment_id']}: {e}")

        return {"id": entry["comment_id"], "prediction": parsed}
    except Exception as e:
        print(f"❌ Error on {entry['comment_id']}: {e}")
        return {"id": entry["comment_id"], "prediction": None}


async def run_inference(entries):
    results = []
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(entries), BATCH_SIZE), desc="Running inference"):
            batch = entries[i : i + BATCH_SIZE]
            batch_results = await asyncio.gather(*[analyze(e) for e in batch])
            for result in batch_results:
                results.append(result)
                f.write(json.dumps(result) + "\n")
            await asyncio.sleep(SLEEP_TIME)
    return results


async def main():
    test_data = load_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")
    await run_inference(test_data)
    print(f"✅ Inference complete. Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
