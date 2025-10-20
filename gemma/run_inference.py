import json, os, asyncio
from tqdm import tqdm
from google import genai
import google.generativeai as genai
from dotenv import load_dotenv

# ========== CONFIG ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemma-3-27b-it")

TEST_FILE = "../data/test.jsonl"
OUTPUT_FILE = "./baseline_data/gemma_baseline_outputs.jsonl"
BATCH_SIZE = 1
SLEEP_TIME = 0
# =============================

SYSTEM_PROMPT = """You are an expert hate speech analyst. Your task is to analyze the provided text and return ONLY a valid JSON object that strictly adheres to the schema below. Do not include explanations, markdown, or any other text outside of the JSON object.

--- IMPORTANT INSTRUCTIONS (READ CAREFULLY) ---
1. FIRST produce a single numeric signed float value named "score" (e.g. -1.52, 0.00, 1.23).  
   - Use a NEGATIVE float for supportive content (e.g. -1.5).  
   - Use a POSITIVE float for hateful content (e.g. 1.52).  
   - Use a value CLOSE TO ZERO for neutral content (e.g. -0.10, 0.00, 0.25).  
   - The score must be a standard JSON number (not a string).

2. AFTER you produce the numeric "score", set the "label" based *only* on these exact thresholds (do not override or invent other thresholds):  
   - "hateful": If score > 0.5  
   - "supportive": If score < -1.0  
   - "neutral": If score is between -1.0 and 0.5 (inclusive)

3. The JSON MUST reflect the score and the label consistent with the thresholds above. Do not contradict the thresholds.

4. Return ONLY valid JSON. No surrounding text, no markdown, no explanation.

--- SCORING RUBRIC ---
Use integers from 0 to 4 for all facets based on this scale:
0 = Absent
1 = Mild
2 = Clear
3 = Severe
4 = Extreme

--- SCHEMA DEFINITIONS ---
1. overall.score: A signed continuous float (negative for supportive, positive for hateful, near-zero for neutral). This must appear first in the overall object before the label.
2. overall.label: A categorical label derived exactly from the score using the thresholds listed above.
3. facets: 10 integer fields (0-4) as listed below.
4. targets: booleans for all target groups; set true only if that group is explicitly targeted in the text.

--- JSON FORMAT TO RETURN (MUST MATCH EXACTLY) ---
{
  "overall": {
    "score": 0.00,
    "label": "neutral"
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

--- TEXT TO ANALYZE ---
{text}
"""


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def analyze(entry):
    prompt = SYSTEM_PROMPT.replace("{text}", entry["text"])
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        output = response.text.strip()

        if output.startswith("```"):
            output = output.strip("`")
            if output.startswith("json"):
                output = output[len("json") :].strip()
            output = output.replace("```", "").strip()

        return {"id": entry["comment_id"], "prediction": json.loads(output)}
    except Exception as e:
        print(f"Error on {entry['comment_id']}: {e}")
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
