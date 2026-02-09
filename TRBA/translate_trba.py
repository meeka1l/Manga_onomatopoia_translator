from sfx_utils import load_sfx_dict, translate_sfx
from parse_trba import parse_trba_output

SFX_DICT_FILE = "sfx_dict.json"
TRBA_OUTPUT_FILE = "trba_output.txt"
CONF_THRESHOLD = 0.85
OUTPUT_CSV = "sfx_translation.csv"

sfx_dict = load_sfx_dict(SFX_DICT_FILE)

# THIS IS THE CRITICAL CHANGE:
# Instead of manually listing 3 rows, read all TRBA output
trba_results = parse_trba_output(TRBA_OUTPUT_FILE)

translated_rows = []

for img, jp, conf in trba_results:
    norm, en = translate_sfx(jp, sfx_dict)

    if conf < CONF_THRESHOLD:
        en += " (CHECK)"

    translated_rows.append({
        "image": img,
        "jp": jp,
        "normalized": norm,
        "en": en,
        "confidence": conf
    })

# Save CSV
import csv
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "jp", "normalized", "en", "confidence"])
    writer.writeheader()
    writer.writerows(translated_rows)

print(f"✓ Saved {OUTPUT_CSV} ({len(translated_rows)} entries)")
