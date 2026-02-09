import json
import re
import unicodedata

# ---------- Normalization ----------
def normalize_sfx(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'[！!…・\.]', '', s)
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)
    return s.strip()

# ---------- Dictionary Loader ----------
def load_sfx_dict(path="sfx_dict.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Translation ----------
def translate_sfx(jp_text, sfx_dict):
    norm = normalize_sfx(jp_text)
    return norm, sfx_dict.get(norm, "UNKNOWN")
