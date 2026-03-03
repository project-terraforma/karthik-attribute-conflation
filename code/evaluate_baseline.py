# evaluate_baseline.py
import argparse
import json
import re
from collections import defaultdict

import pandas as pd


ATTRS = ["names", "categories", "websites", "socials", "emails", "phones", "brand", "addresses"]
LABELS = ["L", "R", "B", "N"]


# -------------------------
# parsing / normalization
# -------------------------
def is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return True
    if s in {"[]", "{}", "[null]"}:
        return True
    # common "empty object" patterns you showed
    if s.replace(" ", "") in {'{"names":{}}', '{"primary":""}', '{"primary":null}'}:
        return True
    return False


def parse_jsonish(x):
    """Your cells often look like JSON arrays/objects encoded as strings."""
    if is_empty(x):
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    # try JSON
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def first_str_from_list(x):
    obj = parse_jsonish(x)
    if obj is None:
        return None
    if isinstance(obj, list) and len(obj) > 0:
        return str(obj[0])
    if isinstance(obj, str):
        return obj
    return str(obj)


def normalize_url(u: str) -> str:
    if u is None:
        return ""
    u = u.strip()
    u = re.sub(r"/+$", "", u)                 # strip trailing slashes
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.lower()


def url_score(u: str) -> float:
    """Higher is better."""
    if not u:
        return -1e9
    raw = u.strip().lower()
    norm = normalize_url(raw)

    score = 0.0

    # punish shorteners
    if any(dom in norm for dom in ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]):
        score -= 5.0

    # reward https
    if raw.startswith("https://"):
        score += 1.0
    elif raw.startswith("http://"):
        score += 0.2

    # punish obvious social-only when comparing "websites"
    if any(dom in norm for dom in ["facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com", "tiktok.com"]):
        score -= 0.5

    # slightly prefer shorter canonical domains/paths (less junk)
    score += max(0.0, 2.0 - (len(norm) / 50.0))
    return score


def normalize_phone(p: str) -> str:
    if p is None:
        return ""
    return re.sub(r"[^\d+]", "", p.strip())


def phone_score(p: str) -> float:
    if not p:
        return -1e9
    s = normalize_phone(p)
    digits = re.sub(r"\D", "", s)

    score = 0.0
    # reward leading + (E.164-like)
    if s.startswith("+"):
        score += 2.0

    # reward plausible length (10-15 digits)
    n = len(digits)
    if 10 <= n <= 15:
        score += 2.0
    else:
        score -= 2.0

    # reward more complete vs partial
    score += min(1.5, n / 10.0)
    return score


def email_score(e: str) -> float:
    if not e:
        return -1e9
    s = e.strip().lower()
    score = 0.0
    # basic validity
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s):
        score += 2.0
    else:
        score -= 2.0
    # prefer business domains over free providers
    if any(s.endswith("@" + dom) or ("@" + dom) in s for dom in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]):
        score -= 0.5
    else:
        score += 0.5
    return score


def names_score(n: str) -> float:
    if not n:
        return -1e9
    s = str(n).strip()
    # prefer "more informative" (a bit longer / more words), but not crazy long
    words = len(re.findall(r"\w+", s))
    score = 0.0
    score += min(3.0, words * 0.6)
    score += min(2.0, len(s) / 20.0)
    # small penalty for all-caps shouting
    if s.isupper() and len(s) >= 6:
        score -= 0.5
    return score


def categories_primary(x):
    obj = parse_jsonish(x)
    if obj is None:
        return None
    if isinstance(obj, dict):
        p = obj.get("primary")
        if p is None:
            return None
        return str(p) if str(p).strip() else None
    # sometimes it's already a string
    if isinstance(obj, str) and obj.strip():
        return obj.strip()
    return None


def categories_score(x) -> float:
    p = categories_primary(x)
    if not p:
        return -1e9
    # prefer non-empty primary; slight preference for more specific-looking tokens
    score = 2.0
    if "_" in p:
        score += 0.2
    return score


def brand_has_name(x) -> bool:
    obj = parse_jsonish(x)
    if obj is None:
        return False
    if isinstance(obj, dict):
        names = obj.get("names")
        if isinstance(names, dict):
            # any non-empty string inside names
            for v in names.values():
                if isinstance(v, str) and v.strip():
                    return True
                if isinstance(v, list) and any(isinstance(it, str) and it.strip() for it in v):
                    return True
        return False
    return False


def brand_score(x) -> float:
    return 2.0 if brand_has_name(x) else -1e9


def addresses_score(x) -> float:
    obj = parse_jsonish(x)
    if obj is None:
        return -1e9
    # expect list of dicts
    if isinstance(obj, list) and obj:
        a = obj[0]
        if isinstance(a, dict):
            score = 0.0
            # reward more complete fields
            for k in ["freeform", "locality", "region", "country", "postcode"]:
                if k in a and isinstance(a[k], str) and a[k].strip():
                    score += 1.0
            # reward ZIP+4 / longer postcodes slightly
            pc = a.get("postcode")
            if isinstance(pc, str):
                score += min(1.0, len(pc.strip()) / 10.0)
            return score
    # fallback
    s = str(obj).strip()
    return 0.5 if s else -1e9


# -------------------------
# prediction per attribute
# -------------------------
def predict_winner(attr: str, left_cell, right_cell) -> str:
    # empty handling
    left_empty = is_empty(left_cell)
    right_empty = is_empty(right_cell)

    if left_empty and right_empty:
        return "N"
    if (not left_empty) and right_empty:
        # special case: treat empty-object brand as empty
        if attr == "brand" and not brand_has_name(left_cell):
            return "N"
        return "L"
    if (not right_empty) and left_empty:
        if attr == "brand" and not brand_has_name(right_cell):
            return "N"
        return "R"

    # parse into comparable strings
    if attr in ["websites", "socials"]:
        l = first_str_from_list(left_cell)
        r = first_str_from_list(right_cell)
        ln, rn = normalize_url(l or ""), normalize_url(r or "")
        if ln and rn and ln == rn:
            return "B"
        ls, rs = url_score(l or ""), url_score(r or "")
        if abs(ls - rs) < 0.25:
            return "B"
        return "L" if ls > rs else "R"

    if attr == "phones":
        l = first_str_from_list(left_cell)
        r = first_str_from_list(right_cell)
        ln = normalize_phone(l or "")
        rn = normalize_phone(r or "")
        # if same digits => both
        ld = re.sub(r"\D", "", ln)
        rd = re.sub(r"\D", "", rn)
        if ld and rd and ld == rd:
            return "B"
        ls, rs = phone_score(l or ""), phone_score(r or "")
        if abs(ls - rs) < 0.25:
            return "B"
        return "L" if ls > rs else "R"

    if attr == "emails":
        l = first_str_from_list(left_cell)
        r = first_str_from_list(right_cell)
        if (l or "").strip().lower() == (r or "").strip().lower() and (l or "").strip():
            return "B"
        ls, rs = email_score(l or ""), email_score(r or "")
        if abs(ls - rs) < 0.25:
            return "B"
        return "L" if ls > rs else "R"

    if attr == "names":
        l = categories_primary(left_cell) if False else None  # placeholder (unused)
        l = parse_jsonish(left_cell)
        r = parse_jsonish(right_cell)

        # names are stored like {"primary":"..."} in your sample
        def get_primary_name(obj):
            if obj is None:
                return ""
            if isinstance(obj, dict) and "primary" in obj and isinstance(obj["primary"], str):
                return obj["primary"].strip()
            return str(obj).strip()

        lp = get_primary_name(l)
        rp = get_primary_name(r)
        if lp and rp and lp.lower() == rp.lower():
            return "B"
        ls, rs = names_score(lp), names_score(rp)
        if abs(ls - rs) < 0.25:
            return "B"
        return "L" if ls > rs else "R"

    if attr == "categories":
        lp = categories_primary(left_cell) or ""
        rp = categories_primary(right_cell) or ""
        if lp and rp and lp == rp:
            return "B"
        ls, rs = categories_score(left_cell), categories_score(right_cell)
        if abs(ls - rs) < 0.1:
            return "B"
        return "L" if ls > rs else "R"

    if attr == "brand":
        l_has = brand_has_name(left_cell)
        r_has = brand_has_name(right_cell)
        if not l_has and not r_has:
            return "N"
        if l_has and not r_has:
            return "L"
        if r_has and not l_has:
            return "R"
        # both have something; if identical strings, B else B (brand is hard w/o more parsing)
        return "B"

    if attr == "addresses":
        l_obj = parse_jsonish(left_cell)
        r_obj = parse_jsonish(right_cell)
        if l_obj == r_obj and l_obj is not None:
            return "B"
        ls, rs = addresses_score(left_cell), addresses_score(right_cell)
        if abs(ls - rs) < 0.25:
            return "B"
        return "L" if ls > rs else "R"

    # fallback
    return "B"


# -------------------------
# metrics
# -------------------------
def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    # compute per-class f1, average over classes that appear in y_true
    present = set(y_true)
    f1s = []
    for c in labels:
        if c not in present:
            continue
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_csv", required=True, help="Path to golden dataset CSV")
    ap.add_argument("--out_pred_csv", default=None, help="Optional path to write predictions CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.gold_csv)

    # find which attrs actually have *_gold columns in your file
    available_attrs = [a for a in ATTRS if f"{a}_gold" in df.columns]
    if not available_attrs:
        raise ValueError("No *_gold columns found. Expected columns like websites_gold, phones_gold, etc.")

    results = []
    all_true = []
    all_pred = []

    for attr in available_attrs:
        left_col = f"{attr}_left"
        right_col = f"{attr}_right"
        gold_col = f"{attr}_gold"

        sub = df[[left_col, right_col, gold_col]].copy()

        # keep only labeled rows (L/R/B/N)
        sub[gold_col] = sub[gold_col].astype(str).str.strip()
        sub = sub[sub[gold_col].isin(LABELS)]

        if len(sub) == 0:
            continue

        y_true = sub[gold_col].tolist()
        y_pred = [
            predict_winner(attr, l, r)
            for l, r in zip(sub[left_col].tolist(), sub[right_col].tolist())
        ]

        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        f1 = macro_f1(y_true, y_pred)

        results.append((attr, len(y_true), acc, f1))
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        # store predictions back if requested
        df[f"{attr}_pred"] = [
            predict_winner(attr, l, r)
            for l, r in zip(df[left_col].tolist(), df[right_col].tolist())
        ]

    # print results
    print("\nPer-attribute scores (baseline rules):")
    print("attr       n_labeled   accuracy   macro_f1")
    for attr, n, acc, f1 in results:
        print(f"{attr:<10} {n:>9}   {acc:>7.3f}    {f1:>7.3f}")

    if all_true:
        overall_acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
        overall_f1 = macro_f1(all_true, all_pred)
        print("\nOverall (micro across all labeled cells):")
        print(f"  accuracy = {overall_acc:.3f}")
        print(f"  macro_f1 = {overall_f1:.3f}")

    if args.out_pred_csv:
        df.to_csv(args.out_pred_csv, index=False)
        print(f"\nWrote predictions to: {args.out_pred_csv}")


if __name__ == "__main__":
    main()