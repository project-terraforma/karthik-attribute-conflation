# evaluate_final_rules.py
import argparse
import json
import re
from urllib.parse import urlparse

import pandas as pd


ATTRS = ["names", "categories", "websites", "socials", "emails", "phones", "brand", "addresses"]
LABELS = ["L", "R", "B", "N"]

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"}
SOCIAL_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "tiktok.com", "youtube.com"
}
FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}

# simple source priors; edit if you learn some sources are better
SOURCE_PRIOR = {
    "google": 0.30,
    "microsoft": 0.20,
    "meta": 0.10,
    "yelp": 0.15,
    "tripadvisor": 0.10,
}


# -------------------------
# generic helpers
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
    if s.replace(" ", "") in {
        '{"names":{}}',
        '{"primary":""}',
        '{"primary":null}',
        '{"alternate":[]}',
    }:
        return True
    return False


def parse_jsonish(x):
    if is_empty(x):
        return None
    if isinstance(x, (dict, list, int, float)):
        return x
    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def first_item(x):
    obj = parse_jsonish(x)
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj[0] if obj else None
    return obj


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def jaccard_tokens(a: str, b: str) -> float:
    sa = set(normalize_text(a).split())
    sb = set(normalize_text(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_confidence(x) -> float:
    if is_empty(x):
        return 0.0
    obj = parse_jsonish(x)
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except Exception:
            return 0.0
    return 0.0


def extract_source_score(x) -> float:
    """
    x may be a string, dict, or list of dicts like:
    [{"dataset":"Microsoft", ...}]
    """
    if is_empty(x):
        return 0.0
    obj = parse_jsonish(x)
    names = []

    if isinstance(obj, str):
        names = [obj.lower()]
    elif isinstance(obj, dict):
        if "dataset" in obj:
            names = [str(obj["dataset"]).lower()]
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and "dataset" in it:
                names.append(str(it["dataset"]).lower())

    best = 0.0
    for n in names:
        for k, v in SOURCE_PRIOR.items():
            if k in n:
                best = max(best, v)
    return best


def tie_breaker(left_source, right_source, left_conf, right_conf):
    ls = extract_source_score(left_source) + extract_confidence(left_conf)
    rs = extract_source_score(right_source) + extract_confidence(right_conf)
    if abs(ls - rs) < 0.08:
        return "B"
    return "L" if ls > rs else "R"


# -------------------------
# websites / socials
# -------------------------
def get_url(x):
    obj = first_item(x)
    if obj is None:
        return ""
    return str(obj).strip()


def canonical_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    u = re.sub(r"/+$", "", u)
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.lower()


def url_domain(u: str) -> str:
    if not u:
        return ""
    raw = u if re.match(r"^https?://", u, flags=re.I) else "https://" + u
    try:
        return urlparse(raw).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def url_score(u: str, is_social=False) -> float:
    if not u:
        return -1e9
    raw = u.strip()
    canon = canonical_url(raw)
    domain = url_domain(raw)

    score = 0.0

    if raw.lower().startswith("https://"):
        score += 1.0
    elif raw.lower().startswith("http://"):
        score += 0.2

    if domain in SHORTENERS:
        score -= 5.0

    if not is_social and domain in SOCIAL_DOMAINS:
        score -= 1.0

    # shorter / more canonical often better
    score += max(0.0, 1.8 - len(canon) / 60.0)

    # penalize obvious query junk
    if "?" in raw:
        score -= 0.3

    return score


# -------------------------
# phones
# -------------------------
def get_phone(x):
    obj = first_item(x)
    if obj is None:
        return ""
    return str(obj).strip()


def normalize_phone(p: str) -> str:
    return re.sub(r"[^\d+]", "", p or "")


def digits_only(p: str) -> str:
    return re.sub(r"\D", "", p or "")


def phone_score(p: str) -> float:
    if not p:
        return -1e9
    s = normalize_phone(p)
    d = digits_only(s)
    n = len(d)

    score = 0.0
    if s.startswith("+"):
        score += 2.0
    if 10 <= n <= 15:
        score += 2.0
    else:
        score -= 2.5
    score += min(1.5, n / 10.0)
    return score


# -------------------------
# emails
# -------------------------
def get_email(x):
    obj = first_item(x)
    if obj is None:
        return ""
    return str(obj).strip().lower()


def email_domain(e: str) -> str:
    parts = (e or "").split("@")
    return parts[1].lower() if len(parts) == 2 else ""


def email_score(e: str) -> float:
    if not e:
        return -1e9
    score = 0.0
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
        score += 2.0
    else:
        score -= 2.0
    domain = email_domain(e)
    if domain in FREE_EMAIL_DOMAINS:
        score -= 0.5
    elif domain:
        score += 0.5
    return score


# -------------------------
# names
# -------------------------
def get_primary_name(x) -> str:
    obj = parse_jsonish(x)
    if obj is None:
        return ""
    if isinstance(obj, dict):
        if isinstance(obj.get("primary"), str):
            return obj["primary"].strip()
    return str(obj).strip()


def names_score(name: str) -> float:
    if not name:
        return -1e9
    score = 0.0
    words = len(re.findall(r"\w+", name))
    score += min(2.5, words * 0.6)
    score += min(1.5, len(name) / 20.0)
    if name.isupper() and len(name) > 5:
        score -= 0.5
    return score


# -------------------------
# categories
# -------------------------
def categories_primary(x):
    obj = parse_jsonish(x)
    if obj is None:
        return ""
    if isinstance(obj, dict):
        p = obj.get("primary")
        return str(p).strip() if p else ""
    return str(obj).strip()


def categories_alternates_count(x) -> int:
    obj = parse_jsonish(x)
    if isinstance(obj, dict):
        alts = obj.get("alternate") or obj.get("alternates") or []
        if isinstance(alts, list):
            return len(alts)
    return 0


def categories_score(x) -> float:
    p = categories_primary(x)
    if not p:
        return -1e9
    score = 2.0
    # slight preference for more specific category strings
    score += 0.15 * p.count("_")
    score += 0.05 * categories_alternates_count(x)
    return score


# -------------------------
# brand
# -------------------------
def brand_name(x) -> str:
    obj = parse_jsonish(x)
    if obj is None or not isinstance(obj, dict):
        return ""
    names = obj.get("names")
    if not isinstance(names, dict):
        return ""
    if isinstance(names.get("primary"), str):
        return names["primary"].strip()
    for v in names.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def brand_score(x) -> float:
    b = brand_name(x)
    return 2.0 if b else -1e9


# -------------------------
# addresses
# -------------------------
def first_address(x):
    obj = parse_jsonish(x)
    if isinstance(obj, list) and obj:
        return obj[0]
    return None


def address_string(a) -> str:
    if not isinstance(a, dict):
        return ""
    parts = []
    for k in ["freeform", "locality", "region", "postcode", "country"]:
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)


def address_postcode(a) -> str:
    if not isinstance(a, dict):
        return ""
    pc = a.get("postcode")
    return pc.strip() if isinstance(pc, str) else ""


def addresses_score(x) -> float:
    a = first_address(x)
    if not isinstance(a, dict):
        return -1e9
    score = 0.0
    for k in ["freeform", "locality", "region", "country", "postcode"]:
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            score += 1.0
    pc = address_postcode(a)
    if pc:
        score += min(1.0, len(pc) / 10.0)  # ZIP+4 style gets slight boost
    return score


# -------------------------
# main prediction logic
# -------------------------
def predict_winner(attr, left_cell, right_cell, left_source=None, right_source=None, left_conf=None, right_conf=None):
    left_empty = is_empty(left_cell)
    right_empty = is_empty(right_cell)

    if left_empty and right_empty:
        return "N"
    if not left_empty and right_empty:
        if attr == "brand" and not brand_name(left_cell):
            return "N"
        return "L"
    if not right_empty and left_empty:
        if attr == "brand" and not brand_name(right_cell):
            return "N"
        return "R"

    if attr == "websites":
        l = get_url(left_cell)
        r = get_url(right_cell)
        lc, rc = canonical_url(l), canonical_url(r)
        if lc and rc and lc == rc:
            return "B"
        ls, rs = url_score(l, is_social=False), url_score(r, is_social=False)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "socials":
        l = get_url(left_cell)
        r = get_url(right_cell)
        lc, rc = canonical_url(l), canonical_url(r)
        if lc and rc and lc == rc:
            return "B"
        ls, rs = url_score(l, is_social=True), url_score(r, is_social=True)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "phones":
        l = get_phone(left_cell)
        r = get_phone(right_cell)
        if digits_only(l) and digits_only(l) == digits_only(r):
            return "B"
        ls, rs = phone_score(l), phone_score(r)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "emails":
        l = get_email(left_cell)
        r = get_email(right_cell)
        if l and r and l == r:
            return "B"
        ls, rs = email_score(l), email_score(r)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "names":
        l = get_primary_name(left_cell)
        r = get_primary_name(right_cell)
        if l and r and normalize_text(l) == normalize_text(r):
            return "B"
        # if nearly same tokens, count as both
        if jaccard_tokens(l, r) >= 0.8 and l and r:
            return "B"
        ls, rs = names_score(l), names_score(r)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "categories":
        l = categories_primary(left_cell)
        r = categories_primary(right_cell)
        if l and r and l == r:
            return "B"
        ls, rs = categories_score(left_cell), categories_score(right_cell)
        if abs(ls - rs) < 0.15:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    if attr == "brand":
        l = brand_name(left_cell)
        r = brand_name(right_cell)
        if not l and not r:
            return "N"
        if l and not r:
            return "L"
        if r and not l:
            return "R"
        if normalize_text(l) == normalize_text(r):
            return "B"
        return tie_breaker(left_source, right_source, left_conf, right_conf)

    if attr == "addresses":
        la = first_address(left_cell)
        ra = first_address(right_cell)

        if la and ra:
            # same address, just formatting/language differences
            if normalize_text(address_string(la)) == normalize_text(address_string(ra)):
                return "B"
            if jaccard_tokens(address_string(la), address_string(ra)) >= 0.8 and address_postcode(la) == address_postcode(ra):
                return "B"

        ls, rs = addresses_score(left_cell), addresses_score(right_cell)
        if abs(ls - rs) < 0.25:
            return tie_breaker(left_source, right_source, left_conf, right_conf)
        return "L" if ls > rs else "R"

    return "B"


# -------------------------
# metrics
# -------------------------
def macro_f1(y_true, y_pred, labels=LABELS):
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
    ap.add_argument("--gold_csv", required=True)
    ap.add_argument("--out_pred_csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.gold_csv)

    # optional metadata columns
    left_source_col = "left_source" if "left_source" in df.columns else None
    right_source_col = "right_source" if "right_source" in df.columns else None
    left_conf_col = "confidence_left" if "confidence_left" in df.columns else None
    right_conf_col = "confidence_right" if "confidence_right" in df.columns else None

    available_attrs = [a for a in ATTRS if f"{a}_gold" in df.columns]

    results = []
    all_true, all_pred = [], []

    for attr in available_attrs:
        left_col = f"{attr}_left"
        right_col = f"{attr}_right"
        gold_col = f"{attr}_gold"

        if left_col not in df.columns or right_col not in df.columns:
            continue

        sub = df.copy()
        sub[gold_col] = sub[gold_col].astype(str).str.strip()
        sub = sub[sub[gold_col].isin(LABELS)]

        if len(sub) == 0:
            continue

        preds = []
        for _, row in sub.iterrows():
            pred = predict_winner(
                attr,
                row[left_col],
                row[right_col],
                row[left_source_col] if left_source_col else None,
                row[right_source_col] if right_source_col else None,
                row[left_conf_col] if left_conf_col else None,
                row[right_conf_col] if right_conf_col else None,
            )
            preds.append(pred)

        y_true = sub[gold_col].tolist()
        y_pred = preds

        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        f1 = macro_f1(y_true, y_pred)

        results.append((attr, len(y_true), acc, f1))
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        df[f"{attr}_pred"] = df.apply(
            lambda row: predict_winner(
                attr,
                row[left_col],
                row[right_col],
                row[left_source_col] if left_source_col else None,
                row[right_source_col] if right_source_col else None,
                row[left_conf_col] if left_conf_col else None,
                row[right_conf_col] if right_conf_col else None,
            ),
            axis=1,
        )

    print("\nPer-attribute scores (final rules):")
    print("attr       n_labeled   accuracy   macro_f1")
    for attr, n, acc, f1 in results:
        print(f"{attr:<10} {n:>9}   {acc:>7.3f}    {f1:>7.3f}")

    if all_true:
        overall_acc = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
        overall_f1 = macro_f1(all_true, all_pred)
        print("\nOverall:")
        print(f"  accuracy = {overall_acc:.3f}")
        print(f"  macro_f1 = {overall_f1:.3f}")

    if args.out_pred_csv:
        df.to_csv(args.out_pred_csv, index=False)
        print(f"\nWrote predictions to: {args.out_pred_csv}")


if __name__ == "__main__":
    main()