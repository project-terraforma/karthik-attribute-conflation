import pandas as pd
import json
import re

ATTRS = [
    "names", "categories", "confidence",
    "websites", "socials", "emails", "phones",
    "brand", "addresses"
]


# ---------------------------
# NORMAL VERSION (machine)
# ---------------------------
def norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


# ---------------------------
# READABLE VERSION HELPERS
# ---------------------------

def pretty(x, attr):
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, float):
        return str(x)

    if attr == "websites" or attr == "socials":
        if isinstance(x, list):
            return " | ".join(clean_url(u) for u in x)
        return clean_url(str(x))

    if attr == "phones":
        if isinstance(x, list):
            return " | ".join(clean_phone(p) for p in x)
        return clean_phone(str(x))

    if attr == "emails":
        if isinstance(x, list):
            return " | ".join(x)
        return str(x)

    if attr == "names":
        return extract_name(x)

    if attr == "categories":
        return extract_category(x)

    if attr == "addresses":
        return extract_address(x)

    if attr == "brand":
        return extract_brand(x)

    return str(x)


def clean_url(u):
    if not isinstance(u, str):
        return str(u)
    u = re.sub(r'^https?://', '', u)
    u = re.sub(r'^www\.', '', u)
    u = u.rstrip("/")
    return u


def clean_phone(p):
    return re.sub(r'[^\d+]', '', str(p))


def extract_name(x):
    if isinstance(x, dict):
        if "primary" in x:
            return x["primary"]
        if "common" in x:
            return str(x["common"])
    if isinstance(x, list):
        return " | ".join(str(i) for i in x)
    return str(x)


def extract_category(x):
    if isinstance(x, dict):
        return x.get("primary", str(x))
    return str(x)


def extract_address(x):
    if isinstance(x, list):
        return " | ".join(str(i) for i in x)
    return str(x)


def extract_brand(x):
    if isinstance(x, dict):
        return x.get("names", str(x))
    return str(x)


# ---------------------------
# MAIN
# ---------------------------

def main():

    df = pd.read_parquet("./data/project_a_samples.parquet")

    long_rows = []
    long_readable_rows = []

    for i, r in df.iterrows():

        left_id = r["base_id"]
        right_id = r["id"]

        left_source = norm(r.get("base_sources"))
        right_source = norm(r.get("sources"))

        wide_row = {
            "pair_id": i,
            "left_id": left_id,
            "right_id": right_id,
            "left_source": left_source,
            "right_source": right_source,
        }

        for attr in ATTRS:

            left_val = norm(r.get(f"base_{attr}"))
            right_val = norm(r.get(attr))

            # MACHINE LONG
            long_rows.append({
                "pair_id": i,
                "left_id": left_id,
                "right_id": right_id,
                "left_source": left_source,
                "right_source": right_source,
                "attr": attr,
                "left_values": left_val,
                "right_values": right_val,
            })

            # HUMAN LONG
            long_readable_rows.append({
                "pair_id": i,
                "attr": attr,
                "left": pretty(left_val, attr),
                "right": pretty(right_val, attr)
            })

            # HUMAN WIDE
            wide_row[f"{attr}_left"] = pretty(left_val, attr)
            wide_row[f"{attr}_right"] = pretty(right_val, attr)

        if i == 0:
            wide_rows = []

        wide_rows.append(wide_row)


    long_df = pd.DataFrame(long_rows)
    long_df.to_csv("data/processed/pairs_long.csv", index=False)

    long_readable_df = pd.DataFrame(long_readable_rows)
    long_readable_df.to_csv("data/processed/pairs_long_readable.csv", index=False)

    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv("data/processed/pairs_wide_readable.csv", index=False)


    print("Created:")
    print("pairs_long.csv")
    print("pairs_long_readable.csv")
    print("pairs_wide_readable.csv")


if __name__ == "__main__":
    main()
