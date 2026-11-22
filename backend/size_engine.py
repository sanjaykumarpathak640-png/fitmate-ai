"""FitMate AI Size Engine (Python port of the engine in frontend/index.html).

ease = garment measurement - body measurement (cm, full circumference).
Each size is scored by how far its ease falls outside the target range for the
user's preferred fit (too tight costs 1.5x, too loose 1x), weighted per
measurement. Each garment shifts the targets for its designed cut (oversized,
skinny, flared...) and stretch fabric lowers the bottom of the range.
The lowest score wins. Missing data lowers confidence instead of guessing.
tests/test_api.py checks that this file and the browser engine agree on
2,100 cases.
"""
from __future__ import annotations

import math

SIZES = ("XS", "S", "M", "L", "XL", "XXL")
FITS = ("slim", "regular", "relaxed")

RANGES = {
    "top": {"chest": {"slim": (0, 4), "regular": (4, 8), "relaxed": (8, 14)},
            "waist": {"slim": (2, 8), "regular": (8, 14), "relaxed": (14, 22)},
            "shoulder": {"slim": (-1, 2), "regular": (-1, 2), "relaxed": (-1, 3)}},
    "outer": {"chest": {"slim": (6, 10), "regular": (10, 14), "relaxed": (14, 20)},
              "waist": {"slim": (8, 14), "regular": (14, 20), "relaxed": (20, 28)},
              "shoulder": {"slim": (0, 3), "regular": (0, 3), "relaxed": (0, 4)}},
    "dress": {"chest": {"slim": (0, 4), "regular": (3, 7), "relaxed": (6, 12)},
              "waist": {"slim": (0, 4), "regular": (3, 8), "relaxed": (7, 14)},
              "hip": {"slim": (2, 6), "regular": (4, 10), "relaxed": (8, 16)}},
    "bottom": {"waist": {"slim": (0, 2), "regular": (1, 4), "relaxed": (3, 6)},
               "hip": {"slim": (2, 6), "regular": (4, 10), "relaxed": (8, 14)}},
}

WEIGHTS = {
    "top": {"chest": 0.5, "waist": 0.3, "shoulder": 0.2},
    "outer": {"chest": 0.5, "waist": 0.3, "shoulder": 0.2},
    "dress": {"chest": 0.4, "waist": 0.35, "hip": 0.25},
    "bottom": {"waist": 0.6, "hip": 0.4},
}


def r1(x: float) -> float:
    """Round to 0.1 the way JavaScript's Math.round does (half up)."""
    return math.floor(x * 10 + 0.5) / 10


def target(product: dict, key: str, fit: str) -> tuple[float, float]:
    lo, hi = RANGES[product["kind"]][key][fit]
    shift = (product.get("shift") or {}).get(key, 0) or 0
    stretch = 0 if key == "shoulder" else (product.get("stretch") or 0)
    return lo + shift - stretch, hi + shift


def penalty(ease: float, rng: tuple[float, float]) -> float:
    lo, hi = rng
    if ease < lo:
        return (lo - ease) * 1.5  # too tight is worse than too loose
    if ease > hi:
        return ease - hi
    return 0.0


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def evaluate_size(product: dict, size: str, profile: dict, fit: str, use_true: bool = False) -> dict:
    kind = product["kind"]
    row = (product["true"] if use_true else product["chart"]).get(size, {})
    per, tw, ts = [], 0.0, 0.0
    for key, w in WEIGHTS[kind].items():
        g, b = row.get(key), _num(profile.get(key))
        if g is None or b is None:
            continue
        ease = r1(g - b)
        rng = target(product, key, fit)
        d = penalty(ease, rng)
        lo0, hi0 = RANGES[kind][key][fit]
        centre = (lo0 + hi0) / 2 + ((product.get("shift") or {}).get(key, 0) or 0)  # stretch widens, never moves the centre
        per.append({"measurement": key, "body": b, "garment": g, "ease": ease, "target": list(rng), "centre": centre, "penalty": d})
        tw += w
        ts += w * d
    return {"size": size, "per": per, "score": ts / tw if tw else None}


def recommend(product: dict, profile: dict, fit: str | None = None) -> dict:
    fit = fit or profile.get("fit") or "regular"
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}")
    kind = product["kind"]
    keys = list(WEIGHTS[kind])
    in_chart = [k for k in keys if any(product["chart"].get(s, {}).get(k) is not None for s in SIZES)]
    missing_chart = [k for k in keys if k not in in_chart]
    missing_user = [k for k in in_chart if _num(profile.get(k)) is None]
    used = [k for k in in_chart if _num(profile.get(k)) is not None]
    rows = [evaluate_size(product, s, profile, fit) for s in SIZES]
    out = {"product_id": product["id"], "fit": fit, "rows": rows, "missing_in_chart": missing_chart,
           "missing_from_profile": missing_user, "used": used}
    if not used:
        return {**out, "size": None, "match": "None", "completeness": 0.0, "confidence": "Low",
                "reasons": ["No usable measurements: add them to the avatar."]}

    def cd(r):
        return sum(abs(x["ease"] - x["centre"]) for x in r["per"])

    best = rows[0]
    for r in rows[1:]:  # same order and tie-break as the browser engine
        if r["score"] < best["score"] - 1e-9 or (abs(r["score"] - best["score"]) < 1e-9 and cd(r) < cd(best) - 1e-9):
            best = r
    completeness = r1(sum(WEIGHTS[kind][k] for k in used))
    match = "Strong" if best["score"] <= 0.5 else "Close" if best["score"] <= 2 else "Weak"
    if match == "Strong" and completeness >= 0.99:
        confidence = "High"
    elif match != "Weak" and completeness >= 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"
    return {**out, "size": best["size"], "score": round(best["score"], 3), "match": match,
            "completeness": completeness, "confidence": confidence,
            "reasons": reasons(best, product, missing_chart, missing_user)}


def reasons(best: dict, product: dict, missing_chart: list, missing_user: list) -> list[str]:
    out = []
    for x in best["per"]:
        lo, hi = x["target"]
        base = f"{x['measurement'].capitalize()}: you {x['body']:g} cm, size {best['size']} {x['garment']:g} cm, ease {x['ease']:g} cm"
        if x["penalty"] == 0:
            out.append(f"{base}, within the target ({lo:g} to {hi:g} cm).")
        elif x["ease"] < lo:
            out.append(f"{base}, {r1(lo - x['ease']):g} cm tighter than the target.")
        else:
            out.append(f"{base}, {r1(x['ease'] - hi):g} cm looser than the target.")
    if product.get("stretch"):
        out.append(f"Stretch fabric: up to {product['stretch']} cm of negative ease is allowed.")
    out += [f"{k.capitalize()}: not in this brand's size chart, so confidence is capped." for k in missing_chart]
    out += [f"{k.capitalize()}: not in your profile; add it to raise confidence." for k in missing_user]
    return out


def all_fits(product: dict, profile: dict) -> dict:
    return {f: recommend(product, profile, f)["size"] for f in FITS}


def fit_flags(product: dict, size: str, profile: dict) -> list[dict]:
    """Where the chosen size is tight or clearly loose on this body (same rule as the avatar markers)."""
    fit = profile.get("fit") or "regular"
    flags = []
    for x in evaluate_size(product, size, profile, fit, use_true=True)["per"]:
        lo, hi = x["target"]
        if x["ease"] < lo - 0.01:
            flags.append({"measurement": x["measurement"], "status": "tight", "by_cm": r1(lo - x["ease"])})
        elif x["ease"] > hi + 4:
            flags.append({"measurement": x["measurement"], "status": "loose", "by_cm": r1(x["ease"] - hi)})
    return flags
