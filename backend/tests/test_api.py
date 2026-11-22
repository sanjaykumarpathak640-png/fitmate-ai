import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["FITMATE_TRYON_MOCK"] = "1"
HERE = Path(__file__).resolve().parent
os.environ["FITMATE_TESTS_FILE"] = str(HERE / "_tmp_user_tests.jsonl")
sys.path.insert(0, str(HERE.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from main import CATALOG, PRODUCTS, app  # noqa: E402
from size_engine import SIZES, all_fits, fit_flags, penalty, recommend  # noqa: E402

client = TestClient(app)
BOY = {"gender": "boy", "height": 172, "chest": 96, "waist": 82, "hip": 98, "shoulder": 45, "fit": "regular"}
BY_NAME = {(p["gender"], p["name"]): p for p in CATALOG}


def test_catalog_is_100_garments_50_50():
    assert len(CATALOG) == 100
    assert sum(p["gender"] == "boy" for p in CATALOG) == 50
    assert all(len(p["colors"]) >= 3 and set(p["true"]) == set(SIZES) for p in CATALOG)


def test_browser_and_python_engines_agree():
    cases = json.loads((HERE / "js_parity_cases.json").read_text())
    assert len(cases) == 2100
    bad = []
    for c in cases:
        r = recommend(PRODUCTS[c["pid"]], c["profile"])
        if (r["size"], r["match"], r["confidence"]) != (c["size"], c["match"], c["confidence"]):
            bad.append((c, r["size"], r["match"], r["confidence"]))
    assert not bad, bad[:3]


def test_proposal_example_regular_m_relaxed_l():
    shirt = BY_NAME[("boy", "Oxford Cotton Shirt")]
    r = recommend(shirt, BOY)
    assert r["size"] == "M" and r["confidence"] == "High"
    assert all_fits(shirt, BOY)["relaxed"] == "L"


@pytest.mark.parametrize("gender", ["boy", "girl"])
def test_standard_body_gets_its_own_size(gender):
    from main import CATALOG as C
    std = {"boy": [84, 90, 96, 102, 108, 116], "girl": [78, 84, 90, 96, 102, 110]}
    for p in C:
        if p["gender"] != gender or p["brand"] not in ("Loomline", "Desi Thread") or p.get("missing"):
            continue
        for i, s in enumerate(SIZES):
            # rebuild the standard body for size s from the garment's own construction
            body = {k: p["true"][s][k] for k in p["true"][s] if k != "len"}
            prof = {"fit": "regular", **{k: body[k] - _mid(p, k) for k in body}}
            assert recommend(p, prof)["size"] == s, (p["name"], s)


def _mid(p, k):
    from size_engine import RANGES
    if k not in RANGES[p["kind"]]:
        return 0.5 if k == "shoulder" else 0
    lo, hi = RANGES[p["kind"]][k]["regular"]
    return (lo + hi) / 2 + (p.get("shift") or {}).get(k, 0)


def test_missing_chart_measurement_caps_confidence():
    miss = [p for p in CATALOG if p.get("missing")]
    assert len(miss) == 16
    assert all(recommend(p, BOY if p["gender"] == "boy" else {**BOY, "gender": "girl"})["confidence"] != "High" for p in miss)


def test_missing_profile_measurement_is_reported():
    r = recommend(BY_NAME[("boy", "Oxford Cotton Shirt")], {**BOY, "waist": None, "shoulder": None})
    assert set(r["missing_from_profile"]) == {"waist", "shoulder"} and r["confidence"] == "Medium"


def test_no_measurements_gives_low():
    r = recommend(BY_NAME[("boy", "Oxford Cotton Shirt")], {"fit": "regular"})
    assert r["size"] is None and r["confidence"] == "Low"


def test_tight_costs_more_than_loose():
    assert penalty(2, (4, 8)) > penalty(10, (4, 8))


def test_stretch_allows_negative_ease():
    leg = BY_NAME[("girl", "Leggings")]
    girl = {"gender": "girl", "chest": 90, "waist": 74, "hip": 98, "shoulder": 39, "fit": "regular"}
    assert recommend(leg, girl)["size"] == "M"
    assert not [f for f in fit_flags(leg, "M", girl) if f["status"] == "tight"]


def test_xs_on_xl_body_is_tight():
    tee = BY_NAME[("boy", "Essential Crew Tee")]
    xl = {"chest": 108, "waist": 96, "hip": 110, "shoulder": 49, "fit": "regular"}
    assert any(f["status"] == "tight" for f in fit_flags(tee, "XS", xl))


def test_bigger_body_never_smaller_size():
    for p in CATALOG:
        last = -1
        for d in range(-12, 21, 4):
            prof = {"chest": 96 + d, "waist": 82 + d, "hip": 98 + d, "shoulder": 45 + d / 4, "fit": "regular"}
            i = SIZES.index(recommend(p, prof)["size"])
            assert i >= last, p["name"]
            last = i


def test_api_recommend_report_fitcheck():
    pid = BY_NAME[("boy", "Oxford Cotton Shirt")]["id"]
    body = {"product_id": pid, "profile": BOY}
    assert client.post("/api/size/recommend", json=body).json()["size"] == "M"
    rep = client.post("/api/fit-report", json=body).json()
    assert rep["by_fit"]["relaxed"] == "L" and rep["reasons"]
    fc = client.post("/api/size/fit-check", json={**body, "size": "XS"}).json()
    assert any(f["status"] == "tight" for f in fc["flags"])


def test_products_filter():
    j = client.get("/api/products", params={"gender": "girl", "category": "Dresses"}).json()
    assert j["count"] == 8 and all(p["kind"] == "dress" for p in j["products"])


def _png():
    b = io.BytesIO()
    Image.new("RGB", (300, 400), (230, 225, 218)).save(b, "PNG")
    return b.getvalue()


def test_tryon_job_lifecycle_mock():
    files = {"person": ("p.png", _png(), "image/png"), "garment": ("g.png", _png(), "image/png")}
    r = client.post("/api/tryon", files=files, data={"description": "sky blue shirt"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    for _ in range(50):
        s = client.get(f"/api/tryon/{jid}").json()
        if s["status"] != "running":
            break
        time.sleep(0.1)
    assert s["status"] == "done" and s["is_ai"] is False
    assert s["result_data_url"].startswith("data:image/png;base64,")
    assert client.delete(f"/api/tryon/{jid}").json()["deleted"]
    assert client.get(f"/api/tryon/{jid}").status_code == 404


def test_tryon_rejects_non_images_and_no_consent():
    bad = {"person": ("p.txt", b"hi", "text/plain"), "garment": ("g.png", _png(), "image/png")}
    assert client.post("/api/tryon", files=bad).status_code == 415
    ok = {"person": ("p.png", _png(), "image/png"), "garment": ("g.png", _png(), "image/png")}
    assert client.post("/api/tryon", files=ok, data={"consent": "false"}).status_code == 400


def test_user_test_results_roundtrip():
    f = HERE / "_tmp_user_tests.jsonl"
    f.unlink(missing_ok=True)
    pid = BY_NAME[("boy", "Oxford Cotton Shirt")]["id"]
    row = {"at": "2026-11-22T10:00:00Z", "alias": "AK", "gender": "boy", "chest": 96, "fit": "regular",
           "pid": pid, "rec": "M", "actual": "M", "before": 2, "after": 4, "done": "yes", "secs": 95}
    assert client.post("/api/tests", json=row).status_code == 201
    assert client.post("/api/tests", json={**row, "actual": "XL"}).status_code == 201
    s = client.get("/api/tests").json()["summary"]
    assert s == {"count": 2, "exact_agreement": 0.5, "within_one_size": 0.5}
    assert client.post("/api/tests", json={**row, "before": 9}).status_code == 422
    f.unlink(missing_ok=True)


def test_web_app_served_with_avatar_assets():
    r = client.get("/")
    assert r.status_code == 200 and "FitMate" in r.text
    m = client.get("/assets/mh/mh.json")
    assert m.status_code == 200 and "targets" in m.json()
    assert client.get("/assets/mh/mh.bin").status_code == 200
