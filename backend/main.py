"""FitMate AI backend (FastAPI).

Run:  uvicorn main:app --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from size_engine import SIZES, all_fits, fit_flags, recommend
from tryon import MOCK, SPACE, run_tryon

ROOT = Path(__file__).parent
CATALOG = json.loads((ROOT / "data" / "products.json").read_text(encoding="utf-8"))["products"]
PRODUCTS = {p["id"]: p for p in CATALOG}
WORK = Path(tempfile.gettempdir()) / "fitmate-jobs"
MAX_BYTES = 15 * 1024 * 1024
RESULT_TTL_S = 3600  # results are purged after an hour
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="FitMate AI API", version="0.3.0",
              description="Virtual try-on jobs, measurement-based size recommendations and the sample catalog.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("FITMATE_CORS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# ---------- catalog and sizing ----------
class Profile(BaseModel):
    gender: Literal["boy", "girl"] | None = None
    height: float | None = None
    chest: float | None = None
    waist: float | None = None
    hip: float | None = None
    shoulder: float | None = None
    fit: Literal["slim", "regular", "relaxed"] = "regular"


class SizeRequest(BaseModel):
    product_id: str = Field(examples=["p1"])
    profile: Profile


@app.get("/api/health")
def health():
    return {"ok": True, "tryon_space": SPACE, "mock": MOCK, "jobs": len(JOBS)}


@app.get("/api/products")
def products(gender: Literal["boy", "girl"] | None = None, category: str | None = None, brand: str | None = None,
             max_price: int | None = None, colour_family: str | None = None):
    out = [p for p in CATALOG
           if (not gender or p["gender"] == gender) and (not category or p["label"] == category or p["type"] == category)
           and (not brand or p["brand"] == brand)
           and (max_price is None or p["price"] <= max_price)
           and (not colour_family or any(c["fam"] == colour_family for c in p["colors"]))]
    return {"count": len(out), "products": out}


def _product(pid: str) -> dict:
    if pid not in PRODUCTS:
        raise HTTPException(404, f"Unknown product {pid}")
    return PRODUCTS[pid]


@app.post("/api/size/recommend")
def size_recommend(req: SizeRequest):
    p = _product(req.product_id)
    r = recommend(p, req.profile.model_dump())
    return {k: r[k] for k in ("product_id", "fit", "size", "match", "confidence", "completeness")}


class FitCheck(SizeRequest):
    size: Literal["XS", "S", "M", "L", "XL", "XXL"]


@app.post("/api/size/fit-check")
def size_fit_check(req: FitCheck):
    """Where a chosen size is tight or loose on this body: the same markers the avatar shows."""
    return {"product_id": req.product_id, "size": req.size,
            "flags": fit_flags(_product(req.product_id), req.size, req.profile.model_dump())}


@app.post("/api/fit-report")
def fit_report(req: SizeRequest):
    p = _product(req.product_id)
    prof = req.profile.model_dump()
    r = recommend(p, prof)
    return {**r, "by_fit": all_fits(p, prof),
            "disclaimer": "Estimate from body measurements and the brand size chart, not a guaranteed physical fit."}


# ---------- try-on jobs ----------
def _purge():
    now = time.time()
    with LOCK:
        old = [j for j, v in JOBS.items() if now - v["created"] > RESULT_TTL_S]
        for j in old:
            JOBS.pop(j, None)
            shutil.rmtree(WORK / j, ignore_errors=True)


def _set(job_id: str, **kw):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


def _worker(job_id: str, person: Path, garment: Path, description: str):
    d = WORK / job_id
    try:
        result, is_ai = run_tryon(person, garment, description, d,
                                  status=lambda m: _set(job_id, message=m))
        mime = mimetypes.guess_type(str(result))[0] or "image/png"
        data = base64.b64encode(Path(result).read_bytes()).decode()
        _set(job_id, status="done", is_ai=is_ai, message="Done",
             result_data_url=f"data:{mime};base64,{data}")
    except Exception as e:  # report every model failure to the client
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}"[:400])
    finally:
        # privacy: input photos and model files are deleted as soon as the job ends
        shutil.rmtree(d, ignore_errors=True)


async def _save(up: UploadFile, path: Path):
    if up.content_type not in ALLOWED:
        raise HTTPException(415, f"{up.filename}: use JPG, PNG or WebP")
    data = await up.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{up.filename}: over 15 MB")
    path.write_bytes(data)


@app.post("/api/tryon", status_code=202)
async def create_tryon(person: UploadFile = File(...), garment: UploadFile = File(...),
                       description: str = Form("garment"), category: str = Form("upper_body"),
                       consent: bool = Form(True)):
    if not consent:
        raise HTTPException(400, "Consent is required before a photo is processed.")
    _purge()
    job_id = uuid.uuid4().hex
    d = WORK / job_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        await _save(person, d / "person.png")
        await _save(garment, d / "garment.png")
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    with LOCK:
        JOBS[job_id] = {"status": "running", "message": "Queued", "created": time.time(), "category": category}
    threading.Thread(target=_worker, args=(job_id, d / "person.png", d / "garment.png", description[:120]),
                     daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/tryon/{job_id}")
def get_tryon(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown or expired job")
        return {k: v for k, v in job.items() if k != "created"}


@app.delete("/api/tryon/{job_id}")
def delete_tryon(job_id: str):
    with LOCK:
        JOBS.pop(job_id, None)
    shutil.rmtree(WORK / job_id, ignore_errors=True)
    return {"deleted": True}


# ---------- user test results (real testing) ----------
TESTS_FILE = Path(os.getenv("FITMATE_TESTS_FILE", ROOT / "data" / "user_tests.jsonl"))


class TestResult(BaseModel):
    at: str
    alias: str = Field("", max_length=30)
    gender: Literal["boy", "girl"]
    height: float | None = None
    chest: float | None = None
    waist: float | None = None
    hip: float | None = None
    shoulder: float | None = None
    fit: Literal["slim", "regular", "relaxed"] = "regular"
    pid: str
    productName: str = ""
    rec: str | None = None
    conf: str | None = None
    actual: Literal["XS", "S", "M", "L", "XL", "XXL"]
    before: int | None = Field(None, ge=1, le=5)
    after: int | None = Field(None, ge=1, le=5)
    done: Literal["yes", "no"] | None = None
    secs: int | None = Field(None, ge=0, le=36000)
    note: str = Field("", max_length=400)


def _load_tests() -> list[dict]:
    if not TESTS_FILE.exists():
        return []
    return [json.loads(line) for line in TESTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.post("/api/tests", status_code=201)
def add_test(t: TestResult):
    _product(t.pid)
    TESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, TESTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(t.model_dump(), ensure_ascii=False) + "\n")
    return {"saved": True}


@app.get("/api/tests")
def list_tests():
    rows = _load_tests()
    with_actual = [r for r in rows if r.get("rec") and r.get("actual")]
    idx = {s: i for i, s in enumerate(SIZES)}
    exact = sum(r["rec"] == r["actual"] for r in with_actual)
    within = sum(abs(idx[r["rec"]] - idx[r["actual"]]) <= 1 for r in with_actual)
    n = len(with_actual)
    return {"tests": rows, "summary": {"count": len(rows),
            "exact_agreement": round(exact / n, 3) if n else None,
            "within_one_size": round(within / n, 3) if n else None}}


# Serve the web app from the same server, so the realistic avatar assets load over http:
# http://localhost:8000/ opens the app (API routes above take precedence).
_WEB = Path(os.getenv("FITMATE_WEB", Path(__file__).resolve().parent.parent / "frontend"))
if _WEB.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")
