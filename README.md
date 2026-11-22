# FitMate AI

**Try before you buy. Fit before you spend.**
Team CodeCodders · PRASUNEX Innovathon 2026 · Round 3 prototype

FitMate AI lets a shopper build a realistic 3D avatar (three.js + MakeHuman CC0 body) from templates and their own measurements, dress it from 100 garments (50 boys, 50 girls) in every colour and size XS–XXL, see where a size is tight or loose, get a size recommendation with an explainable confidence level, and try garments on their real photo with AI.

## What's in this repository

```
frontend/index.html      The web app (HTML/CSS/JS, no build step): 3D avatar + garment engine,
                         100-garment catalog, size engine, wardrobe, reports, testing page
frontend/assets/mh/      Realistic avatar data: body mesh, 143 morph targets, hair/eyes/brows/shoes,
                         skin textures (MakeHuman team assets, CC0 1.0)
frontend/assets/presets/ Template thumbnails; frontend/assets/hair/ hair-style thumbnails
frontend/vendor/         three.js r128 + OrbitControls (MIT), used if the CDN is unreachable
tools/pack_mh.py         Builds assets/mh from the MakeHuman data
backend/main.py          FastAPI API: try-on jobs, size recommendation, fit check, fit report,
                         catalog, user-test results
backend/size_engine.py   Size Engine + Fit Confidence Engine (same rules as the frontend)
backend/tryon.py         AI try-on through the IDM-VTON Hugging Face Space (gradio_client)
backend/data/products.json  The 100-garment catalog exported from the frontend (sample data)
backend/tests/           pytest suite (18 tests, incl. 2,100-case browser↔Python parity)
docs/TECHNICAL_DOC.md    Technical documentation, features, limitations, future plans
docs/DEMO_SCRIPT.md      Script for the demonstration video
```

## Run it

**Frontend only** (everything except AI try-on): serve the folder over http, for example `cd frontend && python -m http.server 8080`, and open http://localhost:8080 in Chrome or Edge. The 3D view needs WebGL; three.js loads from jsDelivr, or from `frontend/vendor/` when offline. Opening `index.html` as a plain file works too, but then the browser blocks the avatar assets and you get the simpler 3D body.

**With the backend** (adds AI try-on):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Then open http://localhost:8000/ (the backend serves the app too), go to **Photo Try-On → step 4**, and set the backend URL to `http://localhost:8000`.
API docs: http://localhost:8000/docs

Options (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | none | Hugging Face token: higher GPU quota on the Space |
| `FITMATE_TRYON_SPACE` | `yisol/IDM-VTON` | Another Space with the same API |
| `FITMATE_TRYON_MOCK` | off | `1` = no GPU; returns a composite stamped "MOCK", never shown as AI |
| `FITMATE_CORS` | `*` | Allowed origins, comma separated |
| `FITMATE_TESTS_FILE` | `data/user_tests.jsonl` | Where user-test results are stored |

Run tests: `cd backend && pytest -q`. In the app, the Testing page runs 18 automated checks (including 300 3D builds and the measurement-accuracy check) and records real user tests.

## Deploy

- Frontend: any static host (GitHub Pages, Netlify, Vercel). Upload the whole `frontend/` folder (index.html, assets/, vendor/).
- Backend: Render or Railway (`uvicorn main:app --host 0.0.0.0 --port $PORT`). Set `FITMATE_CORS` to the frontend's URL.

## Honest notes

- The 3D avatar is realistic but built from templates, not from a selfie, and its cloth is shaped geometrically, not physically simulated.
- Avatar data: MakeHuman base mesh, targets, proxies and textures by the MakeHuman team, CC0 1.0 (https://www.makehumancommunity.org). three.js is MIT.
- The overlay preview is a flat image placed on the photo and is always labelled "not AI-generated". Only the backend produces AI try-on images.
- IDM-VTON is licensed for non-commercial use. The public Space can queue or rate-limit; an `HF_TOKEN` helps.
- Sizes are estimates from measurements and brand charts, not guaranteed physical fit.
