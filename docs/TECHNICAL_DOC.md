# FitMate AI: Technical Documentation (Round 3)

Team CodeCodders · PRASUNEX Innovathon 2026

## 1. Summary

FitMate AI separates **how a garment looks** from **how it fits**. A shopper builds an avatar from their own measurements, dresses it from a 100-garment wardrobe (50 boys, 50 girls) in any colour and size, sees on the avatar where a size is tight or loose, and gets a recommended size with an explainable confidence level. AI try-on on a real photo runs through the FastAPI backend and the IDM-VTON model.

## 2. Architecture

```
Browser (frontend/index.html)                     FastAPI backend (backend/)            GPU
┌────────────────────────────────────┐   REST    ┌───────────────────────────────┐ gradio ┌─────────────┐
│ Realistic avatar (three.js + MH)   │ ────────▶ │ /api/tryon  (async jobs)      │ ─────▶ │ IDM-VTON on │
│ Garment drape (from size charts)   │ ◀──────── │ /api/size/* /api/fit-report   │ ◀───── │ HF Space    │
│ Size + Fit Confidence Engine (JS)  │   JSON    │ /api/products  /api/tests     │        └─────────────┘
│ Wardrobe · Report · Studio · Tests │           │ Size Engine (Python, tested)  │
└────────────────────────────────────┘           └───────────────────────────────┘
        │ shared test results (claude.ai page database) or the backend's /api/tests
```

- **Frontend:** one HTML/CSS/JavaScript file, no build step, hash-routed pages: Home, My Avatar, Wardrobe, Photo Try-On, Fit Report, Style Studio, Testing, Saved, Project. The realistic avatar loads `frontend/assets/mh/` (8 MB: body mesh, morph targets, hair and eye meshes, textures), so the app must be served over http; the FastAPI backend serves it at `http://localhost:8000/`.
- **Backend:** Python 3.10+, FastAPI, Pydantic, gradio_client, Pillow.
- **AI model:** IDM-VTON on the public Hugging Face Space `yisol/IDM-VTON` (configurable).

## 3. Realistic 3D avatar (three.js r128 + MakeHuman CC0 data)

- **Base human:** the MakeHuman base mesh (13,380 skin vertices with UVs), real skin textures (light and dark, male and female), high-poly eyes with 6 iris colours, 9 eyebrow and 2 eyelash cards, 10 hair meshes and 4 shoe meshes. All bundled MakeHuman assets are by the MakeHuman team and released under CC0 1.0; community assets with other licences were left out. A Python script (`tools/pack_mh.py`) packs them into one binary file of sparse morph targets plus a JSON index.
- **Morphs (143 targets):** build (gender × muscle × weight macro targets, bilinear), face base (blend of African, Asian and European macro shapes), face shape, jaw and chin, cheeks, nose, lips, eye shape, ears, expression (smile) and a blink unit.
- **Measurement solver:** after the build and face morphs, a 4×4 Newton/Gauss–Seidel solve on the bust, waist, hip and shoulder targets makes the mesh measure the user's centimetres: circumferences are convex-hull perimeters of fixed vertex loops (a tape measure), shoulder width is the distance between the shoulder joints, and the mesh is scaled to the exact height. Tested on 32 bodies × 4 measurements: worst error 0.5 cm.
- **Pose:** joints are the centroids of body-part boundary rings; the arms are rotated down from the A-pose with the skin weights (linear blend) and the forearms straightened, the feet brought slightly together. Hair, eyes, brows, lashes and shoes are proxy meshes fitted to the morphed body (barycentric weights + offsets), so they follow every face and body change.
- **Hair and facial hair:** hair meshes use alpha-tested + blended strand textures tinted to 10 colours; buzz cut, stubble, moustache, goatee, short and full beard are shell overlays on the scalp or beard region with a strand texture and a per-vertex density mask.
- **Templates:** 12 ready-made avatars (6 boys, 6 girls) with pre-rendered thumbnails; 6 face bases, 6 face shapes, 5 jaws, 4 cheeks, 6 noses, 6 lips, 6 eye shapes, 6 eye colours, 9 brows, 12 hair styles, 10 hair colours, 6 facial-hair styles, 10 skin tones, 6 body types, 4 glasses, 4 shoes, 7 backgrounds; "Surprise me" randomises the look.
- **Rendering:** image-based lighting from a room environment (PMREM) plus key, fill and rim lights, soft shadows, ACES tone mapping; blinking via a morph target; face close-up camera for the face, eye and hair editors; downloadable avatar card. Fallbacks: a simpler procedural 3D body if the assets cannot load (for example when opened as a plain file), and a 2D SVG avatar without WebGL.

## 4. Garment engine and catalog

- Each garment has a full measurement table per size (`true`) and the table the brand publishes (`chart`). It is draped on the realistic body from `true`: the body surface is cut by scalar fields (neckline shape, sleeve length along the arm, hem, waistband, trouser length along the leg, open fronts and armholes) with exact edge clipping; torso cloth is pushed out to the convex hull of each 1 cm body slice scaled by the garment's ease at that height (garment ÷ body circumference), so it bridges hollows and hangs straight down from the chest; sleeves and trouser legs are offset from the limb axis by the style's ease (skinny to palazzo); a volume-keeping (Taubin) smoothing relaxes the cloth and keeps it outside the body. Skirts, dresses and long kurtas fall as a cone from the hips, with folds, pleats, tiers and slits.
- Layering: every garment is built outside the body or the garment beneath it (trousers → top → jacket), and skin hidden under clothes is not drawn, so a size that is smaller than the body cannot sink into it. Where the size chart says the garment is tight, the cloth is tinted red and labelled ("Tight at chest · −10 cm"); where it is very loose it gets a blue label and visible folds.
- Fabric textures are generated in the browser: denim twill, knit rib, checks, stripes, polka dots, floral print, puffer quilting, woven bump; details include collars, lapels, hoods, buttons, zips, pockets, waistbands, cuffs, kurta borders and sneakers.
- 28 garment types (tees, shirts, polos, hoodies, kurtas, kurtis, jackets, blazers, Nehru jackets, cardigans, dresses, skirts, jeans, chinos, cargo, joggers, shorts, palazzos, leggings…), patterns (stripes, checks, denim, dots, print, quilt, rib), 3–5 colours each (390 colour variants), sizes XS–XXL.
- Size charts are generated from standard body charts plus the garment's designed ease (slim, oversized, flared, skinny, stretch) and each brand's offset ("runs small" −3 cm, "runs large" +3 cm). 16 garments deliberately publish an incomplete chart to exercise the confidence rules. All brands and charts are sample data.

## 5. Size Engine and Fit Confidence Engine

`ease = garment − body` (cm). Target ranges by garment kind and preferred fit:

| Kind · measurement | Weight | Slim | Regular | Relaxed |
|---|---|---|---|---|
| Tops · chest | 0.5 | 0–4 | 4–8 | 8–14 |
| Tops · waist | 0.3 | 2–8 | 8–14 | 14–22 |
| Tops · shoulder | 0.2 | −1–2 | −1–2 | −1–3 |
| Jackets · chest | 0.5 | 6–10 | 10–14 | 14–20 |
| Jackets · waist | 0.3 | 8–14 | 14–20 | 20–28 |
| Dresses · chest / waist / hip | 0.4 / 0.35 / 0.25 | 0–4 / 0–4 / 2–6 | 3–7 / 3–8 / 4–10 | 6–12 / 7–14 / 8–16 |
| Bottoms · waist / hip | 0.6 / 0.4 | 0–2 / 2–6 | 1–4 / 4–10 | 3–6 / 8–14 |

A garment's designed cut shifts its ranges (an oversized tee is meant to be loose); stretch fabric lowers the bottom of the range. Each size's score is the weighted distance outside the range (tight × 1.5). Lowest score wins; ties go to the size nearest the centre.

Confidence: **Measurement match** Strong ≤ 0.5, Close ≤ 2, else Weak. **Shopping confidence** High = Strong and the chart has every key measurement; Medium = Strong/Close with at least half the measurement weight; else Low. "Why this recommendation?" is generated from the same calculation.

## 6. API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/products` | Catalog; filters `gender`, `category`, `brand`, `max_price`, `colour_family` |
| POST | `/api/size/recommend` | Size, match, confidence |
| POST | `/api/size/fit-check` | Tight / loose spots for a chosen size |
| POST | `/api/fit-report` | Every size, reasons, size per fit preference |
| POST / GET / DELETE | `/api/tryon`, `/api/tryon/{id}` | AI try-on job (202 → poll → image), deletion |
| POST / GET | `/api/tests` | Store user-test results; summary with agreement rates |

## 7. Testing

1. **Automated checks in the browser** (Testing page, 18 checks, including 300 3D garment builds): catalog completeness, a standard body gets its own size (84 cases), proposal example, missing-data confidence cap, monotonic sizing, "runs small" behaviour, tight detection, 600 garment renders, 390 thumbnails, 240 avatar looks, extreme bodies, the realistic body matching the entered measurements (32 bodies, worst error 0.5 cm) and every template, face option, skin tone and extra building. All pass.
2. **pytest** (`backend/tests`, 18 tests): includes a parity test that the Python and browser engines give the same size, match and confidence on 2,100 cases, plus API, try-on job lifecycle (mock), validation, test-results endpoints and the backend serving the app with its avatar assets. All pass.
3. **User tests** (Testing page): each tester builds their avatar, reads the Fit Report and gives the size that really fits them, confidence before/after (1–5), whether they finished unaided and the time taken. Results are saved to the page's shared database (owner and editors) or the backend, and summarised as exact agreement, within-one-size agreement, confidence change, completion rate and median time. Export as CSV.

## 8. Privacy

Consent before any photo upload; overlay preview never uploads the photo; AI try-on deletes photos when the job ends and results after an hour; "Delete my photo" calls the delete endpoint. Avatar, outfits and saved looks stay in the browser. User tests store measurements and an alias only, never photos or names.

## 9. Limitations

- The avatar is realistic but built from templates, not scanned from a selfie. Cloth is shaped geometrically, not physically simulated: it shows width, length and tight or loose areas, not true drape, stretch or fabric weight. Very large or very small bodies push the morph targets beyond their range and look less natural. A low-end phone may render the 3D view slowly.
- The realistic avatar needs the app served over http and loads about 8 MB once; opened as a plain file it falls back to the simpler 3D body.
- AI try-on depends on the public IDM-VTON Space (queues, rate limits, upper-body garments best; non-commercial licence). It was not run against the live Space during development; the job flow is tested in mock mode.
- The hosted page cannot call outside servers, so AI try-on runs with the frontend and backend together.
- Catalog, brands and charts are sample data; measurements are entered by hand.
- Size thresholds are starting values until enough user tests are recorded.

## 10. Future development

1. Supabase accounts, real retailer catalogs, admin dashboard with CSV size-chart import.
2. Cloth physics (drape and stretch simulation) and poses.
3. GPU worker queue and caching; a commercially licensed try-on model.
4. Selfie-to-face fitting of the face templates, and measurements estimated from two photos (research).
5. Retailer Fit API widget and browser extension; tune thresholds from user-test data.
