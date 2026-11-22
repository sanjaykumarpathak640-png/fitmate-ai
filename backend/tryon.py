"""AI virtual try-on through a Hugging Face Space (IDM-VTON by default).

Set FITMATE_TRYON_SPACE to use another Space with the same API, and HF_TOKEN
for a higher GPU quota. Set FITMATE_TRYON_MOCK=1 to run without a GPU: the
result is then a labelled composite, never presented as AI output.

Licence: IDM-VTON is released for non-commercial use (CC BY-NC-SA 4.0).
Fine for this hackathon prototype; switch to a commercially licensed model
or API before any commercial launch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

SPACE = os.getenv("FITMATE_TRYON_SPACE", "yisol/IDM-VTON")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MOCK = os.getenv("FITMATE_TRYON_MOCK") == "1"


def run_tryon(person: Path, garment: Path, description: str, workdir: Path,
              status: Callable[[str], None] = lambda m: None) -> tuple[Path, bool]:
    """Return (path to the result image, is_ai). Raises on failure."""
    if MOCK:
        status("Mock mode (no GPU)")
        return _mock(person, garment, workdir), False

    from gradio_client import Client, handle_file  # imported lazily: optional in mock mode

    status(f"Connecting to {SPACE}")
    client = Client(SPACE, token=HF_TOKEN, verbose=False, download_files=str(workdir))
    status("Generating on GPU (can queue when the Space is busy)")
    # API of the public IDM-VTON Space. If the Space changes, check client.view_api().
    result = client.predict(
        dict={"background": handle_file(str(person)), "layers": [], "composite": None},
        garm_img=handle_file(str(garment)),
        garment_des=description or "garment",
        is_checked=True,        # auto-mask the body area
        is_checked_crop=False,  # set True for photos that are not 3:4
        denoise_steps=30,
        seed=42,
        api_name="/tryon",
    )
    out = result[0] if isinstance(result, (list, tuple)) else result
    return Path(out), True


def _mock(person: Path, garment: Path, workdir: Path) -> Path:
    """Paste the garment over the torso and stamp it MOCK. For offline tests and demos only."""
    from PIL import Image, ImageDraw

    p = Image.open(person).convert("RGB").resize((768, 1024))
    g = Image.open(garment).convert("RGBA")
    g.thumbnail((430, 520))
    # treat near-white pixels as transparent so a flat product shot overlays cleanly
    px = g.load()
    for y in range(g.height):
        for x in range(g.width):
            r, gg, b, a = px[x, y]
            if r > 245 and gg > 245 and b > 245:
                px[x, y] = (r, gg, b, 0)
    p.paste(g, ((768 - g.width) // 2, 150), g)
    d = ImageDraw.Draw(p)
    d.rectangle((0, 0, 768, 44), fill=(28, 26, 36))
    d.text((16, 14), "MOCK RESULT - not AI-generated (FITMATE_TRYON_MOCK=1)", fill=(246, 242, 236))
    out = workdir / "result.png"
    p.save(out)
    return out
