from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


OUT_DIR = Path(__file__).resolve().parent / "figures" / "ca_modes"
PANEL_W = 900
PANEL_H = 560


PALETTE = np.array(
    [
        [65, 126, 181],
        [92, 153, 141],
        [191, 77, 59],
        [222, 169, 80],
        [126, 117, 177],
        [103, 148, 96],
        [183, 107, 139],
        [94, 135, 158],
    ],
    dtype=np.float32,
)


def neighbor_count(solid: np.ndarray) -> np.ndarray:
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    count = np.zeros_like(solid, dtype=np.uint8)
    for yoff in range(3):
        for xoff in range(3):
            if yoff == 1 and xoff == 1:
                continue
            count += padded[yoff : yoff + solid.shape[0], xoff : xoff + solid.shape[1]]
    return count


def interface_mask(solid: np.ndarray) -> np.ndarray:
    return solid & (neighbor_count(~solid) > 0)


def add_repo_deps() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for dirname in [".video_deps", ".deps"]:
        deps = repo_root / dirname
        if deps.exists():
            sys.path.insert(0, str(deps))


def render_state(
    solid: np.ndarray,
    grain: np.ndarray,
    *,
    liquid_tint: tuple[int, int, int] = (232, 238, 240),
) -> Image.Image:
    h, w = solid.shape
    y, x = np.mgrid[0:h, 0:w]
    thermal = (0.65 * (1.0 - y / h) + 0.35 * (x / w))[..., None]
    img = np.zeros((h, w, 3), dtype=np.float32)

    liquid = np.array(liquid_tint, dtype=np.float32)
    img[:] = liquid * (0.84 + 0.16 * thermal)

    colors = PALETTE[np.maximum(grain, 0) % len(PALETTE)]
    phase = (
        np.sin(0.075 * x + 0.022 * y)
        + np.sin(0.041 * x - 0.061 * y)
        + 0.6 * np.sin(0.026 * x + 0.088 * y)
    )
    shade = (0.78 + 0.18 * thermal + 0.045 * phase[..., None])
    img[solid] = colors[solid] * shade[solid] + 12

    edge = interface_mask(solid)
    img[edge] = np.array([250, 250, 246], dtype=np.float32) * 0.24 + img[edge] * 0.76

    boundary = solid & (neighbor_count(solid) < 8)
    img[boundary] = img[boundary] * 0.72 + np.array([24, 28, 32], dtype=np.float32) * 0.28

    im = Image.fromarray(np.uint8(np.clip(img, 0, 255))).resize(
        (PANEL_W, PANEL_H), Image.Resampling.BICUBIC
    )
    im = im.filter(ImageFilter.SMOOTH)
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=(33, 37, 41, 170), width=3)
    return im


def columnar() -> Image.Image:
    rng = np.random.default_rng(23)
    h, w = 360, 560
    grain = np.full((h, w), -1, dtype=np.int16)
    seed_x = np.array([35, 86, 138, 198, 262, 326, 388, 453, 520])
    drift = rng.normal(0.0, 0.30, len(seed_x))
    fronts = seed_x.astype(float)

    for y in range(h - 1, 45, -1):
        fronts += drift + rng.normal(0.0, 0.82, len(fronts))
        fronts = np.clip(fronts, 0, w - 1)
        xs = np.arange(w)
        dist = np.abs(xs[:, None] - fronts[None, :])
        owner = np.argmin(dist + rng.normal(0.0, 1.8, dist.shape), axis=1)
        waviness = 7 * np.sin(0.055 * xs + 0.11 * y) + rng.normal(0.0, 2.0, w)
        local_front = 48 + 0.10 * np.abs(xs - w / 2) + waviness
        active = y > local_front
        grain[y, active] = owner[active]

    solid = grain >= 0
    return render_state(solid, grain, liquid_tint=(229, 236, 238))


def add_arrow(im: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(im, "RGBA")
    x = PANEL_W - 82
    draw.line([x, PANEL_H - 62, x, PANEL_H - 210], fill=(255, 255, 255, 230), width=8)
    draw.polygon(
        [
            (x, PANEL_H - 210),
            (x - 23, PANEL_H - 166),
            (x + 23, PANEL_H - 166),
        ],
        fill=(255, 255, 255, 230),
    )
    return im


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = {
        "ca_columnar.png": add_arrow(columnar()),
    }
    for name, panel in panels.items():
        panel.save(OUT_DIR / name)

    # Physical solutal CA panels: equiaxed dendritic, columnar dendritic
    # (PDAS/SDAS), and stable planar growth.
    import ca_dendrite_solidification as cads

    cads.render_outputs(cads.simulate(), OUT_DIR)
    cads.render_outputs(cads.simulate(cads.equiaxed_params()), OUT_DIR,
                        basename="ca_dendritic", arrow=False, annotate=False,
                        mush=True)
    cads.render_outputs(cads.simulate(cads.planar_params()), OUT_DIR,
                        basename="ca_planar", annotate=False)


if __name__ == "__main__":
    main()
