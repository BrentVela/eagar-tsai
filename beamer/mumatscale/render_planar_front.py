"""Render a phase-aware planar front from a muMatScale x-z center section."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PARTITION_COEFFICIENT = 0.14
SOLID_MIN, SOLID_MAX = 0.50, 0.60
LIQUID_MIN, LIQUID_MAX = 3.00, 7.00
SOLID_PALETTE = np.array(
    [[22.0, 55.0, 84.0], [31.0, 75.0, 112.0], [49.0, 101.0, 148.0]]
)
LIQUID_PALETTE = np.array(
    [
        [247.0, 251.0, 254.0],
        [229.0, 242.0, 250.0],
        [189.0, 220.0, 241.0],
        [125.0, 184.0, 224.0],
    ]
)


def composition_color(
    composition: np.ndarray,
    minimum: float,
    maximum: float,
    palette: np.ndarray,
) -> np.ndarray:
    normalized = np.clip((composition - minimum) / (maximum - minimum), 0.0, 1.0)
    scaled = normalized * (len(palette) - 1)
    lower = np.floor(scaled).astype(np.intp)
    upper = np.minimum(lower + 1, len(palette) - 1)
    weight = (scaled - lower)[..., None]
    return palette[lower] * (1.0 - weight) + palette[upper] * weight


def load_center_plane(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        groups = [name for name in h5 if name.startswith("SubBlock_")]
        if len(groups) != 1:
            raise ValueError("renderer expects the one-subblock planar case")
        group = h5[groups[0]]
        y = group["FracSolid"].shape[1] // 2
        fs = np.flipud(group["FracSolid"][:, y, :].astype(np.float32))
        ce = np.flipud(group["CE"][:, y, :].astype(np.float32))
        return fs, ce


def crop_front(fs: np.ndarray, ce: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Square crop with the mean solid-front height near the image midpoint."""
    row_solid = np.mean(fs > 0.5, axis=1)
    active_rows = np.flatnonzero(row_solid > 0.05)
    if active_rows.size == 0:
        raise ValueError("snapshot contains no planar solid")
    bottom = fs.shape[0]
    solid_height = bottom - int(active_rows.min())
    crop_size = min(max(2 * solid_height, 40), fs.shape[0], fs.shape[1])
    top = bottom - crop_size
    center = fs.shape[1] // 2
    left = center - crop_size // 2
    return fs[top:bottom, left:left + crop_size], ce[top:bottom, left:left + crop_size]


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 25)
    small_font = ImageFont.truetype(font_path, 18)
    x0, width, height = 28, 260, 16
    draw.text((x0, 14), "Cu composition (wt%)", fill=(33, 37, 41, 255), font=font)
    bars = (
        (62, "Solid Cs", SOLID_MIN, SOLID_MAX, SOLID_PALETTE),
        (108, "Liquid Cl", LIQUID_MIN, LIQUID_MAX, LIQUID_PALETTE),
    )
    for y, label, minimum, maximum, palette in bars:
        draw.text((x0, y - 1), label, fill=(33, 37, 41, 255), font=small_font, anchor="ls")
        gx, gw = x0 + 84, width - 84
        for offset in range(gw):
            value = minimum + offset / (gw - 1) * (maximum - minimum)
            color = composition_color(np.array(value), minimum, maximum, palette)
            draw.line((gx + offset, y - height, gx + offset, y), fill=tuple(np.uint8(color)) + (255,))
        draw.rectangle((gx, y - height, gx + gw, y), outline=(33, 37, 41, 210), width=2)
        draw.text((gx, y + 3), f"{minimum:.2f}", fill=(33, 37, 41, 255), font=small_font, anchor="la")
        draw.text((gx + gw, y + 3), f"{maximum:.2f}", fill=(33, 37, 41, 255), font=small_font, anchor="ra")


def render(fs: np.ndarray, ce: np.ndarray, pixels: int = 900) -> Image.Image:
    fs, ce = crop_front(fs, ce)
    solid_fraction = np.clip(fs, 0.0, 1.0)
    liquid_composition = ce / (1.0 - (1.0 - PARTITION_COEFFICIENT) * solid_fraction)
    solid_composition = PARTITION_COEFFICIENT * liquid_composition
    liquid_rgb = composition_color(liquid_composition, LIQUID_MIN, LIQUID_MAX, LIQUID_PALETTE)
    solid_rgb = composition_color(solid_composition, SOLID_MIN, SOLID_MAX, SOLID_PALETTE)
    phase = solid_fraction[..., None]
    rgb = solid_rgb * phase + liquid_rgb * (1.0 - phase)

    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    image = image.resize((pixels, pixels), Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.SMOOTH)
    draw = ImageDraw.Draw(image, "RGBA")
    draw_legend(draw)
    draw.rectangle((0, 0, pixels - 1, pixels - 1), outline=(33, 37, 41, 170), width=3)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="muMatScale per-rank HDF5 snapshot")
    parser.add_argument("output", type=Path, help="output PNG")
    args = parser.parse_args()
    fs, ce = load_center_plane(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(fs, ce).save(args.output)


if __name__ == "__main__":
    main()
