#!/usr/bin/env python3
"""Animate the held-out ET prior transforming into its predicted warped prior."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .et_temperature_field import load_et_temperature_field
    from .GPR_cylindrical_warped import WarpParameters, evaluate_warped_et
except ImportError:
    from et_temperature_field import load_et_temperature_field
    from GPR_cylindrical_warped import WarpParameters, evaluate_warped_et


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "beamer/figures/data/BU_ET/0_250W_0.5ms"
OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/warped_cartesian_r3d_matern52_gpr"
    / "heldout_250W_0.5ms_15trainingcases"
)
OUTPUT_MP4 = OUTPUT_DIR / "warped_et_prior_animation.mp4"
OUTPUT_POSTER = OUTPUT_DIR / "warped_et_prior_animation_poster.png"

TARGET = np.asarray(
    [0.7455711091, 0.9746879331, 0.7269659774, 0.8565903741,
     2.8715465973, 0.0475944488],
    dtype=float,
)
IDENTITY = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=float)
PARAMETER_LABELS = (
    "Temperature-rise scale",
    "Rear scale",
    "Front scale",
    "Width scale",
    "Depth scale",
    r"$x$ shift / $\sigma$",
)


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def mirrored_warped_temperature(field, coordinates, parameters, sigma_um):
    symmetric = coordinates.copy()
    symmetric[:, 1] = np.abs(symmetric[:, 1])
    return evaluate_warped_et(
        field,
        symmetric,
        WarpParameters.from_array(parameters),
        sigma_um,
    )


def main():
    metadata = pd.read_csv(CASE_DIR / "metadata.csv").iloc[0]
    field = load_et_temperature_field(CASE_DIR / "ET_temperature.vti")
    sigma_um = float(metadata["beam_diameter_m"]) * 1.0e6 / np.sqrt(2.0)
    liquidus_k = float(metadata["liquidus_temperature_k"])

    x = np.linspace(-300.0, 100.0, 321)
    y = np.linspace(-120.0, 120.0, 193)
    z = np.linspace(-140.0, 0.0, 169)
    xx_top, yy_top = np.meshgrid(x, y)
    xx_side, zz_side = np.meshgrid(x, z)
    top_coordinates = np.column_stack(
        [xx_top.ravel(), yy_top.ravel(), np.zeros(xx_top.size)]
    )
    side_coordinates = np.column_stack(
        [xx_side.ravel(), np.zeros(xx_side.size), zz_side.ravel()]
    )

    original_top = mirrored_warped_temperature(
        field, top_coordinates, IDENTITY, sigma_um
    ).reshape(xx_top.shape)
    original_side = mirrored_warped_temperature(
        field, side_coordinates, IDENTITY, sigma_um
    ).reshape(xx_side.shape)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    extent_top = [x.min(), x.max(), y.min(), y.max()]
    extent_side = [x.min(), x.max(), -z.max(), -z.min()]
    image_kwargs = dict(
        cmap="inferno", vmin=298.0, vmax=float(metadata["peak_temperature"]),
        interpolation="bilinear", aspect="auto"
    )
    fixed_top = axes[0, 0].imshow(original_top, extent=extent_top, origin="lower", **image_kwargs)
    moving_top = axes[0, 1].imshow(original_top, extent=extent_top, origin="lower", **image_kwargs)
    fixed_side = axes[1, 0].imshow(
        original_side, extent=extent_side, origin="upper", **image_kwargs
    )
    moving_side = axes[1, 1].imshow(
        original_side, extent=extent_side, origin="upper", **image_kwargs
    )

    contour_artists = []

    def draw_contours(warped_top, warped_side):
        while contour_artists:
            contour_artists.pop().remove()
        for axis, gx, gy, values in (
            (axes[0, 0], xx_top, yy_top, original_top),
            (axes[0, 1], xx_top, yy_top, warped_top),
            (axes[1, 0], xx_side, -zz_side, original_side),
            (axes[1, 1], xx_side, -zz_side, warped_side),
        ):
            contour = axis.contour(
                gx, gy, values, levels=[liquidus_k], colors="#20e6e6", linewidths=2.0
            )
            contour_artists.append(contour)

    for col, title in enumerate(("Original ET", "Progressively warped ET")):
        axes[0, col].set_title(title, fontsize=17, fontweight="bold")
        axes[1, col].set_xlabel("Scan direction, x (µm)", fontsize=13)
    axes[0, 0].set_ylabel("Width, y (µm)", fontsize=13)
    axes[1, 0].set_ylabel("Depth (µm)", fontsize=13)
    axes[0, 0].set_xlabel("Scan direction, x (µm)", fontsize=13)
    axes[0, 1].set_xlabel("Scan direction, x (µm)", fontsize=13)
    # Put the material surface (depth = 0) at the top and increase depth
    # downward, matching the ET heatmap and the physical build convention.
    axes[1, 0].set_ylim(140.0, 0.0)
    axes[1, 1].set_ylim(140.0, 0.0)
    for axis in axes.ravel():
        axis.tick_params(labelsize=10)

    progress_text = axes[1, 1].text(
        0.02, 0.96, "Warp progress: 0%", transform=axes[1, 1].transAxes,
        va="top", color="white", fontsize=13, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4),
    )
    parameter_text = axes[0, 1].text(
        0.02, 0.97, "", transform=axes[0, 1].transAxes, va="top",
        color="white", fontsize=9.5,
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4),
    )
    colorbar = fig.colorbar(fixed_top, ax=axes, shrink=0.90, pad=0.015)
    colorbar.set_label("Temperature (K)", fontsize=13)
    draw_contours(original_top, original_side)

    total_frames = 72

    def update(frame_index):
        raw = frame_index / (total_frames - 1)
        fraction = smoothstep(raw)
        parameters = IDENTITY + fraction * (TARGET - IDENTITY)
        warped_top = mirrored_warped_temperature(
            field, top_coordinates, parameters, sigma_um
        ).reshape(xx_top.shape)
        warped_side = mirrored_warped_temperature(
            field, side_coordinates, parameters, sigma_um
        ).reshape(xx_side.shape)
        moving_top.set_data(warped_top)
        moving_side.set_data(warped_side)
        draw_contours(warped_top, warped_side)
        progress_text.set_text(f"Warp progress: {fraction * 100:3.0f}%")
        parameter_text.set_text(
            "\n".join(
                f"{label}: {value:.3f}"
                for label, value in zip(PARAMETER_LABELS, parameters)
            )
        )
        return [moving_top, moving_side, progress_text, parameter_text]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / 18, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=18, codec="libx264", bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(OUTPUT_MP4, writer=writer, dpi=150)
    update(total_frames - 1)
    fig.savefig(OUTPUT_POSTER, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_MP4}")
    print(f"Wrote {OUTPUT_POSTER}")


if __name__ == "__main__":
    main()
