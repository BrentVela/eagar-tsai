import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5))

    specs = [
        ("small $\\rho$", "#4482be", 0.18, 0.0),
        ("medium $\\rho$", "#5c748e", 0.28, 0.15),
        ("large $\\rho$", "#bf4d3b", 4.50, 0.0),
    ]

    y = np.linspace(-1.3, 1.3, 500)

    for ax, (label, color, rho, x_shift) in zip(axes, specs):
        x = x_shift - (y**2) / (2.0 * rho)

        ax.fill_betweenx(y, -2.0, x, color=color, alpha=0.22)
        ax.plot(x, y, color=color, lw=3.0)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 0.2)
        ax.set_ylim(-1.4, 1.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.04, top=0.99, wspace=0.08)
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kgt_tip_radii.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
