#!/usr/bin/env python3
"""Compare ParaView and ET temperature slices across a YZ projection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


TEMPERATURE_CSV = Path("CalcFiles/Test18/ET_3D_temperature_alloy0.csv")
PARAVIEW_CSV = Path("CalcFiles/Test18/temp3D_fine0.15.csv")
METADATA_CSV = Path("CalcFiles/Test18/ET_v3_OUT_0_alloy0.csv")
OUTPUT_PNG = Path("CalcFiles/Test18/ET_TC_temp_yz_0um_ETmax.png")
ET_X_UM = 0.0
PARAVIEW_X_UM = 0.0
USE_TCAM_SLICE_MAX = False
#TEMPERATURE_VMAX_K = 6000 #Custom max temperature


def load_laser_center_slice():
    parts = []
    for chunk in pd.read_csv(
        TEMPERATURE_CSV,
        usecols=["x", "y", "z", "T"],
        chunksize=500_000,
    ):
        parts.append(chunk.loc[chunk["x"] == ET_X_UM, ["y", "z", "T"]])

    plane = pd.concat(parts, ignore_index=True)
    temperature = plane.pivot(index="z", columns="y", values="T").sort_index()
    y = temperature.columns.to_numpy()
    z = temperature.index.to_numpy()
    values = temperature.to_numpy()

    return y, z, values


def load_paraview_slice():
    points = pd.read_csv(
        PARAVIEW_CSV,
        usecols=["Points_0", "Points_1", "Points_2", "temperature"],
    )
    points["x"] = points["Points_0"] * 1.0e6
    points["y"] = points["Points_1"] * 1.0e6
    points["z"] = points["Points_2"] * 1.0e6
    return points.groupby(
        ["x", "y", "z"],
        as_index=False,
    )["temperature"].mean()


def interpolate_paraview_slice(points, liquidus):
    y = np.linspace(points["y"].min(), points["y"].max(), 90)
    z = np.linspace(points["z"].min(), points["z"].max(), 140)
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, -PARAVIEW_X_UM)
    temperature = griddata(
        points=(points["x"], points["y"], points["z"]),
        values=points["temperature"],
        xi=(xx, yy, zz),
        method="linear",
    )
    return -yy, zz, np.ma.masked_invalid(
        np.ma.masked_less(temperature, liquidus)
    )


def main():
    liquidus = pd.read_csv(METADATA_CSV, usecols=["T_liquidus"]).iloc[0, 0]
    y, z, temperature = load_laser_center_slice()
    paraview = load_paraview_slice()
    paraview_y, paraview_z, paraview_temperature = interpolate_paraview_slice(
        paraview,
        liquidus,
    )
    yy, zz = np.meshgrid(y, z)
    molten = temperature >= liquidus
    if USE_TCAM_SLICE_MAX:
        vmax = paraview_temperature.max()
    else:
        vmax = temperature[molten].max()
        #vmax = TEMPERATURE_VMAX_K #Custom max temperature

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    paraview_molten = ~np.ma.getmaskarray(paraview_temperature)
    paraview_scatter = axis.scatter(
        paraview_y[paraview_molten],
        paraview_z[paraview_molten],
        c=paraview_temperature[paraview_molten],
        cmap="inferno",
        vmin=liquidus,
        vmax=vmax,
        s=10,
        marker="o",
        linewidths=0,
    )
    image = axis.scatter(
        yy[molten],
        zz[molten],
        c=temperature[molten],
        cmap="inferno",
        vmin=liquidus,
        vmax=vmax,
        s=10,
        marker="o",
        linewidths=0,
    )
    paraview_scatter.set_clip_path(
        Rectangle((-100, -100), 100, 100, transform=axis.transData)
    )
    image.set_clip_path(
        Rectangle((0, -100), 100, 100, transform=axis.transData)
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Temperature (K)")

    axis.set_xlabel("y (um)")
    axis.set_ylabel("z (um)")
    axis.set_title(
        f"TCAM x = {PARAVIEW_X_UM:g} um | ET x = {ET_X_UM:g} um"
    )
    axis.set_xlim(-100, 100)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(
        f"Interpolated {len(paraview)} ParaView volume points onto "
        f"x = {PARAVIEW_X_UM:g} um."
    )
    print(f"Saved ParaView and ET temperature comparison to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
