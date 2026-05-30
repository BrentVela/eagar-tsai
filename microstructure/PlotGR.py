import os
import sys

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GR_Map import _load_alloy_from_excel, _tc_python_cache_dir, get_CET_grid, get_CET_lines
from tc_python import CompositionUnit, TCPython


LIQUIDUS_CSV = "CalcFiles/Test16/125_0.2/liquidus_GR_alloy0_125_0.2.csv"
OUTPUT_PATH = "CalcFiles/Test16/125_0.2/projected_alloy0_125_0.2_deepest2.png"

EXCEL_PATH = "et_custom_input_data.xlsx"
ROW_INDEX = 6
ELEMENT_COLS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
MAP_SOURCE = "excel_alloy"
MAP_CONFIGS = {
    "excel_alloy": {
        "thermo_db": "TCHEA8",
        "kinetic_db": "MOBHEA3",
        "primary_phase": "BCC_B2",
        "interfacial_energy": 0.5,
        "composition_unit": CompositionUnit.MOLE_FRACTION,
    },
    "cu70ni30": {
        "thermo_db": "TCHEA7",
        "kinetic_db": "MOBHEA3",
        "primary_phase": "FCC_L12",
        "interfacial_energy": 0.5,
        "composition_unit": CompositionUnit.MASS_PERCENT,
        "elements": ["Cu", "Ni"],
        "solutes": {"Ni": 30.0},
    },
}


def _load_projected_liquidus(csv_path):
    df = pd.read_csv(csv_path)
    required = {"x", "y", "z", "G", "R"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "z", "G", "R"])
    df = df[(df["G"] > 0.0) & (df["R"] > 0.0)].copy()
    if df.empty:
        raise ValueError(f"No positive-G, positive-R liquidus points found in {csv_path}.")

    deepest = df.loc[df["z"].idxmin()]
    return df, deepest


def _points_past_deepest_x(liquidus, deepest):
    deepest_x = float(deepest["x"])
    projected = liquidus[liquidus["x"] >= deepest_x].copy()
    if projected.empty:
        raise ValueError(f"No liquidus points found at or past x = {deepest_x:.3f} um.")
    if not (projected["x"] >= deepest_x).all():
        raise ValueError(f"Projection selection included points before x = {deepest_x:.3f} um.")
    return projected


def _calculate_cet_grid():
    config = MAP_CONFIGS[MAP_SOURCE]
    if MAP_SOURCE == "excel_alloy":
        elements, solutes = _load_alloy_from_excel(EXCEL_PATH, ROW_INDEX, ELEMENT_COLS)
        element_names = list(elements.keys())
    else:
        solutes = config["solutes"]
        element_names = config["elements"]

    with TCPython() as session:
        session.set_cache_folder(_tc_python_cache_dir())
        system = session.select_thermodynamic_and_kinetic_databases_with_elements(
            config["thermo_db"],
            config["kinetic_db"],
            element_names,
        ).get_system()

        r_grid, g_grid, equiaxed_fraction, _, _, _ = get_CET_grid(
            system,
            solutes,
            primary_phase=config["primary_phase"],
            interfacial_energy=config["interfacial_energy"],
            nb_nucleations_site=4.0e11,
            nucleation_undercooling=4.0,
            equiaxed_exponent=3.13,
            disable_output=True,
            composition_unit=config["composition_unit"],
        )

        line_r, cet_lines = get_CET_lines(
            system,
            (3, 10),
            solutes,
            primary_phase=config["primary_phase"],
            interfacial_energy=config["interfacial_energy"],
            nb_nucleations_site=4.0e11,
            nucleation_undercooling=4.0,
            equiaxed_exponent=3.13,
            disable_output=True,
            composition_unit=config["composition_unit"],
        )

    return (
        np.asarray(r_grid),
        np.asarray(g_grid),
        np.asarray(equiaxed_fraction, dtype=float),
        np.asarray(line_r, dtype=float),
        np.asarray(cet_lines[2], dtype=float),
    )


def _planar_front_by_rate(line_r, planar_line):
    line_df = pd.DataFrame({"R": line_r, "G_planar": planar_line})
    line_df = line_df.replace([np.inf, -np.inf], np.nan).dropna()
    line_df = line_df.groupby("R", as_index=False)["G_planar"].median().sort_values("R")
    return line_df["R"].to_numpy(dtype=float), line_df["G_planar"].to_numpy(dtype=float)


def _classify_from_gr_map(df, r_grid, g_grid, equiaxed_fraction, line_r, planar_line):
    log_r_values = np.log10(df["R"].to_numpy(dtype=float))
    log_g_values = np.log10(df["G"].to_numpy(dtype=float))

    log_r_grid = np.log10(r_grid)
    log_g_grid = np.log10(g_grid)
    grid_points = np.column_stack([log_r_grid, log_g_grid])

    query_points = np.column_stack([log_r_values, log_g_values])
    chunk_size = 2000
    nearest = np.empty(len(query_points), dtype=int)

    for start in range(0, len(query_points), chunk_size):
        stop = start + chunk_size
        deltas = query_points[start:stop, None, :] - grid_points[None, :, :]
        nearest[start:stop] = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)

    classified = equiaxed_fraction[nearest]
    classified = np.where(np.isnan(classified), -1.0, classified)

    planar_r, planar_g = _planar_front_by_rate(line_r, planar_line)
    planar_log_g = np.interp(
        log_r_values,
        np.log10(planar_r),
        np.log10(planar_g),
        left=np.log10(planar_g[0]),
        right=np.log10(planar_g[-1]),
    )
    classified[log_g_values >= planar_log_g] = -1.0
    return classified


def plot_projected_liquidus():
    liquidus, deepest = _load_projected_liquidus(LIQUIDUS_CSV)
    projected = _points_past_deepest_x(liquidus, deepest)

    r_grid, g_grid, equiaxed_fraction, line_r, planar_line = _calculate_cet_grid()
    projected["equiaxed_fraction"] = _classify_from_gr_map(
        projected,
        r_grid,
        g_grid,
        equiaxed_fraction,
        line_r,
        planar_line,
    )

    norm = colors.Normalize(vmin=-1.1, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic_r)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    scatter = ax.scatter(
        projected["y"],
        projected["z"],
        c=projected["equiaxed_fraction"],
        cmap=cm.seismic_r,
        norm=norm,
        s=5,
        linewidths=0,
        alpha=0.65,
    )

    ax.scatter(
        [deepest["y"]],
        [deepest["z"]],
        s=60,
        c="none",
        edgecolors="black",
        linewidths=1.2,
        label=f"Deepest point, x={deepest['x']:.1f} um",
        zorder=4,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected y (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"Liquidus projected from x >= {deepest['x']:.1f} um")
    ax.legend(loc="lower right")
    fig.colorbar(scatter, ax=ax, label="Equiaxed Fraction")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    planar_count = int((projected["equiaxed_fraction"] == -1.0).sum())
    print(f"Wrote {OUTPUT_PATH}")
    print(
        "Projected {count} liquidus points at or past x = {x:.3f} um, from x = {xmin:.3f} to {xmax:.3f} um. "
        "Deepest point: y = {y:.3f} um, z = {z:.3f} um.".format(
            count=len(projected),
            xmin=projected["x"].min(),
            xmax=projected["x"].max(),
            x=deepest["x"],
            y=deepest["y"],
            z=deepest["z"],
        )
    )
    print(f"Planar-classified points: {planar_count} / {len(projected)}")


if __name__ == "__main__":
    try:
        plot_projected_liquidus()
    except Exception as exc:
        print(f"PlotGR failed: {exc}", file=sys.stderr)
        raise
