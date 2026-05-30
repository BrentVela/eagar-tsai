import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Edit these before each run.
heatmap_path = "CalcFiles/Test3_(z=0)/ET_xy_heatmap_row_0_z=0.csv"
meta_path = "CalcFiles/Test3_(z=0)/ET_v3_OUT_0.csv"
output_path = "CalcFiles/Test11/liquidus_normal_alloy0.csv"
velocity_col = "Velocity_m/s"  # fallback to "v" if missing
def _load_scan_speed(meta_df, preferred_col):
    if preferred_col in meta_df.columns:
        return float(meta_df[preferred_col].iloc[0])
    if "v" in meta_df.columns:
        return float(meta_df["v"].iloc[0])
    raise ValueError("No scan speed column found (Velocity_m/s or v).")


def _load_liquidus_temp(meta_df):
    if "TL" in meta_df.columns:
        return float(meta_df["TL"].iloc[0])
    if "PROP LT (K)" in meta_df.columns:
        return float(meta_df["PROP LT (K)"].iloc[0])
    raise ValueError("No liquidus temperature column found (TL or PROP LT (K)).")


def compute_liquidus_normals(heatmap_csv, meta_csv, out_csv, v_col):
    # Avoid matplotlib config write issues.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    meta = pd.read_csv(meta_csv)
    TL = _load_liquidus_temp(meta)
    v_scan = _load_scan_speed(meta, v_col)

    df = pd.read_csv(heatmap_csv)
    y = df.iloc[:, 0].to_numpy()
    x = df.columns[1:].astype(float).to_numpy()
    T = df.iloc[:, 1:].to_numpy()

    dTdy, dTdx = np.gradient(T, y, x)

    cs = plt.contour(x, y, T, levels=[TL])
    segments = cs.allsegs[0]

    interp_dx = RegularGridInterpolator((y, x), dTdx)
    interp_dy = RegularGridInterpolator((y, x), dTdy)

    rows = []

    for seg in segments:
        if len(seg) == 0:
            continue
        # seg is (N,2) with columns x,y; interpolator expects (y,x)
        pts_yx = np.column_stack([seg[:, 1], seg[:, 0]])
        gx = interp_dx(pts_yx)
        gy = interp_dy(pts_yx)
        mag = np.sqrt(gx**2 + gy**2)
        mag[mag == 0] = np.nan
        # Normal points toward increasing T, i.e. the hot side.
        nx = gx / mag
        ny = gy / mag
        # Apply the convention flip only when projecting onto R_x.
        rx = -nx * v_scan
        for (xv, yv, gv, nxv, nyv, rxv) in zip(seg[:, 0], seg[:, 1], mag, nx, ny, rx):
            rows.append((xv, yv, gv, nxv, nyv, rxv))

    out_df = pd.DataFrame(rows, columns=["x", "y", "G", "nx", "ny", "R_x"])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    return TL, v_scan, out_path


if __name__ == "__main__":
    TL, v_scan, out_path = compute_liquidus_normals(
        heatmap_csv=heatmap_path,
        meta_csv=meta_path,
        out_csv=output_path,
        v_col=velocity_col,
    )
    print(f"Wrote {out_path}")
    print(f"Liquidus TL = {TL} K, v_scan = {v_scan} m/s")
