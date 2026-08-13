from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import pyvista as pv


DEFAULT_DATA_DIR = Path("CalcFiles/Test16/125_0.2")
DEFAULT_VTI = DEFAULT_DATA_DIR / "ET_3D_temperature_alloy0_125_0.2.vti"
DEFAULT_META = DEFAULT_DATA_DIR / "ET_0000.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "liquidus_GR_alloy0_125_0.2.csv"
TEMP_NAME = "Temperature_K"
DEFAULT_SCAN_DIRECTION_X_SIGN = -1


def _scan_direction_x_sign(grid):
    """Read the VTI scan direction, defaulting to native Eagar--Tsai."""
    field_name = "scan_direction_x_sign"
    if field_name not in grid.field_data:
        return DEFAULT_SCAN_DIRECTION_X_SIGN

    sign = int(np.asarray(grid.field_data[field_name]).flat[0])
    if sign not in (-1, 1):
        raise ValueError(
            f"VTI field data '{field_name}' must be -1 or +1, got {sign}."
        )
    return sign


def compute_gr_from_vti(
    vti_path,
    meta_path,
    output_path,
):
    vti_path = Path(vti_path)
    meta_path = Path(meta_path)
    output_path = Path(output_path)

    meta = pd.read_csv(meta_path)
    tl = float(meta["liquidus_temperature_k"].iloc[0])
    v_scan = float(meta["velocity_m_s"].iloc[0])

    grid = pv.read(vti_path)
    scan_direction_x_sign = _scan_direction_x_sign(grid)
    derived = grid.compute_derivative(scalars=TEMP_NAME, gradient=True, preference="point")
    gradient = np.asarray(derived.point_data["gradient"])

    # VTI coordinates are in microns, so the raw derivative is K/um.
    grad_mag_k_per_um = np.linalg.norm(gradient, axis=1)
    if np.any(grad_mag_k_per_um == 0.0):
        raise ValueError("Zero temperature gradient encountered.")

    derived.point_data["G"] = grad_mag_k_per_um * 1.0e6

    normal_x = gradient[:, 0] / grad_mag_k_per_um
    derived.point_data["R"] = scan_direction_x_sign * normal_x * v_scan

    liquidus = derived.contour(isosurfaces=[tl], scalars=TEMP_NAME)
    if liquidus.n_points == 0:
        raise ValueError(f"No liquidus contour was found at {tl} K.")

    points = np.asarray(liquidus.points)
    out_df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "temp": np.asarray(liquidus.point_data[TEMP_NAME]),
            "G": np.asarray(liquidus.point_data["G"]),
            "R": np.asarray(liquidus.point_data["R"]),
        }
    )
    out_df = out_df[out_df["R"] > 0.0].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    return output_path, tl, v_scan, out_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract liquidus-boundary x, y, z, temp, G, and R from a 3D VTI temperature field."
    )
    parser.add_argument("--vti", dest="vti_path", default=DEFAULT_VTI, help="Input .vti temperature field.")
    parser.add_argument("--meta", dest="meta_path", default=DEFAULT_META, help="CSV containing liquidus temperature and scan speed.")
    parser.add_argument("--output", dest="output_path", default=DEFAULT_OUTPUT, help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    output_path, tl, v_scan, out_df = compute_gr_from_vti(**vars(parse_args()))
    print(f"Wrote {output_path}")
    print(f"Liquidus TL = {tl:.6g} K, scan speed = {v_scan:.6g} m/s, temperature array = {TEMP_NAME}")
    print(
        f"Rows = {len(out_df)}, G = [{out_df['G'].min():.6g}, {out_df['G'].max():.6g}] K/m, "
        f"R = [{out_df['R'].min():.6g}, {out_df['R'].max():.6g}] m/s"
    )
