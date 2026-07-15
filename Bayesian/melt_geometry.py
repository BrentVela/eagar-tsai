#!/usr/bin/env python3
"""Measure ET, TCAM, and corrected melt-pool geometry from GPR/ParaView CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_LIQUIDUS_K = 3454.846
METER_SCALE_THRESHOLD = 1.0e-2

FIELDS = [
    {
        "name": "ET",
        "temperature": "T_ET",
        "coord_options": [("x", "y", "z"), ("Points_0", "Points_1", "Points_2"), ("x_TCAM", "y_TCAM", "z_TCAM")],
    },
    {
        "name": "TCAM",
        "temperature": "T_TCAM",
        "coord_options": [("x_TCAM", "y_TCAM", "z_TCAM"), ("x", "y", "z"), ("Points_0", "Points_1", "Points_2")],
    },
    {
        "name": "Corrected",
        "temperature": "T_corrected",
        "coord_options": [("x", "y", "z"), ("Points_0", "Points_1", "Points_2"), ("x_TCAM", "y_TCAM", "z_TCAM")],
    },
]


def format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute melt-pool depth, mirrored width, and length for ET, TCAM, "
            "and GPR-corrected fields from matched GPR outputs or ParaView "
            "exports with Points_0/1/2 coordinate columns."
        )
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="GPR output CSV, such as matched_tcam_corrected_points.csv.",
    )
    parser.add_argument(
        "--liquidus",
        type=float,
        default=None,
        help=(
            "Liquidus temperature in K. If omitted, use the CSV Tliq column when "
            f"present, otherwise default to {DEFAULT_LIQUIDUS_K:g} K."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for writing the geometry summary as CSV.",
    )
    return parser.parse_args()


def liquidus_threshold(df: pd.DataFrame, override: float | None) -> tuple[pd.Series, str]:
    if override is not None:
        return pd.Series(float(override), index=df.index), "override"
    if "Tliq" in df.columns:
        tliq = pd.to_numeric(df["Tliq"], errors="coerce")
        if tliq.notna().any():
            return tliq, "Tliq"
    return pd.Series(DEFAULT_LIQUIDUS_K, index=df.index), "default"


def select_coordinate_columns(df: pd.DataFrame, options: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    for columns in options:
        if set(columns).issubset(df.columns):
            return columns
    option_text = ", ".join("(" + ", ".join(columns) + ")" for columns in options)
    raise ValueError(f"Missing coordinate columns. Expected one of: {option_text}")


def to_microns(coords: pd.DataFrame) -> pd.DataFrame:
    coords = coords.astype(float).copy()
    max_abs_coord = coords.abs().to_numpy().max()
    if max_abs_coord < METER_SCALE_THRESHOLD:
        coords *= 1.0e6
    return coords


def coordinate_label(row: pd.Series) -> str:
    return f"({format_number(row['x'])}, {format_number(row['y'])}, {format_number(row['z'])})"


def measure_field(
    df: pd.DataFrame,
    field: dict[str, object],
    threshold: pd.Series,
    threshold_source: str,
) -> dict[str, float | int | str]:
    name = str(field["name"])
    temperature_col = str(field["temperature"])

    if temperature_col not in df.columns:
        raise ValueError(f"Missing temperature column for {name}: {temperature_col}")

    x_col, y_col, z_col = select_coordinate_columns(df, field["coord_options"])
    valid = df[[temperature_col, x_col, y_col, z_col]].copy()
    valid["liquidus_threshold_K"] = threshold
    valid = valid.dropna()
    molten = valid[
        valid[temperature_col].astype(float) >= valid["liquidus_threshold_K"].astype(float)
    ].copy()

    result: dict[str, float | int | str] = {
        "field": name,
        "temperature_column": temperature_col,
        "x_column": x_col,
        "y_column": y_col,
        "z_column": z_col,
        "liquidus_source": threshold_source,
        "liquidus_min_K": float(valid["liquidus_threshold_K"].min()) if len(valid) else float("nan"),
        "liquidus_max_K": float(valid["liquidus_threshold_K"].max()) if len(valid) else float("nan"),
        "total_points": int(len(valid)),
        "molten_points": int(len(molten)),
    }

    if molten.empty:
        result.update(
            {
                "depth_um": float("nan"),
                "mirrored_width_um": float("nan"),
                "length_um": float("nan"),
                "deepest_coord_um": "",
                "widest_coord_um": "",
                "length_start_coord_um": "",
                "length_end_coord_um": "",
                "x_min_um": float("nan"),
                "x_max_um": float("nan"),
                "y_abs_max_um": float("nan"),
                "z_min_um": float("nan"),
                "z_max_um": float("nan"),
            }
        )
        return result

    coords = to_microns(molten[[x_col, y_col, z_col]])
    coords.columns = ["x", "y", "z"]

    deepest = coords.loc[coords["z"].idxmin()]
    widest = coords.loc[coords["y"].abs().idxmax()]
    length_start = coords.loc[coords["x"].idxmin()]
    length_end = coords.loc[coords["x"].idxmax()]

    x_min = coords["x"].min()
    x_max = coords["x"].max()
    y_abs_max = coords["y"].abs().max()
    z_min = coords["z"].min()
    z_max = coords["z"].max()

    result.update(
        {
            "depth_um": z_max - z_min,
            "mirrored_width_um": 2.0 * y_abs_max,
            "length_um": x_max - x_min,
            "deepest_coord_um": coordinate_label(deepest),
            "widest_coord_um": coordinate_label(widest),
            "length_start_coord_um": coordinate_label(length_start),
            "length_end_coord_um": coordinate_label(length_end),
            "x_min_um": x_min,
            "x_max_um": x_max,
            "y_abs_max_um": y_abs_max,
            "z_min_um": z_min,
            "z_max_um": z_max,
        }
    )
    return result


def measure_geometry(csv_path: Path, liquidus_k: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    threshold, threshold_source = liquidus_threshold(df, liquidus_k)
    rows = [measure_field(df, field, threshold, threshold_source) for field in FIELDS]
    return pd.DataFrame(rows)


def looks_like_liquidus_surface_export(summary: pd.DataFrame) -> bool:
    corrected = summary[summary["field"] == "corrected"]
    if corrected.empty:
        return False
    row = corrected.iloc[0]
    return (
        row["temperature_column"] == "T_corrected"
        and row["x_column"] == "Points_0"
        and int(row["molten_points"]) == int(row["total_points"])
    )


def print_summary(summary: pd.DataFrame, csv_path: Path) -> None:
    print(f"Input: {csv_path}")
    liquidus_min = summary["liquidus_min_K"].min()
    liquidus_max = summary["liquidus_max_K"].max()
    source = summary["liquidus_source"].iloc[0]
    if liquidus_min == liquidus_max:
        print(f"Liquidus: {format_number(liquidus_min)} K ({source})")
    else:
        print(
            f"Liquidus: {format_number(liquidus_min)} to "
            f"{format_number(liquidus_max)} K ({source})"
        )
    if looks_like_liquidus_surface_export(summary):
        print(
            "Note: this looks like a liquidus-surface/ParaView-style export. "
            "ET and TCAM rows are measured only on that exported surface, not from "
            "their full melt-pool volumes. Use matched_tcam_corrected_points.csv "
            "for full matched-volume ET/TCAM/corrected geometry."
        )
    print()
    for _, row in summary.iterrows():
        print(f"{row['field']}:")
        print(f"molten points: {int(row['molten_points'])}")
        print(f"depth: {format_number(row['depth_um'])} um at {row['deepest_coord_um']}")
        print(
            f"mirrored width: {format_number(row['mirrored_width_um'])} um "
            f"at {row['widest_coord_um']}"
        )
        print(
            f"length: {format_number(row['length_um'])} um "
            f"from {row['length_start_coord_um']} to {row['length_end_coord_um']}"
        )
        print()


def main() -> None:
    args = parse_args()
    summary = measure_geometry(args.csv, args.liquidus)
    print_summary(summary, args.csv)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print()
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
