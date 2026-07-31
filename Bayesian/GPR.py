"""
Test GPR correction for one ET/TCAM temperature-field pair.

Goal:
    Learn the residual:
        delta_T = T_TCAM - T_ET

    Then predict:
        T_corrected = T_ET + delta_T_pred

This follows the informative-prior idea:
    prior = ET
    ground truth = TCAM
    discrepancy = TCAM - ET
"""

from pathlib import Path
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial import cKDTree

try:
    from .et_temperature_field import load_et_temperature_field
except ImportError:
    from et_temperature_field import load_et_temperature_field


# ============================================================
# USER SETTINGS
# ============================================================

ET_FIELD = Path("beamer/figures/data/BU_ET/0_250W_0.5ms/ET_temperature.vti")
ET_META_CSV = Path("beamer/figures/data/BU_ET/0_250W_0.5ms/metadata.csv")
MATERIAL_DATA = Path("effective_cp_data.xlsx")
TCAM_FIELD = Path("beamer/figures/data/BU_TCAM/alloy0_250_0.5_row11/TCAM_cropped_alloy0_250_0.5.vtu")
OUTPUT_DIR = Path("beamer/figures/bayesian/cartesian_solvertest2")

# Exact GPR gets slow with too many points.
# Start small. Increase later if it runs comfortably.
MAX_TRAIN_POINTS = 5000
N_RESTARTS_OPTIMIZER = 0
LIQUIDUS_TRAINING_BAND_K = 300.0

RANDOM_SEED = 42
BASE_TEMPERATURE_K = 298.0
FEATURE_SCALE_RANGE = (0.0, 1.0)  # Use (0.01, 1.0) to avoid scaled zeros.
PROCESS_INPUT_RANGES = {
    "P": (50.0, 400.0),  # W
    "V": (0.01, 2.00),   # m/s
    "A": (0.30, 0.80),
}

TCAM_COORDS_IN_METERS = True
# Raw TCAM output uses +x as the scanning direction.
TCAM_SCAN_DIRECTION_X_SIGN = 1
TCAM_SUPPORT_DISTANCE_UM = 2.0
APPLY_EVAPORATION_ONSET_CAP = True
WRITE_EXTRAPOLATED_ET_ROI = False
WRITE_HOT_REGION_OUTPUT = False
HOT_REGION_T_ET_FRACTION_OF_TLIQ = 0.75
EXTRAPOLATED_ET_ROI_UM = {
    "x": (-200.0, 100.0),
    "y": (0.0, 100.0),
    "z": (-100.0, 0.0),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _load_evaporation_onset(meta, material_data):
    """Load the independently known equilibrium evaporation onset."""
    metadata_column = "evaporation_onset_temperature_k"
    if metadata_column in meta.index and pd.notna(meta[metadata_column]):
        return float(meta[metadata_column])

    table = pd.read_excel(material_data)
    source_column = "Boiling Onset (K)"
    if source_column not in table.columns:
        raise ValueError(
            f"{material_data} is missing required column '{source_column}'."
        )
    if "source_excel_row" not in meta.index or pd.isna(
        meta["source_excel_row"]
    ):
        raise ValueError(
            f"{meta.name} has no evaporation onset or source_excel_row. "
            "Regenerate ET metadata or provide a source Excel row."
        )

    dataframe_index = int(meta["source_excel_row"]) - 2
    if dataframe_index < 0 or dataframe_index >= len(table):
        raise IndexError(
            f"source_excel_row {int(meta['source_excel_row'])} is outside "
            f"{material_data}."
        )
    value = table.iloc[dataframe_index][source_column]
    if pd.isna(value):
        raise ValueError(
            f"{source_column} is empty at Excel row "
            f"{int(meta['source_excel_row'])} in {material_data}."
        )
    return float(value)


def load_process_inputs(meta_csv, material_data=MATERIAL_DATA):
    meta = pd.read_csv(meta_csv).iloc[0]
    return {
        "P": float(meta["power_w"]),
        "V": float(meta["velocity_m_s"]),
        "A": float(meta["absorptivity"]),
        "k": float(meta["thermal_conductivity_w_mk"]),
        "Cp": float(meta["specific_heat_j_kgk"]),
        "rho": float(meta["density_kg_m3"]),
        "Tliq": float(meta["liquidus_temperature_k"]),
        # Used as a physical output constraint. It is intentionally excluded
        # from the single-alloy GPR kernel because it is constant here.
        "Tevap": _load_evaporation_onset(meta, material_data),
    }


def set_corrected_temperature(dataframe, temperature_cap_k):
    """Construct corrected temperature and apply the physical TCAM ceiling."""
    corrected = dataframe["T_ET"] + dataframe["delta_T_pred"]
    if temperature_cap_k is None:
        dataframe["T_corrected"] = corrected
        return 0

    capped = corrected > temperature_cap_k
    dataframe["T_corrected"] = corrected.clip(upper=temperature_cap_k)
    return int(capped.sum())


def _nearest_axis_values(axis, query_values):
    """Return the nearest regular-grid coordinate for each query value."""
    insertion = np.searchsorted(axis, query_values)
    lower = np.clip(insertion - 1, 0, len(axis) - 1)
    upper = np.clip(insertion, 0, len(axis) - 1)
    use_upper = (
        np.abs(query_values - axis[upper])
        < np.abs(query_values - axis[lower])
    )
    nearest_index = np.where(use_upper, upper, lower)
    return axis[nearest_index]


def _normalize_tcam_table(tcam, source, require_temperature):
    """Normalize TCAM tabular columns to x, y, z, and T_TCAM."""
    aliases = {
        "Points_0": "x",
        "Points_1": "y",
        "Points_2": "z",
        "temperature": "T_TCAM",
    }
    rename_columns = {
        source: target
        for source, target in aliases.items()
        if source in tcam.columns and target not in tcam.columns
    }
    tcam = tcam.rename(columns=rename_columns)

    required = {"x", "y", "z"}
    if require_temperature:
        required.add("T_TCAM")
    missing = required - set(tcam.columns)
    if missing:
        raise ValueError(
            f"TCAM field {source} is missing columns: {missing}. "
            "Accepted ParaView aliases are Points_0, Points_1, Points_2, "
            "and temperature."
        )
    return tcam


def _load_tcam_vtu(path, require_temperature):
    """Load TCAM point coordinates and temperature from an unstructured grid."""
    import pyvista as pv

    root = pv.read(path)
    mesh = root.combine() if isinstance(root, pv.MultiBlock) else root
    if mesh.n_points == 0:
        raise ValueError(f"TCAM VTU contains no points: {path}")

    temperature_aliases = ("temperature", "T_TCAM", "Temperature_K")
    temperature_name = next(
        (name for name in temperature_aliases if name in mesh.point_data),
        None,
    )
    if temperature_name is None:
        cell_temperature_name = next(
            (name for name in temperature_aliases if name in mesh.cell_data),
            None,
        )
        if cell_temperature_name is not None:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
            temperature_name = cell_temperature_name

    if require_temperature and temperature_name is None:
        available = sorted(
            set(mesh.point_data.keys()) | set(mesh.cell_data.keys())
        )
        raise ValueError(
            f"TCAM VTU has no temperature array in {path}. "
            f"Accepted names: {temperature_aliases}; available: {available}"
        )

    tcam = pd.DataFrame(
        np.asarray(mesh.points, dtype=float),
        columns=["x", "y", "z"],
    )
    if temperature_name is not None:
        temperature = np.asarray(
            mesh.point_data[temperature_name],
            dtype=float,
        )
        if temperature.ndim != 1 or len(temperature) != len(tcam):
            raise ValueError(
                f"TCAM temperature array '{temperature_name}' in {path} "
                "must have one scalar value per mesh point."
            )
        tcam["T_TCAM"] = temperature
    return tcam


def _load_tcam_field(tcam_field_path, require_temperature=True):
    """
    Load a TCAM VTU or legacy CSV and return normalized point data.

    ParaView CSV coordinates may be named Points_0, Points_1, and Points_2.
    Every downstream table consistently receives x, y, z, and T_TCAM.
    """
    path = Path(tcam_field_path)
    suffix = path.suffix.lower()
    if suffix == ".vtu":
        return _load_tcam_vtu(path, require_temperature)
    if suffix == ".csv":
        return _normalize_tcam_table(
            pd.read_csv(path),
            path,
            require_temperature,
        )

    raise ValueError(
        f"Unsupported TCAM field {path}. Use .vtu or legacy .csv."
    )


def _tcam_coordinates_in_et_frame(tcam, et_scan_direction_x_sign):
    """Return TCAM x/y/z coordinates in micrometers and the ET axis frame."""
    coordinates = tcam[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
    if TCAM_COORDS_IN_METERS:
        coordinates *= 1.0e6
    coordinates[:, 0] *= (
        et_scan_direction_x_sign / TCAM_SCAN_DIRECTION_X_SIGN
    )
    return coordinates


def _deduplicate_tcam_points(tcam):
    """Merge repeated VTU block-boundary nodes after checking temperature."""
    coordinate_columns = ["x", "y", "z"]
    duplicate_mask = tcam.duplicated(coordinate_columns, keep=False)
    if not duplicate_mask.any():
        return tcam

    if "T_TCAM" in tcam.columns:
        grouped_temperature = tcam.loc[duplicate_mask].groupby(
            coordinate_columns,
            sort=False,
        )["T_TCAM"]
        bounds = grouped_temperature.agg(["min", "max"])
        spread = bounds["max"] - bounds["min"]
        tolerance = 1.0e-6 + 1.0e-10 * bounds.abs().max(axis=1)
        conflicting = spread > tolerance
        if conflicting.any():
            raise ValueError(
                f"TCAM field has {int(conflicting.sum())} duplicate "
                "coordinates with materially different temperatures; "
                f"maximum spread is {float(spread.max()):g} K."
            )
        value_columns = [
            column
            for column in tcam.columns
            if column not in coordinate_columns
        ]
        deduplicated = tcam.groupby(
            coordinate_columns,
            as_index=False,
            sort=False,
        )[value_columns].mean()
    else:
        deduplicated = tcam.drop_duplicates(
            coordinate_columns
        ).reset_index(drop=True)

    removed = len(tcam) - len(deduplicated)
    print(f"Merged {removed} duplicate TCAM mesh-point rows.")
    return deduplicated


def load_and_merge_fields(et_field_path, tcam_field_path):
    et_field = load_et_temperature_field(et_field_path)
    tcam = _deduplicate_tcam_points(
        _load_tcam_field(tcam_field_path)
    )

    axes = et_field.axes
    et_temperature_interpolator = et_field.interpolator()
    tcam_coordinates = _tcam_coordinates_in_et_frame(
        tcam,
        et_field.scan_direction_x_sign,
    )

    grid_tolerance_um = max(float(np.diff(axis)[0]) for axis in axes) * 1.0e-6
    for axis_index, (name, axis) in enumerate(zip(("x", "y", "z"), axes)):
        values = tcam_coordinates[:, axis_index]
        outside = (
            (values < axis[0] - grid_tolerance_um)
            | (values > axis[-1] + grid_tolerance_um)
        )
        if outside.any():
            raise ValueError(
                f"{outside.sum()} TCAM {name} coordinates fall outside the "
                f"ET interpolation range [{axis[0]}, {axis[-1]}] um."
            )
        tcam_coordinates[:, axis_index] = np.clip(
            values,
            axis[0],
            axis[-1],
        )

    nearest_et_coordinates = np.column_stack(
        [
            _nearest_axis_values(axis, tcam_coordinates[:, axis_index])
            for axis_index, axis in enumerate(axes)
        ]
    )
    distance_um = np.linalg.norm(
        tcam_coordinates - nearest_et_coordinates,
        axis=1,
    )

    df = tcam.copy()
    df[["x", "y", "z"]] = tcam_coordinates
    df["T_ET"] = et_temperature_interpolator(tcam_coordinates)
    df["nearest_et_grid_distance_um"] = distance_um

    df = _deduplicate_tcam_points(df)

    keep_cols = [
        "x",
        "y",
        "z",
        "nearest_et_grid_distance_um",
        "T_ET",
        "T_TCAM",
    ]
    df = df[keep_cols].copy()

    return df


def add_process_inputs(df, inputs):
    for key, value in inputs.items():
        df[key] = value
    return df


def add_physics_features(df, base_temperature_k=BASE_TEMPERATURE_K):
    df = df.copy()
    df["r_xy"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2) #try removing x,y,z inputs
    df["r_3d"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def load_et_prediction_roi(et_field_path, tcam_field_path, inputs, roi_um):
    tcam = _deduplicate_tcam_points(
        _load_tcam_field(
            tcam_field_path,
            require_temperature=False,
        )[["x", "y", "z"]].copy()
    )

    et_field = load_et_temperature_field(et_field_path)
    et = et_field.roi_dataframe(roi_um)

    tcam_tree = cKDTree(
        _tcam_coordinates_in_et_frame(
            tcam,
            et_field.scan_direction_x_sign,
        )
    )
    distance_um, _ = tcam_tree.query(et[["x", "y", "z"]].to_numpy(), k=1)

    et["distance_to_tcam_um"] = distance_um
    et["has_tcam_neighbor"] = distance_um <= TCAM_SUPPORT_DISTANCE_UM
    et = add_process_inputs(et, inputs)
    return et


def balanced_temperature_sample(df, max_points, seed=42):
    """
    Avoid training only on cold solid points.

    This samples from:
        cold region
        warm region
        near-liquidus region
        hot/melted region
        high-discrepancy points
    """

    if len(df) <= max_points:
        return df.copy().reset_index(drop=True)

    tliq = float(df["Tliq"].iloc[0])
    abs_delta = df["delta_T"].abs()
    high_delta_cutoff = abs_delta.quantile(0.80)

    groups = {
        "cold": df[df["T_ET"] < 500],
        "warm": df[
            (df["T_ET"] >= 500)
            & (df["T_ET"] < tliq - LIQUIDUS_TRAINING_BAND_K)
        ],
        "near_liquidus": df[
            ((df["T_ET"] - tliq).abs() <= LIQUIDUS_TRAINING_BAND_K)
            | ((df["T_TCAM"] - tliq).abs() <= LIQUIDUS_TRAINING_BAND_K)
        ],
        "molten_or_hot": df[
            df["T_ET"] > tliq + LIQUIDUS_TRAINING_BAND_K
        ],
        "high_discrepancy": df[abs_delta >= high_delta_cutoff],
    }

    group_fractions = {
        "cold": 0.10,
        "warm": 0.15,
        "near_liquidus": 0.35,
        "molten_or_hot": 0.20,
        "high_discrepancy": 0.20,
    }
    sampled_parts = []
    used_indices = set()

    for name, group in groups.items():
        if len(group) == 0:
            print(f"Sampled 0 points from {name}")
            continue

        n = min(int(round(max_points * group_fractions[name])), len(group))
        sampled = group.sample(n=n, random_state=seed)
        sampled_parts.append(sampled)
        used_indices.update(sampled.index.tolist())
        print(f"Sampled {len(sampled)} points from {name}")

    sampled_df = pd.concat(sampled_parts, axis=0) if sampled_parts else pd.DataFrame()
    if len(sampled_df) > 0:
        sampled_df = sampled_df.loc[~sampled_df.index.duplicated(keep="first")]

    # Fill remaining points randomly from points not already sampled.
    remaining = max_points - len(sampled_df)

    if remaining > 0:
        leftover = df.loc[~df.index.isin(used_indices)]

        if len(leftover) > 0:
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=seed)
            sampled_df = pd.concat([sampled_df, extra], axis=0)

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled_df


def make_kernel(n_features):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=np.ones(n_features),
            length_scale_bounds=(0.05, 1e3),
        )
        + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(1e-6, 1e3),
        )
    )


def fit_feature_scaler(X_train, feature_cols):
    """
    Fit using fixed ranges for selected process inputs and training-data
    ranges for all other features.
    """
    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)

    for name, (lower, upper) in PROCESS_INPUT_RANGES.items():
        if lower >= upper:
            raise ValueError(f"Invalid normalization range for {name}: {(lower, upper)}")

        if name not in feature_cols:
            continue

        column_index = feature_cols.index(name)
        observed_min = X_train[:, column_index].min()
        observed_max = X_train[:, column_index].max()

        if observed_min < lower or observed_max > upper:
            raise ValueError(
                f"{name} values [{observed_min}, {observed_max}] fall outside "
                f"the configured normalization range [{lower}, {upper}]."
            )

        feature_min[column_index] = lower
        feature_max[column_index] = upper

    scaler = MinMaxScaler(feature_range=FEATURE_SCALE_RANGE)
    scaler.fit(np.vstack([feature_min, feature_max]))
    return scaler


def predict_in_batches(model, scaler, X, batch_size=50000, return_std=True):
    """
    Predict in batches so large fields do not overload memory.
    """

    means = []
    stds = []

    for start in range(0, len(X), batch_size):
        stop = start + batch_size
        X_batch_scaled = scaler.transform(X[start:stop])

        if return_std:
            mean_batch, std_batch = model.predict(X_batch_scaled, return_std=True)
            means.append(mean_batch)
            stds.append(std_batch)
        else:
            mean_batch = model.predict(X_batch_scaled)
            means.append(mean_batch)

    means = np.concatenate(means)

    if return_std:
        stds = np.concatenate(stds)
        return means, stds

    return means


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ET and TCAM fields...")
    df = load_and_merge_fields(ET_FIELD, TCAM_FIELD)

    inputs = load_process_inputs(ET_META_CSV)
    df = add_process_inputs(df, inputs)
    df = add_physics_features(df)
    temperature_cap_k = (
        float(inputs["Tevap"])
        if APPLY_EVAPORATION_ONSET_CAP
        else None
    )

    print(f"Merged coordinate points: {len(df)}")
    print(
        "Nearest ET-grid distance: "
        f"max={df['nearest_et_grid_distance_um'].max():.3f} um, "
        f"mean={df['nearest_et_grid_distance_um'].mean():.3f} um"
    )
    if temperature_cap_k is not None:
        print(
            "Equilibrium evaporation-onset ceiling: "
            f"{temperature_cap_k:.6f} K"
        )

    # Residual target.
    # This is the quantity the GPR learns.
    df["delta_T"] = df["T_TCAM"] - df["T_ET"]

    print("\nRaw ET vs TCAM error statistics:")
    raw_mae = mean_absolute_error(df["T_TCAM"], df["T_ET"])
    raw_rmse = mean_squared_error(df["T_TCAM"], df["T_ET"]) ** 0.5
    print(f"Raw ET MAE  = {raw_mae:.3f} K")
    print(f"Raw ET RMSE = {raw_rmse:.3f} K")

    print("\nResidual delta_T statistics:")
    print(df["delta_T"].describe())

    # Features for the GPR.
    # Only use columns that are known during prediction.
    candidate_feature_cols = [
        "P",
        "V",
        "A",
        "k",
        "Cp",
        "rho",
        "Tliq",
        "x",
        "y",
        "z",
        "T_ET",
        "r_xy",
        "r_3d",
    ]

    feature_cols = [col for col in candidate_feature_cols if col in df.columns]

    print("\nUsing GPR input features:")
    for col in feature_cols:
        print(f"  - {col}")

    target_col = "delta_T"

    # Sample training points.
    print("\nSampling points for exact GPR...")
    df_sample = balanced_temperature_sample(df, MAX_TRAIN_POINTS, RANDOM_SEED)
    print(f"Using {len(df_sample)} sampled points for GPR training/testing.")

    X = df_sample[feature_cols].to_numpy(dtype=float)
    y = df_sample[target_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED
    )

    # Scale features.
    # This matters because P, V, x, y, z, and temperature have very different magnitudes.
    scaler = fit_feature_scaler(X_train, feature_cols)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build GPR.
    kernel = make_kernel(n_features=X_train_scaled.shape[1])

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        optimizer="fmin_l_bfgs_b",
        random_state=RANDOM_SEED
    )

    print("\nTraining GPR on residual delta_T = T_TCAM - T_ET...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_train_scaled, y_train)

    print("\nLearned kernel:")
    print(gpr.kernel_)

    # Evaluate residual prediction.
    print("\nEvaluating residual prediction on held-out sampled points...")
    y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

    residual_mae = mean_absolute_error(y_test, y_pred)
    residual_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    residual_r2 = r2_score(y_test, y_pred)

    print(f"Residual MAE  = {residual_mae:.3f} K")
    print(f"Residual RMSE = {residual_rmse:.3f} K")
    print(f"Residual R^2  = {residual_r2:.3f}")

    # Predict correction for sampled points.
    X_sample_scaled = scaler.transform(X)
    delta_pred_sample, delta_std_sample = gpr.predict(X_sample_scaled, return_std=True)

    df_sample["delta_T_pred"] = delta_pred_sample
    df_sample["delta_T_std"] = delta_std_sample
    sample_cap_count = set_corrected_temperature(
        df_sample,
        temperature_cap_k,
    )

    corrected_sample_mae = mean_absolute_error(df_sample["T_TCAM"], df_sample["T_corrected"])
    corrected_sample_rmse = mean_squared_error(df_sample["T_TCAM"], df_sample["T_corrected"]) ** 0.5

    print("\nSampled field comparison:")
    print(f"Raw ET MAE vs TCAM       = {mean_absolute_error(df_sample['T_TCAM'], df_sample['T_ET']):.3f} K")
    print(f"Corrected ET MAE vs TCAM = {corrected_sample_mae:.3f} K")
    print(f"Raw ET RMSE vs TCAM      = {mean_squared_error(df_sample['T_TCAM'], df_sample['T_ET']) ** 0.5:.3f} K")
    print(f"Corrected ET RMSE vs TCAM= {corrected_sample_rmse:.3f} K")
    if temperature_cap_k is not None:
        print(f"Predictions capped at evaporation ceiling: {sample_cap_count}")

    # Predict correction for the full merged field.
    # This uses the trained GPR from sampled data.
    print("\nPredicting correction over full merged field...")
    X_full = df[feature_cols].to_numpy(dtype=float)

    delta_pred_full, delta_std_full = predict_in_batches(
        model=gpr,
        scaler=scaler,
        X=X_full,
        batch_size=50000,
        return_std=True
    )

    df["delta_T_pred"] = delta_pred_full
    df["delta_T_std"] = delta_std_full
    full_cap_count = set_corrected_temperature(df, temperature_cap_k)

    corrected_full_mae = mean_absolute_error(df["T_TCAM"], df["T_corrected"])
    corrected_full_rmse = mean_squared_error(df["T_TCAM"], df["T_corrected"]) ** 0.5

    print("\nFull field comparison:")
    print(f"Raw ET MAE vs TCAM       = {raw_mae:.3f} K")
    print(f"Corrected ET MAE vs TCAM = {corrected_full_mae:.3f} K")
    print(f"Raw ET RMSE vs TCAM      = {raw_rmse:.3f} K")
    print(f"Corrected ET RMSE vs TCAM= {corrected_full_rmse:.3f} K")
    if temperature_cap_k is not None:
        print(f"Predictions capped at evaporation ceiling: {full_cap_count}")

    df_hot = None
    if WRITE_HOT_REGION_OUTPUT:
        hot_threshold = HOT_REGION_T_ET_FRACTION_OF_TLIQ * float(inputs["Tliq"])
        df_hot = df[df["T_ET"] >= hot_threshold].copy()
        print(
            "\nHot-region matched output: "
            f"{len(df_hot)} / {len(df)} points with T_ET >= "
            f"{HOT_REGION_T_ET_FRACTION_OF_TLIQ:g} * Tliq = {hot_threshold:.3f} K."
        )

    df_et_roi = None
    if WRITE_EXTRAPOLATED_ET_ROI:
        print("\nPredicting correction over cropped ET ROI, including TCAM-unsupported regions...")
        df_et_roi = load_et_prediction_roi(
            ET_FIELD,
            TCAM_FIELD,
            inputs,
            EXTRAPOLATED_ET_ROI_UM,
        )
        df_et_roi = add_physics_features(df_et_roi)
        X_et_roi = df_et_roi[feature_cols].to_numpy(dtype=float)
        delta_pred_et_roi, delta_std_et_roi = predict_in_batches(
            model=gpr,
            scaler=scaler,
            X=X_et_roi,
            batch_size=50000,
            return_std=True
        )
        df_et_roi["delta_T_pred"] = delta_pred_et_roi
        df_et_roi["delta_T_std"] = delta_std_et_roi
        set_corrected_temperature(df_et_roi, temperature_cap_k)
        supported_count = int(df_et_roi["has_tcam_neighbor"].sum())
        print(
            "Cropped ET ROI support summary: "
            f"{supported_count} / {len(df_et_roi)} points within "
            f"{TCAM_SUPPORT_DISTANCE_UM:g} um of a TCAM node."
        )

    # Save outputs.
    sampled_out = OUTPUT_DIR / "sampled_corrected_points.csv"
    full_out = OUTPUT_DIR / "BU_corrected_data.csv"
    hot_out = OUTPUT_DIR / "hot_region_corrected_points.csv"
    et_roi_out = OUTPUT_DIR / "et_roi_corrected_extrapolated.csv"
    model_out = OUTPUT_DIR / "gpr_et_to_tcam_residual_model.joblib"

    df_sample.to_csv(sampled_out, index=False)
    df.to_csv(full_out, index=False)
    if df_hot is not None:
        df_hot.to_csv(hot_out, index=False)
    if df_et_roi is not None:
        df_et_roi.to_csv(et_roi_out, index=False)

    joblib.dump(
        {
            "gpr": gpr,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "process_inputs": inputs,
            "process_input_ranges": PROCESS_INPUT_RANGES,
            "liquidus_training_band_k": LIQUIDUS_TRAINING_BAND_K,
            "tcam_support_distance_um": TCAM_SUPPORT_DISTANCE_UM,
            "apply_evaporation_onset_cap": APPLY_EVAPORATION_ONSET_CAP,
            "temperature_cap_k": temperature_cap_k,
            "write_hot_region_output": WRITE_HOT_REGION_OUTPUT,
            "hot_region_t_et_fraction_of_tliq": HOT_REGION_T_ET_FRACTION_OF_TLIQ,
            "write_extrapolated_et_roi": WRITE_EXTRAPOLATED_ET_ROI,
            "extrapolated_et_roi_um": EXTRAPOLATED_ET_ROI_UM,
            "description": "GPR trained on delta_T = T_TCAM - T_ET. Corrected temperature is T_ET + delta_T_pred."
        },
        model_out
    )

    print("\nSaved:")
    print(f"  {sampled_out}")
    print(f"  {full_out}")
    if df_hot is not None:
        print(f"  {hot_out}")
    if df_et_roi is not None:
        print(f"  {et_roi_out}")
    print(f"  {model_out}")

    # Diagnostic plots.
    print("\nMaking diagnostic plots...")

    # 1. True residual vs predicted residual.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=10, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel("True residual, T_TCAM - T_ET [K]")
    plt.ylabel("Predicted residual [K]")
    plt.title("GPR residual prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_prediction_scatter.png", dpi=200)
    plt.close()

    # 2. TCAM vs raw/corrected temperature.
    plt.figure(figsize=(6, 6))
    plt.scatter(df_sample["T_TCAM"], df_sample["T_ET"], s=10, alpha=0.35, label="Raw ET")
    plt.scatter(df_sample["T_TCAM"], df_sample["T_corrected"], s=10, alpha=0.35, label="Corrected ET")
    min_val = min(df_sample["T_TCAM"].min(), df_sample["T_ET"].min(), df_sample["T_corrected"].min())
    max_val = max(df_sample["T_TCAM"].max(), df_sample["T_ET"].max(), df_sample["T_corrected"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel("TCAM temperature [K]")
    plt.ylabel("Predicted temperature [K]")
    plt.legend()
    plt.title("Raw ET vs corrected ET")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temperature_comparison_scatter.png", dpi=200)
    plt.close()

    # 3. Uncertainty vs absolute residual error.
    df_sample["abs_residual_error"] = np.abs(df_sample["delta_T"] - df_sample["delta_T_pred"])

    plt.figure(figsize=(6, 5))
    plt.scatter(df_sample["delta_T_std"], df_sample["abs_residual_error"], s=10, alpha=0.4)
    plt.xlabel("Predicted residual standard deviation [K]")
    plt.ylabel("Absolute residual prediction error [K]")
    plt.title("GPR uncertainty check")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uncertainty_vs_error.png", dpi=200)
    plt.close()

    print("Saved plots:")
    print(f"  {OUTPUT_DIR / 'residual_prediction_scatter.png'}")
    print(f"  {OUTPUT_DIR / 'temperature_comparison_scatter.png'}")
    print(f"  {OUTPUT_DIR / 'uncertainty_vs_error.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
