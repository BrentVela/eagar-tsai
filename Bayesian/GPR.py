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


# ============================================================
# USER SETTINGS
# ============================================================

CASE_ROOT = Path("CalcFiles/Bayesian_Data/New")
ET_DIR = CASE_ROOT / "ET" / "alloy0" / "250_0.5_5um"
TCAM_DIR = CASE_ROOT / "TCAM" / "alloy0_250_0.5"

ET_CSV = ET_DIR / "ET_0_250W_0.5ms.csv"
ET_META_CSV = ET_DIR / "ET_meta_0_250W_0.5ms.csv"
TCAM_CSV = TCAM_DIR / "TC_cropped_0_250_0.5.csv"

# Exact GPR gets slow with too many points.
# Start small. Increase later if it runs comfortably.
MAX_TRAIN_POINTS = 5000
N_RESTARTS_OPTIMIZER = 0

RANDOM_SEED = 42
BASE_TEMPERATURE_K = 298.0

TCAM_COORDS_IN_METERS = True
TCAM_SUPPORT_DISTANCE_UM = 2.0
WRITE_EXTRAPOLATED_ET_ROI = False
WRITE_HOT_REGION_OUTPUT = False
HOT_REGION_T_ET_FRACTION_OF_TLIQ = 0.75
EXTRAPOLATED_ET_ROI_UM = {
    "x": (-200.0, 100.0),
    "y": (0.0, 100.0),
    "z": (-100.0, 0.0),
}
OUTPUT_DIR = TCAM_DIR / "GPR_outputs_5um/physics_based_2inputs_0restarts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_process_inputs(meta_csv):
    meta = pd.read_csv(meta_csv).iloc[0]
    return {
        "P": float(meta["power_w"]),
        "V": float(meta["velocity_m_s"]),
        "A": float(meta["absorptivity"]),
        "k": float(meta["thermal_conductivity_w_mk"]),
        "Cp": float(meta["specific_heat_j_kgk"]),
        "rho": float(meta["density_kg_m3"]),
        "Tliq": float(meta["liquidus_temperature_k"]),
        "beam_diameter_m": float(meta["beam_diameter_m"]),
    }


def load_and_merge_fields(et_csv, tcam_csv):
    et = pd.read_csv(et_csv)
    tcam = pd.read_csv(tcam_csv)

    required_et = {"x", "y", "z", "T_ET"}
    required_tcam = {"x", "y", "z", "T_TCAM"}

    missing_et = required_et - set(et.columns)
    missing_tcam = required_tcam - set(tcam.columns)

    if missing_et:
        raise ValueError(f"ET file is missing columns: {missing_et}")

    if missing_tcam:
        raise ValueError(f"TCAM file is missing columns: {missing_tcam}")

    if TCAM_COORDS_IN_METERS:
        tcam[["x", "y", "z"]] *= 1.0e6

    tree = cKDTree(et[["x", "y", "z"]].to_numpy())
    distance_um, et_index = tree.query(tcam[["x", "y", "z"]].to_numpy(), k=1)

    nearest_et = et.iloc[et_index].reset_index(drop=True)
    df = tcam.copy().reset_index(drop=True)
    df = df.rename(columns={"x": "x_TCAM", "y": "y_TCAM", "z": "z_TCAM"})
    df[["x_ET", "y_ET", "z_ET", "T_ET"]] = nearest_et[["x", "y", "z", "T_ET"]]
    df["match_distance_um"] = distance_um
    df["x"] = df["x_ET"]
    df["y"] = df["y_ET"]
    df["z"] = df["z_ET"]

    keep_cols = [
        "x",
        "y",
        "z",
        "x_TCAM",
        "y_TCAM",
        "z_TCAM",
        "match_distance_um",
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
    df["r_xy"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["r_3d"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def load_et_prediction_roi(et_csv, tcam_csv, inputs, roi_um):
    et = pd.read_csv(et_csv)
    roi_mask = (
        et["x"].between(*roi_um["x"])
        & et["y"].between(*roi_um["y"])
        & et["z"].between(*roi_um["z"])
    )
    et = et.loc[roi_mask].copy()
    if et.empty:
        raise ValueError(f"ET extrapolation ROI selected no points: {roi_um}")

    tcam = pd.read_csv(tcam_csv, usecols=["x", "y", "z"])

    if TCAM_COORDS_IN_METERS:
        tcam[["x", "y", "z"]] *= 1.0e6

    tcam_tree = cKDTree(tcam[["x", "y", "z"]].to_numpy())
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
    """

    if len(df) <= max_points:
        return df.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    tliq = float(df["Tliq"].iloc[0])

    abs_delta = df["delta_T"].abs()
    high_delta_cutoff = abs_delta.quantile(0.80)

    groups = {
        "cold": df[df["T_ET"] < 500],
        "warm": df[(df["T_ET"] >= 500) & (df["T_ET"] < 0.75 * tliq)],
        "near_liquidus": df[(df["T_ET"] >= 0.75 * tliq) & (df["T_ET"] <= 1.10 * tliq)],
        "molten_or_hot": df[df["T_ET"] > 1.10 * tliq],
        "high_discrepancy": df[abs_delta >= high_delta_cutoff],
    }

    group_fractions = {
        "cold": 0.10,
        "warm": 0.20,
        "near_liquidus": 0.25,
        "molten_or_hot": 0.25,
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

    print("Loading ET and TCAM fields...")
    df = load_and_merge_fields(ET_CSV, TCAM_CSV)

    inputs = load_process_inputs(ET_META_CSV)
    df = add_process_inputs(df, inputs)
    df = add_physics_features(df)

    print(f"Merged coordinate points: {len(df)}")
    print(
        "Nearest ET-grid distance: "
        f"max={df['match_distance_um'].max():.3f} um, "
        f"mean={df['match_distance_um'].mean():.3f} um"
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
        "beam_diameter_m",
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
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
    df_sample["T_corrected"] = df_sample["T_ET"] + df_sample["delta_T_pred"]

    corrected_sample_mae = mean_absolute_error(df_sample["T_TCAM"], df_sample["T_corrected"])
    corrected_sample_rmse = mean_squared_error(df_sample["T_TCAM"], df_sample["T_corrected"]) ** 0.5

    print("\nSampled field comparison:")
    print(f"Raw ET MAE vs TCAM       = {mean_absolute_error(df_sample['T_TCAM'], df_sample['T_ET']):.3f} K")
    print(f"Corrected ET MAE vs TCAM = {corrected_sample_mae:.3f} K")
    print(f"Raw ET RMSE vs TCAM      = {mean_squared_error(df_sample['T_TCAM'], df_sample['T_ET']) ** 0.5:.3f} K")
    print(f"Corrected ET RMSE vs TCAM= {corrected_sample_rmse:.3f} K")

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
    df["T_corrected"] = df["T_ET"] + df["delta_T_pred"]

    corrected_full_mae = mean_absolute_error(df["T_TCAM"], df["T_corrected"])
    corrected_full_rmse = mean_squared_error(df["T_TCAM"], df["T_corrected"]) ** 0.5

    print("\nFull field comparison:")
    print(f"Raw ET MAE vs TCAM       = {raw_mae:.3f} K")
    print(f"Corrected ET MAE vs TCAM = {corrected_full_mae:.3f} K")
    print(f"Raw ET RMSE vs TCAM      = {raw_rmse:.3f} K")
    print(f"Corrected ET RMSE vs TCAM= {corrected_full_rmse:.3f} K")

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
        df_et_roi = load_et_prediction_roi(ET_CSV, TCAM_CSV, inputs, EXTRAPOLATED_ET_ROI_UM)
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
        df_et_roi["T_corrected"] = df_et_roi["T_ET"] + df_et_roi["delta_T_pred"]
        supported_count = int(df_et_roi["has_tcam_neighbor"].sum())
        print(
            "Cropped ET ROI support summary: "
            f"{supported_count} / {len(df_et_roi)} points within "
            f"{TCAM_SUPPORT_DISTANCE_UM:g} um of a TCAM node."
        )

    # Save outputs.
    sampled_out = OUTPUT_DIR / "sampled_corrected_points.csv"
    full_out = OUTPUT_DIR / "matched_tcam_corrected_points.csv"
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
            "tcam_support_distance_um": TCAM_SUPPORT_DISTANCE_UM,
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
