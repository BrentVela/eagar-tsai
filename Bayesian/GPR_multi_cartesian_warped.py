#!/usr/bin/env python3
"""Cartesian ET warp followed by a conventional multicase residual GPR.

The default model contains no cylindrical, radial, or angular features.  The
``--radial-feature`` ablation can add exactly one derived Cartesian distance,
``r_yz`` or ``r_3d``, while keeping all other modeling choices fixed.  For each
of 15 training cases, a bounded six-parameter Cartesian warp is fit between ET
and TCAM.  A small P-V response surface predicts the warp for the complete
held-out process condition.  The default residual GP uses exactly

    P, V, x, y, z, T_warped_ET

and learns ``T_TCAM - T_warped_ET``.  The held-out prediction is frozen on a
coarse Cartesian grid spanning the ET domain before the held-out TCAM field is
opened.  A separate comparison VTI is written afterward for ParaView.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from .GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
    )
    from .GPR_multi import (
        DEFAULT_ET_ROOT,
        DEFAULT_HELD_OUT_CASE,
        DEFAULT_MATERIAL_DATA,
        DEFAULT_TCAM_ROOT,
        DEFAULT_TRAINING_CASES,
        EVAPORATION_MATCH_TOLERANCE_K,
        _parse_case_specification,
        discover_paired_cases,
        resolve_case_split,
        sample_training_case,
        write_case_map,
    )
    from .GPR_cylindrical_warped import (
        ET_GRID_GP_BATCH_SIZE,
        ET_GRID_GP_STRIDE,
        ETPriorCase,
        ETGridPrediction,
        LOG_WARP_PARAMETER_INDICES,
        WARP_BOUNDS,
        WARP_DIFFERENTIAL_EVOLUTION_ITERATIONS,
        WARP_PARAMETER_NAMES,
        WARP_POINTS_PER_CASE,
        WarpParameters,
        evaluate_warped_et,
        fit_case_warp,
        load_case_physics,
        write_et_grid_prediction_vti,
    )
    from .et_temperature_field import load_et_temperature_field
except ImportError:
    from GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
    )
    from GPR_multi import (
        DEFAULT_ET_ROOT,
        DEFAULT_HELD_OUT_CASE,
        DEFAULT_MATERIAL_DATA,
        DEFAULT_TCAM_ROOT,
        DEFAULT_TRAINING_CASES,
        EVAPORATION_MATCH_TOLERANCE_K,
        _parse_case_specification,
        discover_paired_cases,
        resolve_case_split,
        sample_training_case,
        write_case_map,
    )
    from GPR_cylindrical_warped import (
        ET_GRID_GP_BATCH_SIZE,
        ET_GRID_GP_STRIDE,
        ETPriorCase,
        ETGridPrediction,
        LOG_WARP_PARAMETER_INDICES,
        WARP_BOUNDS,
        WARP_DIFFERENTIAL_EVOLUTION_ITERATIONS,
        WARP_PARAMETER_NAMES,
        WARP_POINTS_PER_CASE,
        WarpParameters,
        evaluate_warped_et,
        fit_case_warp,
        load_case_physics,
        write_et_grid_prediction_vti,
    )
    from et_temperature_field import load_et_temperature_field


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/warped_cartesian_gpr"
    / "heldout_250W_0.5ms_15trainingcases"
)
RYZ_OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/warped_cartesian_ryz_gpr"
    / "heldout_250W_0.5ms_15trainingcases"
)
R3D_OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/warped_cartesian_r3d_gpr"
    / "heldout_250W_0.5ms_15trainingcases"
)

BASE_FEATURE_COLUMNS = ["P", "V", "x", "y", "z", "T_warped_ET"]
# Backward-compatible name for the original controlled six-input experiment.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
POINTS_PER_TRAINING_CASE = 500
N_RESTARTS_OPTIMIZER = 1
WARP_SURFACE_RIDGE_ALPHA = 2.0e-2


def feature_columns_for(radial_feature: str) -> list[str]:
    if radial_feature == "none":
        return list(BASE_FEATURE_COLUMNS)
    if radial_feature == "r_yz":
        return ["P", "V", "x", "y", "z", "r_yz", "T_warped_ET"]
    if radial_feature == "r_3d":
        return ["P", "V", "x", "y", "z", "r_3d", "T_warped_ET"]
    raise ValueError(f"Unknown radial feature: {radial_feature!r}")


def add_requested_radial_feature(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = dataframe.copy()
    if "r_yz" in feature_columns:
        result["r_yz"] = np.hypot(
            result["y"].to_numpy(dtype=float),
            result["z"].to_numpy(dtype=float),
        )
    if "r_3d" in feature_columns:
        coordinates = result[["x", "y", "z"]].to_numpy(dtype=float)
        result["r_3d"] = np.linalg.norm(coordinates, axis=1)
    return result


class CartesianWarpParameterSurface:
    """Regularized quadratic P-V response surface for six warp parameters."""

    def __init__(self, ridge_alpha: float = WARP_SURFACE_RIDGE_ALPHA):
        self.ridge_alpha = float(ridge_alpha)
        self.models: list | None = None

    @staticmethod
    def _features(power_w: np.ndarray, velocity_m_s: np.ndarray) -> np.ndarray:
        power = np.asarray(power_w, dtype=float)
        velocity = np.asarray(velocity_m_s, dtype=float)
        if power.shape != velocity.shape:
            raise ValueError("power_w and velocity_m_s must have matching shapes.")
        if np.any(power <= 0.0) or np.any(velocity <= 0.0):
            raise ValueError("Power and velocity must be positive.")
        return np.column_stack([power, velocity])

    def fit(
        self,
        power_w: np.ndarray,
        velocity_m_s: np.ndarray,
        parameter_values: np.ndarray,
    ) -> "CartesianWarpParameterSurface":
        values = np.asarray(parameter_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(WARP_PARAMETER_NAMES):
            raise ValueError("parameter_values has the wrong shape.")
        X = self._features(power_w, velocity_m_s)
        if len(X) != len(values):
            raise ValueError("Process inputs and parameter values must align.")

        self.models = []
        for index, target in enumerate(values.T):
            if index in LOG_WARP_PARAMETER_INDICES:
                if np.any(target <= 0.0):
                    raise ValueError("Positive warp parameters must exceed zero.")
                target = np.log(target)
            model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=2, include_bias=False),
                Ridge(alpha=self.ridge_alpha),
            )
            model.fit(X, target)
            self.models.append(model)
        return self

    def predict(
        self,
        power_w: np.ndarray,
        velocity_m_s: np.ndarray,
    ) -> np.ndarray:
        if self.models is None:
            raise RuntimeError("Warp parameter surface has not been fit.")
        X = self._features(power_w, velocity_m_s)
        columns = []
        for index, model in enumerate(self.models):
            values = model.predict(X)
            if index in LOG_WARP_PARAMETER_INDICES:
                values = np.exp(values)
            columns.append(values)
        return np.clip(
            np.column_stack(columns),
            WARP_BOUNDS[:, 0],
            WARP_BOUNDS[:, 1],
        )


def fit_warp_surface(warp_table: pd.DataFrame) -> CartesianWarpParameterSurface:
    return CartesianWarpParameterSurface().fit(
        warp_table["power_w"].to_numpy(),
        warp_table["velocity_m_s"].to_numpy(),
        warp_table[list(WARP_PARAMETER_NAMES)].to_numpy(),
    )


def leave_one_case_out_warp_predictions(warp_table: pd.DataFrame) -> pd.DataFrame:
    """Training-only check of P-V interpolation for every fitted warp."""
    rows = []
    for held_index in range(len(warp_table)):
        fit_table = warp_table.drop(index=warp_table.index[held_index])
        surface = fit_warp_surface(fit_table)
        held = warp_table.iloc[held_index]
        predicted = surface.predict(
            np.asarray([held["power_w"]]),
            np.asarray([held["velocity_m_s"]]),
        )[0]
        row = {
            "case_id": held["case_id"],
            "power_w": held["power_w"],
            "velocity_m_s": held["velocity_m_s"],
        }
        for name, value in zip(WARP_PARAMETER_NAMES, predicted):
            row[f"actual_{name}"] = held[name]
            row[f"predicted_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_warped_training_table(
    cases,
    material_data: Path,
    points_per_case: int,
    warp_points_per_case: int,
    warp_max_iterations: int,
    seed: int,
    feature_columns: list[str],
):
    """Fit each training warp and sample Cartesian residual-GP observations."""
    sampled_cases = []
    warp_rows = []
    case_rows = []

    for index, case in enumerate(cases):
        print(f"\nLoading training case {index + 1}/{len(cases)}: {case.case_id}")
        dataframe = load_and_merge_fields(case.et_field, case.tcam_field)
        inputs = load_process_inputs(case.et_metadata, material_data)
        for key, value in inputs.items():
            dataframe[key] = value
        dataframe["case_id"] = case.case_id

        physics = load_case_physics(case.et_metadata, material_data)
        et_field = load_et_temperature_field(case.et_field)
        parameters, diagnostics, _ = fit_case_warp(
            dataframe,
            et_field,
            physics,
            max_points=warp_points_per_case,
            seed=seed + index,
            max_iterations=warp_max_iterations,
        )
        print(
            "Warp "
            f"{'accepted' if diagnostics['fitted_warp_accepted'] else 'rejected'}; "
            f"full RMSE {diagnostics['identity_full_rmse_k']:.1f} -> "
            f"{diagnostics['selected_full_rmse_k']:.1f} K"
        )

        coordinates = dataframe[["x", "y", "z"]].to_numpy(dtype=float)
        dataframe["T_warped_ET"] = evaluate_warped_et(
            et_field,
            coordinates,
            parameters,
            physics.sigma_um,
        )
        dataframe = add_requested_radial_feature(dataframe, feature_columns)
        dataframe["delta_T_warped"] = (
            dataframe["T_TCAM"] - dataframe["T_warped_ET"]
        )
        # The inherited sampler uses this generic name for discrepancy strata.
        dataframe["delta_T"] = dataframe["delta_T_warped"]
        sampled = sample_training_case(
            dataframe,
            points_per_case=points_per_case,
            seed=seed + index,
        )
        sampled_cases.append(sampled)

        warp_rows.append(
            {
                "case_id": case.case_id,
                "power_w": case.power_w,
                "velocity_m_s": case.velocity_m_s,
                **asdict(parameters),
                **diagnostics,
            }
        )
        evaporation_threshold = inputs["Tevap"] - EVAPORATION_MATCH_TOLERANCE_K
        case_rows.append(
            {
                "case_id": case.case_id,
                "power_w": case.power_w,
                "velocity_m_s": case.velocity_m_s,
                "available_points": len(dataframe),
                "sampled_points": len(sampled),
                "sampled_evaporation_points": int(
                    (sampled["T_TCAM"] >= evaporation_threshold).sum()
                ),
                "raw_et_rmse_k": float(
                    mean_squared_error(dataframe["T_TCAM"], dataframe["T_ET"])
                    ** 0.5
                ),
                "warped_et_rmse_k": diagnostics["selected_full_rmse_k"],
            }
        )

    training = pd.concat(sampled_cases, ignore_index=True)
    training = training.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return training, pd.DataFrame(warp_rows), pd.DataFrame(case_rows)


def fit_residual_gpr(
    training: pd.DataFrame,
    seed: int,
    n_restarts_optimizer: int,
    feature_columns: list[str],
):
    missing = set(feature_columns + ["delta_T_warped"]) - set(training.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {missing}")
    X = training[feature_columns].to_numpy(dtype=float)
    target = training["delta_T_warped"].to_numpy(dtype=float)
    scaler = fit_feature_scaler(X, feature_columns)
    model = GaussianProcessRegressor(
        kernel=make_kernel(len(feature_columns)),
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        optimizer="fmin_l_bfgs_b",
        random_state=seed,
    )
    print(
        f"Fitting Cartesian warped-prior RBF GPR on {len(training)} points "
        f"with features {feature_columns}..."
    )
    model.fit(scaler.transform(X), target)
    print(f"Learned kernel: {model.kernel_}")
    return model, scaler


def _coarse_axis(axis: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("Prediction grid stride must be positive.")
    intervals = int(np.ceil((len(axis) - 1) / stride))
    return np.linspace(float(axis[0]), float(axis[-1]), intervals + 1)


def _cartesian_coordinates(axes) -> np.ndarray:
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack(
        [xx.ravel(order="C"), yy.ravel(order="C"), zz.ravel(order="C")]
    )


def generate_coarse_prediction(
    case: ETPriorCase,
    material_data: Path,
    warp_surface: CartesianWarpParameterSurface,
    model,
    scaler,
    stride: int,
    batch_size: int,
    feature_columns: list[str],
):
    """Generate the complete held-out field without accessing held-out TCAM."""
    physics = load_case_physics(case.et_metadata, material_data)
    parameters = WarpParameters.from_array(
        warp_surface.predict(
            np.asarray([case.power_w]),
            np.asarray([case.velocity_m_s]),
        )[0]
    )
    et_field = load_et_temperature_field(case.et_field)
    axes = tuple(_coarse_axis(axis, stride) for axis in et_field.axes)
    coordinates = _cartesian_coordinates(axes)
    dimensions = tuple(len(axis) for axis in axes)
    print(
        "Generating TCAM-blind warped Cartesian grid: "
        f"{dimensions} ({len(coordinates)} points, stride={stride})"
    )

    T_et = et_field.interpolator(bounds_error=False)(coordinates)
    T_warped = evaluate_warped_et(
        et_field,
        coordinates,
        parameters,
        physics.sigma_um,
    )
    support = pd.DataFrame(coordinates, columns=["x", "y", "z"])
    support["P"] = case.power_w
    support["V"] = case.velocity_m_s
    support["T_warped_ET"] = T_warped
    support = add_requested_radial_feature(support, feature_columns)
    delta_mean, delta_std = predict_in_batches(
        model=model,
        scaler=scaler,
        X=support[feature_columns].to_numpy(dtype=float),
        batch_size=batch_size,
        return_std=True,
    )
    unconstrained = T_warped + delta_mean
    arrays = {
        "T_ET": T_et,
        "T_warped_ET": T_warped,
        "delta_T_pred": delta_mean,
        "delta_T_std": delta_std,
        "T_pred_unconstrained": unconstrained,
        "T_pred": np.clip(
            unconstrained, BASE_TEMPERATURE_K, physics.evaporation_k
        ),
        "T_pred_lower_95": np.clip(
            unconstrained - 1.96 * delta_std,
            BASE_TEMPERATURE_K,
            physics.evaporation_k,
        ),
        "T_pred_upper_95": np.clip(
            unconstrained + 1.96 * delta_std,
            BASE_TEMPERATURE_K,
            physics.evaporation_k,
        ),
    }
    arrays = {
        name: np.asarray(values, dtype=np.float32).reshape(dimensions, order="C")
        for name, values in arrays.items()
    }
    bounds = {
        "ambient_lower_bound": int(np.sum(unconstrained < BASE_TEMPERATURE_K)),
        "evaporation_upper_bound": int(
            np.sum(unconstrained > physics.evaporation_k)
        ),
    }
    prediction = ETGridPrediction(
        axes_um=axes,
        arrays_k=arrays,
        gp_support_dimensions=dimensions,
        native_dimensions=dimensions,
        gp_stride=stride,
        bound_counts=bounds,
    )
    return prediction, physics, parameters


def evaluate_frozen_prediction(case, material_data: Path, prediction):
    """Open held-out TCAM only after the prediction field has been written."""
    print(f"Revealing held-out TCAM only for evaluation: {case.case_id}")
    held_out = load_and_merge_fields(case.et_field, case.tcam_field)
    inputs = load_process_inputs(case.et_metadata, material_data)
    coordinates = held_out[["x", "y", "z"]].to_numpy(dtype=float)
    for name in prediction.arrays_k:
        held_out[name] = prediction.interpolate(coordinates, name)
    held_out["delta_T_warped"] = (
        held_out["T_TCAM"] - held_out["T_warped_ET"]
    )
    return held_out, inputs


def regression_metrics(truth, prediction) -> dict:
    return {
        "mae_k": float(mean_absolute_error(truth, prediction)),
        "rmse_k": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
    }


def masked_metrics(truth, prediction, mask: np.ndarray) -> dict:
    count = int(np.sum(mask))
    if count < 2:
        return {"point_count": count, "mae_k": None, "rmse_k": None, "r2": None}
    return {"point_count": count, **regression_metrics(truth[mask], prediction[mask])}


def melt_pool_point_extents(dataframe, column: str, liquidus_k: float) -> dict:
    molten = dataframe[dataframe[column] >= liquidus_k]
    if molten.empty:
        return {
            "molten_point_count": 0,
            "length_um": 0.0,
            "full_width_um": 0.0,
            "depth_um": 0.0,
        }
    return {
        "molten_point_count": len(molten),
        "length_um": float(molten["x"].max() - molten["x"].min()),
        "full_width_um": float(2.0 * molten["y"].abs().max()),
        "depth_um": float(max(0.0, -molten["z"].min())),
    }


def calculate_metrics(held_out: pd.DataFrame, liquidus_k: float) -> dict:
    truth = held_out["T_TCAM"]
    residual_truth = truth - held_out["T_warped_ET"]
    residual_error = residual_truth - held_out["delta_T_pred"]
    std = np.maximum(held_out["delta_T_std"].to_numpy(), 1.0e-12)
    hot_threshold = BASE_TEMPERATURE_K + 0.75 * (
        liquidus_k - BASE_TEMPERATURE_K
    )
    masks = {
        "hot_region": truth.to_numpy() >= hot_threshold,
        "above_liquidus": truth.to_numpy() >= liquidus_k,
    }
    result = {}
    for name, column in {
        "raw_et": "T_ET",
        "warped_et": "T_warped_ET",
        "warped_cartesian_gpr_unconstrained": "T_pred_unconstrained",
        "warped_cartesian_gpr": "T_pred",
    }.items():
        result[name] = {
            "full_field": regression_metrics(truth, held_out[column]),
            **{
                mask_name: masked_metrics(truth, held_out[column], mask)
                for mask_name, mask in masks.items()
            },
            "peak_temperature_error_k": float(held_out[column].max() - truth.max()),
            "melt_pool_point_extents": melt_pool_point_extents(
                held_out, column, liquidus_k
            ),
        }
    result["residual_gp"] = regression_metrics(
        residual_truth, held_out["delta_T_pred"]
    )
    result["uncertainty_calibration"] = {
        "nominal_95_interval_coverage": float(
            np.mean(np.abs(residual_error.to_numpy()) <= 1.96 * std)
        ),
        "standardized_residual_rmse": float(
            np.sqrt(np.mean((residual_error.to_numpy() / std) ** 2))
        ),
        "mean_predictive_std_k": float(np.mean(std)),
    }
    result["ground_truth_melt_pool_point_extents"] = melt_pool_point_extents(
        held_out, "T_TCAM", liquidus_k
    )
    result["thresholds_k"] = {
        "ambient": BASE_TEMPERATURE_K,
        "hot_region": hot_threshold,
        "liquidus": liquidus_k,
    }
    return result


def write_parity_plot(held_out: pd.DataFrame, output_path: Path) -> None:
    truth = held_out["T_TCAM"]
    series = {
        "Raw ET": held_out["T_ET"],
        "Warped ET": held_out["T_warped_ET"],
        "Warped ET + Cartesian GPR": held_out["T_pred"],
    }
    lower = min(float(truth.min()), *(float(v.min()) for v in series.values()))
    upper = max(float(truth.max()), *(float(v.max()) for v in series.values()))
    figure, axis = plt.subplots(figsize=(7.4, 6.5))
    for (label, values), color in zip(
        series.items(), ("tab:blue", "tab:orange", "tab:green")
    ):
        axis.scatter(truth, values, s=8, alpha=0.22, label=label, color=color)
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.4)
    axis.set_xlabel("Held-out TCAM temperature (K)")
    axis.set_ylabel("Predicted temperature (K)")
    axis.set_title("Cartesian warped ET + residual GPR: held-out case")
    axis.grid(alpha=0.15)
    axis.legend()
    mae = mean_absolute_error(truth, held_out["T_pred"])
    rmse = mean_squared_error(truth, held_out["T_pred"]) ** 0.5
    axis.text(
        0.97,
        0.03,
        f"GPR MAE: {mae:.1f} K\nGPR RMSE: {rmse:.1f} K",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.35",
            "alpha": 0.92,
        },
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_residual_plot(held_out: pd.DataFrame, output_path: Path) -> None:
    actual = held_out["T_TCAM"] - held_out["T_warped_ET"]
    predicted = held_out["delta_T_pred"]
    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    points = axis.scatter(
        actual,
        predicted,
        c=held_out["T_warped_ET"],
        cmap="inferno",
        s=8,
        alpha=0.32,
    )
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.4)
    axis.set_xlabel("True residual, TCAM - warped ET (K)")
    axis.set_ylabel("Predicted residual (K)")
    axis.set_title("Cartesian warped-prior discrepancy on unseen P-V case")
    figure.colorbar(points, ax=axis, label="Warped ET temperature (K)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_comparison_vti(
    prediction: ETGridPrediction,
    held_out: pd.DataFrame,
    output_path: Path,
    case,
) -> int:
    """Write a post-freeze VTI with linearly interpolated TCAM and errors."""
    import pyvista as pv

    coordinates = held_out[["x", "y", "z"]].to_numpy(dtype=float)
    interpolator = LinearNDInterpolator(
        coordinates,
        held_out["T_TCAM"].to_numpy(dtype=float),
        fill_value=np.nan,
    )
    grid_coordinates = _cartesian_coordinates(prediction.axes_um)
    T_tcam = np.asarray(interpolator(grid_coordinates), dtype=float)
    valid = np.isfinite(T_tcam)
    shape = prediction.native_dimensions
    T_pred = prediction.arrays_k["T_pred"].ravel(order="C")
    error = np.full(len(T_tcam), np.nan, dtype=np.float32)
    error[valid] = (T_pred[valid] - T_tcam[valid]).astype(np.float32)

    spacing = tuple(float(np.diff(axis)[0]) for axis in prediction.axes_um)
    grid = pv.ImageData(
        dimensions=shape,
        spacing=spacing,
        origin=tuple(float(axis[0]) for axis in prediction.axes_um),
    )
    for name, values in prediction.arrays_k.items():
        grid.point_data[name] = values.ravel(order="F")
    grid.point_data["T_TCAM"] = T_tcam.reshape(shape, order="C").ravel(order="F")
    grid.point_data["T_pred_minus_TCAM"] = error.reshape(shape, order="C").ravel(order="F")
    grid.point_data["abs_error"] = np.abs(error).reshape(shape, order="C").ravel(order="F")
    grid.point_data["TCAM_valid_mask"] = (
        valid.astype(np.uint8).reshape(shape, order="C").ravel(order="F")
    )
    grid.field_data["case_id"] = np.asarray([case.case_id])
    grid.field_data["prediction_generated_before_TCAM"] = np.asarray(
        [1], dtype=np.uint8
    )
    grid.field_data["contains_TCAM_for_evaluation"] = np.asarray(
        [1], dtype=np.uint8
    )
    grid.save(output_path, binary=True)
    return int(np.sum(valid))


def case_record(case, held_out: bool = False) -> dict:
    record = {
        "case_id": case.case_id,
        "power_w": case.power_w,
        "velocity_m_s": case.velocity_m_s,
        "et_field": str(case.et_field),
    }
    key = "tcam_field_used_only_for_evaluation" if held_out else "tcam_field"
    record[key] = str(case.tcam_field)
    return record


def export_saved_model_prediction(args) -> dict:
    """Write a finer TCAM-blind grid from an already-fitted model bundle."""
    started = time.perf_counter()
    bundle = joblib.load(args.load_bundle)
    feature_columns = bundle.get("feature_columns")
    valid_feature_sets = {
        tuple(feature_columns_for(name)) for name in ("none", "r_yz", "r_3d")
    }
    if tuple(feature_columns or ()) not in valid_feature_sets:
        raise ValueError(f"Unsupported saved feature columns: {feature_columns!r}.")
    held = bundle["held_out_case"]
    et_field = Path(held["et_field"])
    held_et_case = ETPriorCase(
        case_id=held["case_id"],
        power_w=float(held["power_w"]),
        velocity_m_s=float(held["velocity_m_s"]),
        et_field=et_field,
        et_metadata=et_field.with_name("metadata.csv"),
    )
    if not held_et_case.et_field.is_file():
        raise FileNotFoundError(held_et_case.et_field)
    if not held_et_case.et_metadata.is_file():
        raise FileNotFoundError(held_et_case.et_metadata)

    prediction, _, held_warp = generate_coarse_prediction(
        held_et_case,
        args.material_data,
        bundle["warp_surface"],
        bundle["gpr"],
        bundle["scaler"],
        stride=args.prediction_grid_stride,
        batch_size=args.prediction_batch_size,
        feature_columns=feature_columns,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.prediction_output or (
        args.output_dir
        / f"heldout_prediction_on_stride{args.prediction_grid_stride}_et_grid.vti"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_et_grid_prediction_vti(prediction, output_path, held_et_case)
    elapsed = time.perf_counter() - started
    report = {
        "mode": "saved-model TCAM-blind prediction only",
        "model_bundle": str(args.load_bundle),
        "case_id": held_et_case.case_id,
        "source_et_field": str(held_et_case.et_field),
        "prediction_vti": str(output_path),
        "feature_columns": feature_columns,
        "predicted_warp": asdict(held_warp),
        "output_dimensions": prediction.native_dimensions,
        "output_point_count": int(np.prod(prediction.native_dimensions)),
        "output_spacing_um": [
            float(np.diff(axis)[0]) for axis in prediction.axes_um
        ],
        "prediction_grid_stride": prediction.gp_stride,
        "bounded_prediction_counts_on_output_grid": prediction.bound_counts,
        "TCAM_information_used": False,
        "runtime_seconds": elapsed,
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Wrote TCAM-blind saved-model prediction: {output_path}")
    print(
        f"Grid: {prediction.native_dimensions} "
        f"({int(np.prod(prediction.native_dimensions))} points)"
    )
    print(f"Prediction-only runtime: {elapsed:.1f} s")
    return report


def run_experiment(args) -> dict:
    started = time.perf_counter()
    np.random.seed(args.seed)
    feature_columns = feature_columns_for(args.radial_feature)
    cases = discover_paired_cases(args.et_root, args.tcam_root)
    training_keys = (
        args.training_cases
        if args.training_cases is not None
        else list(DEFAULT_TRAINING_CASES)
    )
    training_cases, held_out_case = resolve_case_split(
        cases, training_keys, args.held_out_case
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_case_map(training_cases, held_out_case, args.output_dir / "case_split.png")

    warp_started = time.perf_counter()
    training, warp_table, case_summary = build_warped_training_table(
        training_cases,
        material_data=args.material_data,
        points_per_case=args.points_per_case,
        warp_points_per_case=args.warp_points_per_case,
        warp_max_iterations=args.warp_max_iterations,
        seed=args.seed,
        feature_columns=feature_columns,
    )
    warp_surface = fit_warp_surface(warp_table)
    warp_cv = leave_one_case_out_warp_predictions(warp_table)
    warp_seconds = time.perf_counter() - warp_started

    gp_started = time.perf_counter()
    model, scaler = fit_residual_gpr(
        training,
        args.seed,
        args.n_restarts_optimizer,
        feature_columns=feature_columns,
    )
    gp_fit_seconds = time.perf_counter() - gp_started

    # TCAM-BLIND HELD-OUT STAGE. ETPriorCase contains no TCAM path.
    prediction_started = time.perf_counter()
    held_et_case = ETPriorCase.from_paired_case(held_out_case)
    prediction, physics, held_warp = generate_coarse_prediction(
        held_et_case,
        args.material_data,
        warp_surface,
        model,
        scaler,
        stride=args.prediction_grid_stride,
        batch_size=args.prediction_batch_size,
        feature_columns=feature_columns,
    )
    prediction_vti = args.output_dir / "heldout_prediction_on_coarse_et_grid.vti"
    write_et_grid_prediction_vti(prediction, prediction_vti, held_et_case)
    prediction_seconds = time.perf_counter() - prediction_started
    print(f"Frozen TCAM-blind prediction: {prediction_vti}")

    # EVALUATION-ONLY STAGE. This is the first held-out TCAM read.
    evaluation_started = time.perf_counter()
    held_out, _ = evaluate_frozen_prediction(
        held_out_case, args.material_data, prediction
    )
    metrics = calculate_metrics(held_out, physics.liquidus_k)
    comparison_vti = (
        args.output_dir / "heldout_prediction_vs_tcam_on_coarse_et_grid.vti"
    )
    comparison_valid_points = write_comparison_vti(
        prediction, held_out, comparison_vti, held_out_case
    )
    write_parity_plot(held_out, args.output_dir / "heldout_parity.png")
    write_residual_plot(held_out, args.output_dir / "heldout_residual_parity.png")
    held_out.to_csv(
        args.output_dir / "heldout_predictions_at_tcam_points.csv", index=False
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    training.to_csv(args.output_dir / "training_sample.csv", index=False)
    warp_table.to_csv(args.output_dir / "training_case_warps.csv", index=False)
    warp_cv.to_csv(args.output_dir / "warp_leave_one_case_out.csv", index=False)
    case_summary.to_csv(args.output_dir / "training_case_summary.csv", index=False)
    split = {
        "training_cases": [case_record(case) for case in training_cases],
        "held_out_case": case_record(held_out_case, held_out=True),
    }
    with (args.output_dir / "case_split.json").open("w", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)
        stream.write("\n")

    grid_report = {
        "source_et_field": str(held_out_case.et_field),
        "prediction_vti": str(prediction_vti),
        "comparison_vti_created_after_prediction_freeze": str(comparison_vti),
        "output_dimensions": prediction.native_dimensions,
        "output_point_count": int(np.prod(prediction.native_dimensions)),
        "output_spacing_um": [
            float(np.diff(axis)[0]) for axis in prediction.axes_um
        ],
        "comparison_tcam_valid_point_count": comparison_valid_points,
        "bounded_prediction_counts_on_output_grid": prediction.bound_counts,
        "TCAM_information_used_for_prediction": False,
    }
    with (args.output_dir / "heldout_et_grid_prediction.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(grid_report, stream, indent=2)
        stream.write("\n")

    bundle_path = args.output_dir / "warped_cartesian_residual_gpr.joblib"
    joblib.dump(
        {
            "gpr": model,
            "scaler": scaler,
            "warp_surface": warp_surface,
            "feature_columns": feature_columns,
            "training_cases": split["training_cases"],
            "held_out_case": split["held_out_case"],
            "held_out_warp": asdict(held_warp),
            "base_temperature_k": BASE_TEMPERATURE_K,
        },
        bundle_path,
    )

    timings = {
        "training_warp_fit_and_surface_seconds": warp_seconds,
        "residual_gp_fit_seconds": gp_fit_seconds,
        "held_out_blind_prediction_and_vti_seconds": prediction_seconds,
        "held_out_evaluation_and_outputs_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    report = {
        "model": "Cartesian warped ET prior plus Cartesian residual GPR",
        "prior": "Eagar-Tsai (ET)",
        "ground_truth": "TCAM",
        "held_out_case": held_out_case.case_id,
        "training_case_count": len(training_cases),
        "training_point_count": len(training),
        "feature_columns": feature_columns,
        "radial_feature": args.radial_feature,
        "excluded_feature_families": [
            "cylindrical coordinates",
            "angular coordinates",
            *(["radial distances"] if args.radial_feature == "none" else []),
        ],
        "warp_parameter_names": list(WARP_PARAMETER_NAMES),
        "warp_response_inputs": ["P", "V"],
        "warp_boundary_handling": (
            "Queries outside the ET support only by floating-point roundoff are "
            "clipped to the stored boundary; genuinely unsupported queries remain "
            "at ambient temperature."
        ),
        "held_out_predicted_warp": asdict(held_warp),
        "held_out_grid": grid_report,
        "metrics": metrics,
        "learned_kernel": str(model.kernel_),
        "timings": timings,
        "leakage_boundary": (
            "The complete warped-prior plus residual-GP prediction was generated "
            "and written on the held-out ET-domain grid before TCAM was opened. "
            "TCAM was subsequently used only for evaluation outputs."
        ),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nHeld-out full-field RMSE:")
    for name in ("raw_et", "warped_et", "warped_cartesian_gpr"):
        print(f"  {name}: {metrics[name]['full_field']['rmse_k']:.3f} K")
    print(f"Total runtime: {timings['total_seconds']:.1f} s")
    print(f"Wrote experiment outputs to {args.output_dir}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit Cartesian ET warps and a P,V,x,y,z,T_warped_ET residual "
            "GPR on 15 cases, then predict a TCAM-blind held-out ET grid."
        )
    )
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET_ROOT)
    parser.add_argument("--tcam-root", type=Path, default=DEFAULT_TCAM_ROOT)
    parser.add_argument("--material-data", type=Path, default=DEFAULT_MATERIAL_DATA)
    parser.add_argument(
        "--held-out-case",
        type=_parse_case_specification,
        default=DEFAULT_HELD_OUT_CASE,
        metavar="POWER,VELOCITY",
    )
    parser.add_argument(
        "--training-case",
        dest="training_cases",
        type=_parse_case_specification,
        action="append",
        metavar="POWER,VELOCITY",
    )
    parser.add_argument("--points-per-case", type=int, default=POINTS_PER_TRAINING_CASE)
    parser.add_argument(
        "--warp-points-per-case", type=int, default=WARP_POINTS_PER_CASE
    )
    parser.add_argument(
        "--warp-max-iterations",
        type=int,
        default=WARP_DIFFERENTIAL_EVOLUTION_ITERATIONS,
    )
    parser.add_argument(
        "--n-restarts-optimizer", type=int, default=N_RESTARTS_OPTIMIZER
    )
    parser.add_argument(
        "--prediction-grid-stride", type=int, default=ET_GRID_GP_STRIDE
    )
    parser.add_argument(
        "--prediction-batch-size", type=int, default=ET_GRID_GP_BATCH_SIZE
    )
    parser.add_argument(
        "--radial-feature",
        choices=("none", "r_yz", "r_3d"),
        default="none",
        help=(
            "Optional controlled radial GP input. r_yz=sqrt(y^2+z^2); "
            "r_3d=sqrt(x^2+y^2+z^2)."
        ),
    )
    parser.add_argument(
        "--load-bundle",
        type=Path,
        help=(
            "Skip training and generate a TCAM-blind grid from this saved "
            "warped_cartesian_residual_gpr.joblib bundle."
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        help="Optional VTI path for --load-bundle prediction-only mode.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = {
            "none": OUTPUT_DIR,
            "r_yz": RYZ_OUTPUT_DIR,
            "r_3d": R3D_OUTPUT_DIR,
        }[args.radial_feature]
    if args.points_per_case < 10 or args.warp_points_per_case < 10:
        parser.error("Training and warp point counts must each be at least 10.")
    if args.warp_max_iterations < 1:
        parser.error("--warp-max-iterations must be positive.")
    if args.n_restarts_optimizer < 0:
        parser.error("--n-restarts-optimizer cannot be negative.")
    if args.prediction_grid_stride < 1 or args.prediction_batch_size < 1:
        parser.error("Prediction stride and batch size must be positive.")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.load_bundle is None:
        run_experiment(parsed_args)
    else:
        export_saved_model_prediction(parsed_args)
