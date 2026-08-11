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
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from .GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        balanced_temperature_sample,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
    )
    from .et_temperature_field import (
        ETTemperatureField,
        load_et_temperature_field,
    )
except ImportError:
    from GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        balanced_temperature_sample,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
    )
    from et_temperature_field import ETTemperatureField, load_et_temperature_field


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ET_ROOT = REPO_ROOT / "beamer/figures/data/BU_ET"

DEFAULT_TCAM_ROOT = REPO_ROOT / "beamer/figures/data/BU_TCAM"

DEFAULT_MATERIAL_DATA = REPO_ROOT / "effective_cp_data.xlsx"

DEFAULT_HELD_OUT_CASE = (250.0, 0.5)

DEFAULT_TRAINING_CASES = (
    (50.0, 0.1),
    (50.0, 0.5),
    (50.0, 1.0),
    (50.0, 1.5),
    (150.0, 0.1),
    (150.0, 0.5),
    (150.0, 1.0),
    (150.0, 1.5),
    (250.0, 0.1),
    (250.0, 1.0),
    (250.0, 1.5),
    (350.0, 0.1),
    (350.0, 0.5),
    (350.0, 1.0),
    (350.0, 1.5),
)

EVAPORATION_SAMPLE_FRACTION = 0.25

EVAPORATION_MATCH_TOLERANCE_K = 10.0

TCAM_FILENAME_PATTERN = re.compile(
    r"^TCAM_cropped_alloy(?P<alloy>[^_]+)_"
    r"(?P<power>[-+0-9.eE]+)_(?P<velocity>[-+0-9.eE]+)\.vtu$"
)

@dataclass(frozen=True)
class PairedCase:
    alloy: str
    power_w: float
    velocity_m_s: float
    et_field: Path
    et_metadata: Path
    tcam_field: Path

    @property
    def process_key(self):
        return _process_key(self.power_w, self.velocity_m_s)

    @property
    def case_id(self):
        return (
            f"alloy{self.alloy}_"
            f"{self.power_w:g}W_{self.velocity_m_s:g}ms"
        )

def _process_key(power_w, velocity_m_s):
    return round(float(power_w), 12), round(float(velocity_m_s), 12)

def _format_alloy(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else f"{number:g}"

def _parse_case_specification(text):
    try:
        power, velocity = text.split(",", maxsplit=1)
        return _process_key(float(power), float(velocity))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"Case must be POWER,VELOCITY, received {text!r}."
        ) from error

def discover_paired_cases(et_root, tcam_root):
    """Discover and validate ET/TCAM pairs using metadata and filenames."""
    et_cases = {}
    for metadata_path in sorted(Path(et_root).glob("*/metadata.csv")):
        metadata = pd.read_csv(metadata_path).iloc[0]
        required = {"power_w", "velocity_m_s"}
        missing = required - set(metadata.index)
        if missing:
            raise ValueError(
                f"{metadata_path} is missing metadata fields: {missing}"
            )
        et_field = metadata_path.with_name("ET_temperature.vti")
        if not et_field.is_file():
            raise FileNotFoundError(f"Missing ET field: {et_field}")
        alloy_value = metadata.get("alloy", metadata.get("Alloy", 0))
        key = (
            _format_alloy(alloy_value),
            *_process_key(metadata["power_w"], metadata["velocity_m_s"]),
        )
        if key in et_cases:
            raise ValueError(f"Duplicate ET case for {key}.")
        et_cases[key] = (et_field, metadata_path)

    tcam_cases = {}
    for tcam_field in sorted(
        Path(tcam_root).glob("*/TCAM_cropped_alloy*.vtu")
    ):
        match = TCAM_FILENAME_PATTERN.match(tcam_field.name)
        if match is None:
            raise ValueError(
                f"Unrecognized cropped TCAM filename: {tcam_field}"
            )
        key = (
            match.group("alloy"),
            *_process_key(
                match.group("power"),
                match.group("velocity"),
            ),
        )
        if key in tcam_cases:
            raise ValueError(f"Duplicate TCAM case for {key}.")
        tcam_cases[key] = tcam_field

    missing_tcam = sorted(set(et_cases) - set(tcam_cases))
    missing_et = sorted(set(tcam_cases) - set(et_cases))
    if missing_tcam or missing_et:
        raise ValueError(
            "ET/TCAM case inventory does not match. "
            f"Missing TCAM: {missing_tcam}; missing ET: {missing_et}."
        )

    cases = {}
    for (alloy, power, velocity), (
        et_field,
        metadata_path,
    ) in et_cases.items():
        case = PairedCase(
            alloy=alloy,
            power_w=power,
            velocity_m_s=velocity,
            et_field=et_field,
            et_metadata=metadata_path,
            tcam_field=tcam_cases[(alloy, power, velocity)],
        )
        if case.process_key in cases:
            raise ValueError(
                "Multiple alloys share a process condition; specify alloy "
                "selection before using this script."
            )
        cases[case.process_key] = case
    return cases

def resolve_case_split(cases, training_keys, held_out_key):
    """Resolve a disjoint set of complete training cases and one target."""
    if held_out_key not in cases:
        raise ValueError(
            f"Held-out case {held_out_key} is unavailable. "
            f"Available cases: {sorted(cases)}"
        )
    if held_out_key in training_keys:
        raise ValueError("The held-out case cannot be a training case.")
    if len(set(training_keys)) != len(training_keys):
        raise ValueError("Training cases contain duplicates.")

    missing = [key for key in training_keys if key not in cases]
    if missing:
        raise ValueError(
            f"Training cases are unavailable: {missing}. "
            f"Available cases: {sorted(cases)}"
        )
    return [cases[key] for key in training_keys], cases[held_out_key]

def sample_training_case(dataframe, points_per_case, seed):
    """Sample one case while deliberately retaining evaporation-plateau data."""
    evaporation_threshold = (
        float(dataframe["Tevap"].iloc[0])
        - EVAPORATION_MATCH_TOLERANCE_K
    )
    evaporation = dataframe[
        dataframe["T_TCAM"] >= evaporation_threshold
    ]
    other = dataframe.drop(index=evaporation.index)

    evaporation_budget = min(
        len(evaporation),
        int(round(points_per_case * EVAPORATION_SAMPLE_FRACTION)),
    )
    if evaporation_budget:
        evaporation_sample = evaporation.sample(
            n=evaporation_budget,
            random_state=seed,
        )
    else:
        evaporation_sample = evaporation.iloc[0:0].copy()

    other_budget = points_per_case - len(evaporation_sample)
    other_sample = balanced_temperature_sample(
        other,
        max_points=other_budget,
        seed=seed,
    )
    sampled = pd.concat(
        [evaporation_sample, other_sample],
        ignore_index=True,
    )
    if len(sampled) > points_per_case:
        raise AssertionError("Per-case sampler exceeded its point budget.")
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def write_case_map(training_cases, held_out_case, output_path):
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    axis.scatter(
        [case.velocity_m_s for case in training_cases],
        [case.power_w for case in training_cases],
        s=90,
        marker="o",
        label="Training case",
    )
    axis.scatter(
        [held_out_case.velocity_m_s],
        [held_out_case.power_w],
        s=140,
        marker="X",
        color="tab:red",
        label="Held-out case",
    )
    axis.set_xlabel("Velocity (m/s)", fontsize=16)
    axis.set_ylabel("Power (W)", fontsize=16)
    axis.set_title("Multi-case GPR split", fontsize=18)
    axis.set_xticks([0.1, 0.5, 1.0, 1.5])
    axis.set_yticks([50, 150, 250, 350])
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

WARP_POINTS_PER_CASE = 900

WARP_DIFFERENTIAL_EVOLUTION_ITERATIONS = 35

ET_GRID_GP_BATCH_SIZE = 2_000

ET_GRID_GP_STRIDE = 4

WARP_PARAMETER_NAMES = (
    "amplitude",
    "rear_scale",
    "front_scale",
    "width_scale",
    "depth_scale",
    "x_shift_over_sigma",
)

WARP_BOUNDS = np.asarray(
    [
        (0.20, 5.00),
        (0.30, 6.00),
        (0.30, 6.00),
        (0.30, 6.00),
        (0.30, 8.00),
        (-4.00, 4.00),
    ],
    dtype=float,
)

LOG_WARP_PARAMETER_INDICES = (0, 1, 2, 3, 4)

@dataclass(frozen=True)
class CasePhysics:
    """ET-known quantities used for nondimensionalization and constraints."""

    sigma_um: float
    alpha_m2_s: float
    peclet: float
    normalized_power: float
    temperature_scale_k: float
    liquidus_k: float
    evaporation_k: float

@dataclass(frozen=True)
class ETPriorCase:
    """Held-out process information with no TCAM path or TCAM-derived data."""

    case_id: str
    power_w: float
    velocity_m_s: float
    et_field: Path
    et_metadata: Path

    @classmethod
    def from_paired_case(cls, case: PairedCase) -> "ETPriorCase":
        return cls(
            case_id=case.case_id,
            power_w=case.power_w,
            velocity_m_s=case.velocity_m_s,
            et_field=case.et_field,
            et_metadata=case.et_metadata,
        )

@dataclass(frozen=True)
class WarpParameters:
    """Six-parameter asymmetric geometric warp of an ET field."""

    amplitude: float = 1.0
    rear_scale: float = 1.0
    front_scale: float = 1.0
    width_scale: float = 1.0
    depth_scale: float = 1.0
    x_shift_over_sigma: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in WARP_PARAMETER_NAMES],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "WarpParameters":
        values = np.asarray(values, dtype=float)
        if values.shape != (len(WARP_PARAMETER_NAMES),):
            raise ValueError(
                f"Warp vector must have shape ({len(WARP_PARAMETER_NAMES)},), "
                f"received {values.shape}."
            )
        return cls(**dict(zip(WARP_PARAMETER_NAMES, values)))

@dataclass
class ETGridPrediction:
    """A complete TCAM-blind prediction on a Cartesian ET-domain grid."""

    axes_um: tuple[np.ndarray, np.ndarray, np.ndarray]
    arrays_k: dict[str, np.ndarray]
    gp_support_dimensions: tuple[int, int, int]
    native_dimensions: tuple[int, int, int]
    gp_stride: int
    bound_counts: dict[str, int]

    def interpolate(self, coordinates_um: np.ndarray, column: str) -> np.ndarray:
        if column not in self.arrays_k:
            raise KeyError(f"Prediction grid has no array {column!r}.")
        return RegularGridInterpolator(
            self.axes_um,
            self.arrays_k[column],
            method="linear",
            bounds_error=True,
        )(np.asarray(coordinates_um, dtype=float))

def load_case_physics(metadata_path: Path, material_data: Path) -> CasePhysics:
    """Load only ET/material metadata; no TCAM information is accessed."""
    metadata = pd.read_csv(metadata_path).iloc[0]
    inputs = load_process_inputs(metadata_path, material_data)

    beam_diameter_m = float(metadata["beam_diameter_m"])
    if beam_diameter_m <= 0.0:
        raise ValueError(f"beam_diameter_m must be positive in {metadata_path}.")

    # et_melt_pool_script.BeamInputs uses sigma=sqrt(2)*(diameter/2).
    sigma_m = beam_diameter_m / np.sqrt(2.0)
    alpha_m2_s = inputs["k"] / (inputs["rho"] * inputs["Cp"])
    if alpha_m2_s <= 0.0:
        raise ValueError(f"Thermal diffusivity must be positive in {metadata_path}.")

    temperature_scale_k = inputs["A"] * inputs["P"] / (inputs["k"] * sigma_m)
    liquidus_rise_k = inputs["Tliq"] - BASE_TEMPERATURE_K
    if temperature_scale_k <= 0.0 or liquidus_rise_k <= 0.0:
        raise ValueError(f"Invalid temperature scale in {metadata_path}.")

    return CasePhysics(
        sigma_um=sigma_m * 1.0e6,
        alpha_m2_s=alpha_m2_s,
        peclet=inputs["V"] * sigma_m / (2.0 * alpha_m2_s),
        normalized_power=temperature_scale_k / liquidus_rise_k,
        temperature_scale_k=temperature_scale_k,
        liquidus_k=inputs["Tliq"],
        evaporation_k=inputs["Tevap"],
    )

def warp_query_coordinates(
    coordinates_um: np.ndarray,
    parameters: WarpParameters,
    sigma_um: float,
) -> np.ndarray:
    """Map target coordinates back into the Cartesian ET prior frame."""
    coordinates = np.asarray(coordinates_um, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates_um must have shape (n, 3).")
    if sigma_um <= 0.0:
        raise ValueError("sigma_um must be positive.")

    centered_x = coordinates[:, 0] - parameters.x_shift_over_sigma * sigma_um
    x_scale = np.where(
        centered_x < 0.0,
        parameters.rear_scale,
        parameters.front_scale,
    )
    return np.column_stack(
        [
            centered_x / x_scale,
            coordinates[:, 1] / parameters.width_scale,
            coordinates[:, 2] / parameters.depth_scale,
        ]
    )

def evaluate_warped_et(
    et_field: ETTemperatureField,
    coordinates_um: np.ndarray,
    parameters: WarpParameters,
    sigma_um: float,
    base_temperature_k: float = BASE_TEMPERATURE_K,
) -> np.ndarray:
    """Evaluate the Cartesian ET interpolant at geometrically warped points."""
    query = warp_query_coordinates(coordinates_um, parameters, sigma_um)
    lower = np.asarray([axis[0] for axis in et_field.axes])
    upper = np.asarray([axis[-1] for axis in et_field.axes])
    # VTK axes that conceptually end at zero can instead end at a value such
    # as -2.84e-14.  A depth stretch maps that endpoint infinitesimally closer
    # to zero, and a strict comparison would incorrectly discard the complete
    # surface plane.  Admit only roundoff-scale excursions, then clamp those
    # queries to the exact stored endpoints.  Genuinely unsupported queries
    # (for example hundreds of micrometers beyond the x range) remain outside.
    spacing = np.asarray(
        [float(np.min(np.diff(axis))) for axis in et_field.axes],
        dtype=float,
    )
    tolerance = np.maximum(
        spacing * 1.0e-9,
        64.0
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper))),
    )
    inside = np.all(
        (query >= lower - tolerance) & (query <= upper + tolerance),
        axis=1,
    )

    prior = np.full(len(query), base_temperature_k, dtype=float)
    if inside.any():
        safe_query = np.clip(query[inside], lower, upper)
        prior[inside] = et_field.interpolator(bounds_error=True)(safe_query)
    prior_rise = np.maximum(prior - base_temperature_k, 0.0)
    return base_temperature_k + parameters.amplitude * prior_rise

def _knn_gradient_proxy(dataframe: pd.DataFrame, neighbors: int = 8) -> np.ndarray:
    """Estimate |grad T| on an unstructured point cloud for warp sampling."""
    coordinates = dataframe[["x", "y", "z"]].to_numpy(dtype=float)
    temperature = dataframe["T_TCAM"].to_numpy(dtype=float)
    if len(dataframe) < 2:
        return np.zeros(len(dataframe), dtype=float)

    k = min(neighbors + 1, len(dataframe))
    distances, indices = cKDTree(coordinates).query(coordinates, k=k)
    if k == 1:
        return np.zeros(len(dataframe), dtype=float)
    distances = np.asarray(distances)[:, 1:]
    indices = np.asarray(indices)[:, 1:]
    slopes = np.abs(temperature[indices] - temperature[:, None]) / np.maximum(
        distances,
        1.0e-9,
    )
    return np.nanmax(slopes, axis=1)

def gradient_weighted_sample(
    dataframe: pd.DataFrame,
    max_points: int,
    seed: int,
    top_surface_fraction: float = 0.15,
) -> pd.DataFrame:
    """Sample steep TCAM regions, with explicit top and far-field support."""
    if max_points < 10:
        raise ValueError("max_points must be at least 10.")
    if len(dataframe) <= max_points:
        return dataframe.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    gradient = _knn_gradient_proxy(dataframe)
    positive = gradient[gradient > 0.0]
    g97 = float(np.percentile(positive, 97.0)) if positive.size else 1.0
    softened = np.minimum(gradient / max(g97, 1.0e-12), 1.0) ** 0.65
    probability = 0.04 + softened

    selected: list[int] = []
    top_z = float(dataframe["z"].max())
    z_tolerance = max(1.0e-6, 0.01 * float(dataframe["z"].max() - dataframe["z"].min()))
    top_indices = np.flatnonzero(dataframe["z"].to_numpy() >= top_z - z_tolerance)
    top_budget = min(len(top_indices), int(round(max_points * top_surface_fraction)))
    if top_budget:
        top_weights = probability[top_indices]
        top_weights /= top_weights.sum()
        selected.extend(
            rng.choice(top_indices, size=top_budget, replace=False, p=top_weights).tolist()
        )

    # Reserve cold, remote observations so the GP retains an ambient far field.
    coordinate_span = dataframe[["x", "y", "z"]].max() - dataframe[["x", "y", "z"]].min()
    normalized_edge_distance = np.column_stack(
        [
            (dataframe[name] - dataframe[name].min()) / max(float(coordinate_span[name]), 1.0e-12)
            for name in ("x", "y", "z")
        ]
    )
    boundary = np.any(
        (normalized_edge_distance <= 0.02) | (normalized_edge_distance >= 0.98),
        axis=1,
    )
    cold = dataframe["T_TCAM"].to_numpy() <= BASE_TEMPERATURE_K + 25.0
    far_indices = np.flatnonzero(boundary & cold & ~np.isin(np.arange(len(dataframe)), selected))
    far_budget = min(len(far_indices), int(round(max_points * 0.10)))
    if far_budget:
        selected.extend(rng.choice(far_indices, size=far_budget, replace=False).tolist())

    remaining_budget = max_points - len(selected)
    available = np.flatnonzero(~np.isin(np.arange(len(dataframe)), selected))
    available_weights = probability[available]
    available_weights /= available_weights.sum()
    selected.extend(
        rng.choice(
            available,
            size=min(remaining_budget, len(available)),
            replace=False,
            p=available_weights,
        ).tolist()
    )
    sampled = dataframe.iloc[selected].copy().reset_index(drop=True)
    sampled["gradient_proxy_k_per_um"] = gradient[np.asarray(selected)]
    return sampled

def _warp_objective(
    values: np.ndarray,
    et_field: ETTemperatureField,
    coordinates_um: np.ndarray,
    truth_k: np.ndarray,
    physics: CasePhysics,
) -> float:
    prediction = evaluate_warped_et(
        et_field,
        coordinates_um,
        WarpParameters.from_array(values),
        physics.sigma_um,
    )
    # Normalization gives every alloy/process case a comparable objective.
    error = (prediction - truth_k) / max(
        physics.liquidus_k - BASE_TEMPERATURE_K,
        1.0,
    )
    return float(np.mean(error**2))

def fit_case_warp(
    dataframe: pd.DataFrame,
    et_field: ETTemperatureField,
    physics: CasePhysics,
    max_points: int,
    seed: int,
    max_iterations: int,
) -> tuple[WarpParameters, dict, pd.DataFrame]:
    """Fit one training-case warp and reject it if full-case RMSE worsens."""
    sample = gradient_weighted_sample(dataframe, max_points=max_points, seed=seed)
    sample_coordinates = sample[["x", "y", "z"]].to_numpy(dtype=float)
    sample_truth = sample["T_TCAM"].to_numpy(dtype=float)
    objective_args = (et_field, sample_coordinates, sample_truth, physics)

    result = differential_evolution(
        _warp_objective,
        bounds=[tuple(row) for row in WARP_BOUNDS],
        args=objective_args,
        seed=seed,
        maxiter=max_iterations,
        popsize=8,
        tol=2.0e-4,
        polish=True,
        updating="immediate",
        workers=1,
    )
    fitted = WarpParameters.from_array(np.clip(result.x, WARP_BOUNDS[:, 0], WARP_BOUNDS[:, 1]))

    full_coordinates = dataframe[["x", "y", "z"]].to_numpy(dtype=float)
    truth = dataframe["T_TCAM"].to_numpy(dtype=float)
    identity_prediction = evaluate_warped_et(
        et_field,
        full_coordinates,
        WarpParameters(),
        physics.sigma_um,
    )
    fitted_prediction = evaluate_warped_et(
        et_field,
        full_coordinates,
        fitted,
        physics.sigma_um,
    )
    identity_rmse = float(mean_squared_error(truth, identity_prediction) ** 0.5)
    fitted_rmse = float(mean_squared_error(truth, fitted_prediction) ** 0.5)

    # A sparse optimizer can occasionally overfit sharp gradients.  Retaining
    # identity is a training-only safeguard; the holdout is never consulted.
    accepted = fitted_rmse <= identity_rmse
    selected = fitted if accepted else WarpParameters()
    selected_rmse = min(identity_rmse, fitted_rmse)
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev),
        "warp_sample_points": len(sample),
        "identity_full_rmse_k": identity_rmse,
        "fitted_full_rmse_k": fitted_rmse,
        "selected_full_rmse_k": selected_rmse,
        "fitted_warp_accepted": accepted,
        "full_rmse_improvement_k": identity_rmse - selected_rmse,
    }
    return selected, diagnostics, sample

def write_et_grid_prediction_vti(
    prediction: ETGridPrediction,
    output_path: Path,
    case: ETPriorCase,
) -> None:
    """Write the frozen Cartesian field before TCAM is revealed."""
    import pyvista as pv

    axes = prediction.axes_um
    spacing = tuple(float(np.diff(axis)[0]) for axis in axes)
    grid = pv.ImageData(
        dimensions=prediction.native_dimensions,
        spacing=spacing,
        origin=tuple(float(axis[0]) for axis in axes),
    )
    for name, values in prediction.arrays_k.items():
        grid.point_data[name] = values.ravel(order="F")
    grid.field_data["case_id"] = np.asarray([case.case_id])
    grid.field_data["power_w"] = np.asarray([case.power_w])
    grid.field_data["velocity_m_s"] = np.asarray([case.velocity_m_s])
    grid.field_data["gp_support_stride"] = np.asarray([prediction.gp_stride])
    grid.field_data["TCAM_blind_prediction"] = np.asarray([1], dtype=np.uint8)
    grid.save(output_path, binary=True)

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
RECOMMENDED_OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/multi_warped_gpr"
    / "heldout_250W_0.5ms_15cases"
)

BASE_FEATURE_COLUMNS = ["P", "V", "x", "y", "z", "T_warped_ET"]
# Backward-compatible name for the original controlled six-input experiment.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
POINTS_PER_TRAINING_CASE = 500
N_RESTARTS_OPTIMIZER = 3
WARP_SURFACE_RIDGE_ALPHA = 2.0e-2
KERNEL_CHOICES = ("rbf", "matern32", "matern52")
REAR_LIQUIDUS_BAND_K = 300.0
REAR_LIQUIDUS_TIP_WINDOW_UM = 25.0


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


def sample_warped_training_case(
    dataframe: pd.DataFrame,
    points_per_case: int,
    seed: int,
    rear_liquidus_points: int,
    rear_liquidus_band_k: float = REAR_LIQUIDUS_BAND_K,
) -> pd.DataFrame:
    """Reserve part of a case's fixed sample budget for its rear melt-pool tip."""
    if rear_liquidus_points < 0:
        raise ValueError("rear_liquidus_points cannot be negative.")
    if rear_liquidus_band_k <= 0.0:
        raise ValueError("rear_liquidus_band_k must be positive.")
    if rear_liquidus_points == 0:
        sampled = sample_training_case(dataframe, points_per_case, seed)
        sampled["rear_liquidus_sample"] = False
        return sampled

    required = {"x", "T_TCAM", "T_warped_ET", "Tliq"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Rear-liquidus sampling requires missing columns: {sorted(missing)}"
        )

    row_id_column = "_rear_liquidus_row_id"
    working = dataframe.copy()
    working[row_id_column] = np.arange(len(working), dtype=np.int64)
    liquidus_k = float(working["Tliq"].iloc[0])
    truth = working["T_TCAM"].to_numpy(dtype=float)
    warped = working["T_warped_ET"].to_numpy(dtype=float)
    x_coordinate = working["x"].to_numpy(dtype=float)
    rear = x_coordinate < 0.0
    distance = np.abs(truth - liquidus_k)
    near_liquidus = distance <= rear_liquidus_band_k
    warped_false_negative = (truth >= liquidus_k) & (warped < liquidus_k)
    molten_truth = truth >= liquidus_k
    if molten_truth.any():
        rear_tip_x = float(np.min(x_coordinate[molten_truth]))
        rear_tip_window = (
            x_coordinate <= rear_tip_x + REAR_LIQUIDUS_TIP_WINDOW_UM
        )
    else:
        rear_tip_window = np.zeros(len(working), dtype=bool)
    candidate_mask = (
        rear & rear_tip_window & (near_liquidus | warped_false_negative)
    )
    candidates = working.loc[candidate_mask].copy()

    dedicated_budget = min(
        rear_liquidus_points,
        points_per_case,
        len(candidates),
    )
    if dedicated_budget:
        candidate_distance = np.abs(
            candidates["T_TCAM"].to_numpy(dtype=float) - liquidus_k
        )
        candidate_false_negative = (
            (candidates["T_TCAM"].to_numpy(dtype=float) >= liquidus_k)
            & (candidates["T_warped_ET"].to_numpy(dtype=float) < liquidus_k)
        )
        weights = np.exp(-candidate_distance / rear_liquidus_band_k)
        weights *= np.where(candidate_false_negative, 3.0, 1.0)
        # Extremely hot false negatives can be many exponential scales away
        # from liquidus. Keep their priority negligible but nonzero so pandas
        # can still draw the requested number without replacement when the
        # near-liquidus subset is smaller than the dedicated budget.
        weights = np.maximum(weights, 1.0e-12)
        # Gumbel top-k performs weighted sampling without replacement in log
        # space and remains stable across the very large thermal weight range.
        rng = np.random.default_rng(seed)
        selection_key = np.log(weights) + rng.gumbel(size=len(weights))
        selected_positions = np.argpartition(
            selection_key,
            -dedicated_budget,
        )[-dedicated_budget:]
        dedicated = candidates.iloc[selected_positions].copy()
    else:
        dedicated = candidates.iloc[0:0].copy()
    dedicated["rear_liquidus_sample"] = True

    remaining_budget = points_per_case - len(dedicated)
    remaining_pool = working.drop(index=dedicated.index)
    # The inherited sampler rounds five overlapping stratum fractions. For
    # some small or reduced budgets, those rounded targets can sum to one more
    # than requested and trigger its budget assertion. Back off only as much
    # as required, then fill the exact remainder from unused rows.
    general_target = remaining_budget
    while True:
        try:
            general = sample_training_case(
                remaining_pool,
                points_per_case=general_target,
                seed=seed,
            )
            break
        except AssertionError:
            general_target -= 1
            if general_target < 1:
                raise
    fill_count = remaining_budget - len(general)
    if fill_count:
        used_ids = set(general[row_id_column].to_numpy(dtype=np.int64))
        leftover = remaining_pool[
            ~remaining_pool[row_id_column].isin(used_ids)
        ]
        extra = leftover.sample(
            n=min(fill_count, len(leftover)),
            random_state=seed + 104729,
        )
        general = pd.concat([general, extra], ignore_index=True)
    general["rear_liquidus_sample"] = False
    sampled = pd.concat([dedicated, general], ignore_index=True)
    if len(sampled) != points_per_case:
        raise AssertionError(
            "Rear-liquidus sampler did not preserve its exact point budget."
        )
    sampled = sampled.drop(columns=[row_id_column])
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_residual_kernel(n_features: int, kernel_name: str):
    """Build a controlled anisotropic residual-GP kernel."""
    if kernel_name == "rbf":
        return make_kernel(n_features)
    if kernel_name in {"matern32", "matern52"}:
        nu = 1.5 if kernel_name == "matern32" else 2.5
        return (
            ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=(0.05, 1.0e3),
                nu=nu,
            )
            + WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1.0e-6, 1.0e3),
            )
        )
    raise ValueError(f"Unknown residual kernel: {kernel_name!r}")


def default_output_dir(
    radial_feature: str,
    kernel_name: str,
    rear_liquidus_points: int = 0,
) -> Path:
    """Keep every controlled feature/kernel ablation in its own directory."""
    if (
        radial_feature == "r_3d"
        and kernel_name == "matern52"
        and rear_liquidus_points <= 0
    ):
        return RECOMMENDED_OUTPUT_DIR
    if kernel_name == "rbf":
        output = {
            "none": OUTPUT_DIR,
            "r_yz": RYZ_OUTPUT_DIR,
            "r_3d": R3D_OUTPUT_DIR,
        }[radial_feature]
    else:
        radial_label = {
            "none": "",
            "r_yz": "_ryz",
            "r_3d": "_r3d",
        }[radial_feature]
        output = (
            REPO_ROOT
            / f"beamer/figures/bayesian/warped_cartesian{radial_label}_{kernel_name}_gpr"
            / "heldout_250W_0.5ms_15trainingcases"
        )
    if rear_liquidus_points <= 0:
        return output
    experiment_name = output.parent.name.removesuffix("_gpr")
    return (
        output.parent.parent
        / f"{experiment_name}_rear_tip{rear_liquidus_points}_gpr"
        / output.name
    )


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
    rear_liquidus_points_per_case: int,
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
        sampled = sample_warped_training_case(
            dataframe,
            points_per_case=points_per_case,
            seed=seed + index,
            rear_liquidus_points=rear_liquidus_points_per_case,
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
                "sampled_rear_liquidus_points": int(
                    sampled["rear_liquidus_sample"].sum()
                ),
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
    kernel_name: str,
):
    missing = set(feature_columns + ["delta_T_warped"]) - set(training.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {missing}")
    X = training[feature_columns].to_numpy(dtype=float)
    target = training["delta_T_warped"].to_numpy(dtype=float)
    scaler = fit_feature_scaler(X, feature_columns)
    model = GaussianProcessRegressor(
        kernel=make_residual_kernel(len(feature_columns), kernel_name),
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        optimizer="fmin_l_bfgs_b",
        random_state=seed,
    )
    print(
        f"Fitting Cartesian warped-prior {kernel_name} GPR on {len(training)} points "
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
    axis.set_xlabel("Held-out TCAM temperature (K)", fontsize=16)
    axis.set_ylabel("Predicted temperature (K)", fontsize=16)
    axis.set_title(
        "Cartesian warped ET + residual GPR: held-out case", fontsize=18
    )
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(alpha=0.15)
    axis.legend(fontsize=14)
    mae = mean_absolute_error(truth, held_out["T_pred"])
    rmse = mean_squared_error(truth, held_out["T_pred"]) ** 0.5
    axis.text(
        0.97,
        0.03,
        f"GPR MAE: {mae:.1f} K\nGPR RMSE: {rmse:.1f} K",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=16,
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
    et_field = args.prediction_et_field or Path(held["et_field"])
    power_w = (
        float(args.prediction_power_w)
        if args.prediction_power_w is not None
        else float(held["power_w"])
    )
    velocity_m_s = (
        float(args.prediction_velocity_m_s)
        if args.prediction_velocity_m_s is not None
        else float(held["velocity_m_s"])
    )
    held_et_case = ETPriorCase(
        case_id=f"prediction_{power_w:g}W_{velocity_m_s:g}ms",
        power_w=power_w,
        velocity_m_s=velocity_m_s,
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
        rear_liquidus_points_per_case=args.rear_liquidus_points_per_case,
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
        kernel_name=args.kernel,
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
            "kernel_name": args.kernel,
            "rear_liquidus_points_per_case": args.rear_liquidus_points_per_case,
            "rear_liquidus_band_k": REAR_LIQUIDUS_BAND_K,
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
        "kernel_family": args.kernel,
        "rear_liquidus_sampling": {
            "requested_points_per_case": args.rear_liquidus_points_per_case,
            "temperature_band_k": REAR_LIQUIDUS_BAND_K,
            "rear_tip_window_um": REAR_LIQUIDUS_TIP_WINDOW_UM,
            "candidate_definition": (
                "Within the first rear_tip_window_um ahead of the minimum-x "
                "TCAM-liquidus point, with x < 0, and either "
                "abs(T_TCAM - Tliq) <= band or T_TCAM >= Tliq while "
                "T_warped_ET < Tliq"
            ),
            "held_out_TCAM_used": False,
        },
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
        "--kernel",
        choices=KERNEL_CHOICES,
        default="rbf",
        help=(
            "Residual-GP covariance family. The Matérn choices change only "
            "the base kernel smoothness; feature scaling, bounds, white noise, "
            "and workflow remain the same."
        ),
    )
    parser.add_argument(
        "--rear-liquidus-points-per-case",
        type=int,
        default=0,
        help=(
            "Reserve this many points from each fixed per-case GP budget for "
            "the training-only TCAM rear-tip liquidus neighborhood and "
            "warped-prior false negatives."
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
    parser.add_argument(
        "--prediction-et-field",
        type=Path,
        help="Optional arbitrary ET VTI for --load-bundle prediction-only mode.",
    )
    parser.add_argument(
        "--prediction-power-w",
        type=float,
        help="Power associated with --prediction-et-field.",
    )
    parser.add_argument(
        "--prediction-velocity-m-s",
        type=float,
        help="Velocity associated with --prediction-et-field.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = default_output_dir(
            args.radial_feature,
            args.kernel,
            args.rear_liquidus_points_per_case,
        )
    if args.points_per_case < 10 or args.warp_points_per_case < 10:
        parser.error("Training and warp point counts must each be at least 10.")
    if args.warp_max_iterations < 1:
        parser.error("--warp-max-iterations must be positive.")
    if args.n_restarts_optimizer < 0:
        parser.error("--n-restarts-optimizer cannot be negative.")
    if not 0 <= args.rear_liquidus_points_per_case < args.points_per_case:
        parser.error(
            "--rear-liquidus-points-per-case must be nonnegative and smaller "
            "than --points-per-case."
        )
    if args.prediction_grid_stride < 1 or args.prediction_batch_size < 1:
        parser.error("Prediction stride and batch size must be positive.")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.load_bundle is None:
        run_experiment(parsed_args)
    else:
        export_saved_model_prediction(parsed_args)
