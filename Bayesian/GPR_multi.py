#!/usr/bin/env python3
"""Train a residual GPR on multiple P-V cases and evaluate one held-out case.

The held-out TCAM field is not loaded until after model fitting. Its ET field
and process/material inputs are available to the model, while its TCAM
temperatures are used only for final whole-case evaluation.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        add_physics_features,
        add_process_inputs,
        balanced_temperature_sample,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
        set_corrected_temperature,
    )
except ImportError:
    from GPR import (
        BASE_TEMPERATURE_K,
        RANDOM_SEED,
        add_physics_features,
        add_process_inputs,
        balanced_temperature_sample,
        fit_feature_scaler,
        load_and_merge_fields,
        load_process_inputs,
        make_kernel,
        predict_in_batches,
        set_corrected_temperature,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ET_ROOT = REPO_ROOT / "beamer/figures/data/BU_ET"
DEFAULT_TCAM_ROOT = REPO_ROOT / "beamer/figures/data/BU_TCAM"
DEFAULT_MATERIAL_DATA = REPO_ROOT / "effective_cp_data.xlsx"
OUTPUT_DIR = (
    REPO_ROOT
    / "beamer/figures/bayesian/multicase_gpr"
    / "heldout_250W_0.5ms_15trainingcases"
)

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

POINTS_PER_TRAINING_CASE = 500
EVAPORATION_SAMPLE_FRACTION = 0.25
EVAPORATION_MATCH_TOLERANCE_K = 10.0
PREDICTION_BATCH_SIZE = 50_000
N_RESTARTS_OPTIMIZER = 1

# Constant Alloy 0 quantities are intentionally omitted. They do not provide
# information to the kernel until multiple compositions are trained together.
FEATURE_COLUMNS = [
    "P",
    "V",
    "x",
    "y",
    "z",
    "T_ET",
    "r_xy",
    "r_3d",
]

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


def load_case_dataframe(case, material_data):
    """Match one ET/TCAM pair and construct residual-model features."""
    dataframe = load_and_merge_fields(case.et_field, case.tcam_field)
    inputs = load_process_inputs(case.et_metadata, material_data)
    if not np.isclose(inputs["P"], case.power_w) or not np.isclose(
        inputs["V"],
        case.velocity_m_s,
    ):
        raise ValueError(
            f"Process metadata does not match discovered case {case.case_id}."
        )

    dataframe = add_process_inputs(dataframe, inputs)
    dataframe = add_physics_features(dataframe)
    dataframe["delta_T"] = dataframe["T_TCAM"] - dataframe["T_ET"]
    dataframe["case_id"] = case.case_id
    return dataframe, inputs


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


def build_training_table(cases, material_data, points_per_case, seed):
    sampled_cases = []
    case_summary = []
    for index, case in enumerate(cases):
        print(f"Loading training case {index + 1}/{len(cases)}: {case.case_id}")
        full_case, inputs = load_case_dataframe(case, material_data)
        sampled = sample_training_case(
            full_case,
            points_per_case=points_per_case,
            seed=seed + index,
        )
        evaporation_threshold = (
            inputs["Tevap"] - EVAPORATION_MATCH_TOLERANCE_K
        )
        plateau_count = int(
            (sampled["T_TCAM"] >= evaporation_threshold).sum()
        )
        sampled_cases.append(sampled)
        case_summary.append(
            {
                "case_id": case.case_id,
                "power_w": case.power_w,
                "velocity_m_s": case.velocity_m_s,
                "available_points": len(full_case),
                "sampled_points": len(sampled),
                "sampled_evaporation_points": plateau_count,
            }
        )
        print(
            f"Sampled {len(sampled)} / {len(full_case)} points "
            f"({plateau_count} near evaporation ceiling)."
        )

    training = pd.concat(sampled_cases, ignore_index=True)
    training = training.sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )
    return training, pd.DataFrame(case_summary)


def fit_residual_gpr(training, seed):
    missing = set(FEATURE_COLUMNS + ["delta_T"]) - set(training.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {missing}")

    features = training[FEATURE_COLUMNS].to_numpy(dtype=float)
    target = training["delta_T"].to_numpy(dtype=float)
    scaler = fit_feature_scaler(features, FEATURE_COLUMNS)
    scaled_features = scaler.transform(features)
    model = GaussianProcessRegressor(
        kernel=make_kernel(len(FEATURE_COLUMNS)),
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        optimizer="fmin_l_bfgs_b",
        random_state=seed,
    )
    print(f"Fitting GPR on {len(training)} points from complete cases...")
    model.fit(scaled_features, target)
    print(f"Learned kernel: {model.kernel_}")
    return model, scaler


def evaluate_held_out_case(
    case,
    material_data,
    model,
    scaler,
):
    """
    Reveal the held-out TCAM field only after fitting and evaluate the model.

    Loading it here cannot influence training, feature scaling, kernel
    optimization, case selection, or evaporation-onset selection.
    """
    print(f"Revealing held-out TCAM field for evaluation: {case.case_id}")
    held_out, inputs = load_case_dataframe(case, material_data)
    features = held_out[FEATURE_COLUMNS].to_numpy(dtype=float)
    delta_mean, delta_std = predict_in_batches(
        model=model,
        scaler=scaler,
        X=features,
        batch_size=PREDICTION_BATCH_SIZE,
        return_std=True,
    )
    held_out["delta_T_pred"] = delta_mean
    held_out["delta_T_std"] = delta_std
    held_out["T_corrected_unconstrained"] = (
        held_out["T_ET"] + held_out["delta_T_pred"]
    )
    upper_capped_count = set_corrected_temperature(
        held_out,
        float(inputs["Tevap"]),
    )
    lower_capped = held_out["T_corrected"] < BASE_TEMPERATURE_K
    held_out["T_corrected"] = held_out["T_corrected"].clip(
        lower=BASE_TEMPERATURE_K
    )
    bound_counts = {
        "ambient_lower_bound": int(lower_capped.sum()),
        "evaporation_upper_bound": upper_capped_count,
    }
    return held_out, inputs, bound_counts


def calculate_metrics(held_out):
    def metrics(truth, prediction):
        return {
            "mae_k": float(mean_absolute_error(truth, prediction)),
            "rmse_k": float(mean_squared_error(truth, prediction) ** 0.5),
            "r2": float(r2_score(truth, prediction)),
        }

    temperature_truth = held_out["T_TCAM"]
    residual_truth = held_out["T_TCAM"] - held_out["T_ET"]
    return {
        "raw_et": metrics(temperature_truth, held_out["T_ET"]),
        "gpr_unconstrained": metrics(
            temperature_truth,
            held_out["T_corrected_unconstrained"]
        ),
        "gpr_physics_constrained": metrics(
            temperature_truth,
            held_out["T_corrected"]
        ),
        "residual": metrics(
            residual_truth,
            held_out["delta_T_pred"],
        ),
    }


def write_parity_plot(held_out, output_path):
    truth = held_out["T_TCAM"]
    series = {
        "Raw ET": held_out["T_ET"],
        "GPR, unconstrained": held_out["T_corrected_unconstrained"],
        "GPR + physical bounds": held_out["T_corrected"],
    }
    lower = min(float(truth.min()), *(float(values.min()) for values in series.values()))
    upper = max(float(truth.max()), *(float(values.max()) for values in series.values()))

    figure, axis = plt.subplots(figsize=(7.2, 6.5))
    colors = ("tab:blue", "tab:orange", "tab:green")
    for (label, values), color in zip(series.items(), colors):
        axis.scatter(
            truth,
            values,
            s=8,
            alpha=0.25,
            label=label,
            color=color,
        )
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.5)
    axis.set_xlabel("Held-out TCAM temperature (K)")
    axis.set_ylabel("Predicted temperature (K)")
    axis.set_title("Whole-case held-out temperature prediction")
    axis.legend()
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_residual_plot(held_out, output_path):
    actual = held_out["T_TCAM"] - held_out["T_ET"]
    predicted = held_out["delta_T_pred"]
    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))

    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    points = axis.scatter(
        actual,
        predicted,
        c=held_out["T_ET"],
        cmap="inferno",
        s=8,
        alpha=0.35,
    )
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.5)
    axis.set_xlabel("True held-out residual, TCAM − ET (K)")
    axis.set_ylabel("Predicted residual (K)")
    axis.set_title("Residual prediction on unseen P–V case")
    figure.colorbar(points, ax=axis, label="ET temperature (K)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


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
    axis.set_xlabel("Velocity (m/s)")
    axis.set_ylabel("Power (W)")
    axis.set_title("Multi-case GPR split")
    axis.set_xticks([0.1, 0.5, 1.0, 1.5])
    axis.set_yticks([50, 150, 250, 350])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_experiment(args):
    np.random.seed(args.seed)
    cases = discover_paired_cases(args.et_root, args.tcam_root)
    training_keys = (
        args.training_cases
        if args.training_cases is not None
        else list(DEFAULT_TRAINING_CASES)
    )
    training_cases, held_out_case = resolve_case_split(
        cases,
        training_keys,
        args.held_out_case,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_case_map(
        training_cases,
        held_out_case,
        output_dir / "case_split.png",
    )
    training, training_summary = build_training_table(
        training_cases,
        material_data=args.material_data,
        points_per_case=args.points_per_case,
        seed=args.seed,
    )
    model, scaler = fit_residual_gpr(training, args.seed)

    # Strict leakage boundary: target TCAM is first opened inside this call.
    held_out, held_out_inputs, bound_counts = evaluate_held_out_case(
        held_out_case,
        material_data=args.material_data,
        model=model,
        scaler=scaler,
    )
    metrics = calculate_metrics(held_out)

    training.to_csv(output_dir / "training_sample.csv", index=False)
    training_summary.to_csv(
        output_dir / "training_case_summary.csv",
        index=False,
    )
    held_out.to_csv(
        output_dir / "heldout_predictions_at_tcam_points.csv",
        index=False,
    )
    write_parity_plot(held_out, output_dir / "heldout_parity.png")
    write_residual_plot(
        held_out,
        output_dir / "heldout_residual_parity.png",
    )

    split = {
        "training_cases": [
            {
                "case_id": case.case_id,
                "power_w": case.power_w,
                "velocity_m_s": case.velocity_m_s,
                "et_field": str(case.et_field),
                "tcam_field": str(case.tcam_field),
            }
            for case in training_cases
        ],
        "held_out_case": {
            "case_id": held_out_case.case_id,
            "power_w": held_out_case.power_w,
            "velocity_m_s": held_out_case.velocity_m_s,
            "et_field": str(held_out_case.et_field),
            "tcam_field_used_only_for_evaluation": str(
                held_out_case.tcam_field
            ),
        },
    }
    with (output_dir / "case_split.json").open("w", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)
        stream.write("\n")

    report = {
        "held_out_case": held_out_case.case_id,
        "training_case_count": len(training_cases),
        "training_point_count": len(training),
        "held_out_point_count": len(held_out),
        "feature_columns": FEATURE_COLUMNS,
        "evaporation_onset_temperature_k": float(
            held_out_inputs["Tevap"]
        ),
        "bounded_prediction_counts": bound_counts,
        "metrics": metrics,
        "learned_kernel": str(model.kernel_),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    joblib.dump(
        {
            "gpr": model,
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "training_cases": split["training_cases"],
            "held_out_case": split["held_out_case"],
            "evaporation_onset_temperature_k": float(
                held_out_inputs["Tevap"]
            ),
        },
        output_dir / "multicase_residual_gpr.joblib",
    )

    print("\nHeld-out metrics:")
    for name, values in metrics.items():
        print(
            f"  {name}: MAE={values['mae_k']:.3f} K, "
            f"RMSE={values['rmse_k']:.3f} K, R2={values['r2']:.4f}"
        )
    print(f"Wrote experiment outputs to {output_dir}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual GPR on complete ET/TCAM P-V cases and "
            "evaluate one completely held-out P-V case."
        )
    )
    parser.add_argument("--et-root", type=Path, default=DEFAULT_ET_ROOT)
    parser.add_argument("--tcam-root", type=Path, default=DEFAULT_TCAM_ROOT)
    parser.add_argument(
        "--material-data",
        type=Path,
        default=DEFAULT_MATERIAL_DATA,
    )
    parser.add_argument(
        "--held-out-case",
        type=_parse_case_specification,
        default=DEFAULT_HELD_OUT_CASE,
        metavar="POWER,VELOCITY",
        help="Complete P-V case withheld from model fitting (default: 250,0.5).",
    )
    parser.add_argument(
        "--training-case",
        dest="training_cases",
        type=_parse_case_specification,
        action="append",
        metavar="POWER,VELOCITY",
        help=(
            "Training P-V case; repeat exactly once per case. When omitted, "
            "use all 15 documented cases other than the default holdout."
        ),
    )
    parser.add_argument(
        "--points-per-case",
        type=int,
        default=POINTS_PER_TRAINING_CASE,
        help=f"Maximum sampled points per training case (default: {POINTS_PER_TRAINING_CASE}).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Experiment output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    if args.points_per_case < 10:
        parser.error("--points-per-case must be at least 10.")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
