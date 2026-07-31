from pathlib import Path
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from tc_python import *
from tc_python import CompositionUnit

"""
Batch AM steady-state example using TCHEA8.

What this script does:
- Reads Alloy 0 compositions and process conditions from effective_cp_data.xlsx.
- Uses TCHEA8 + Scheil to derive material properties for each alloy row.
- Runs the AM steady-state calculation with fluid flow due to Marangoni effect.
- Uses the TCAM GUI settings documented for Alloy 0.
- Does not create any plots or figures.
- Writes one CSV with melt pool width, depth, length, and peak temperature.
- Archives the complete native TC-Python result bundle for every AM run.

Expected Excel layout:
- `Alloy` and optionally `Composition`
- Composition columns for the alloying elements used in the alloy family.
  The largest component in each row is used as the dependent element.
- `Velocity (m/s)` or `velocity_m_s`
- `Power (W)` or `power_w`

Recognized element columns for this setup:
`W`, `Re`, `Nb`, `Ta`, `Mo`, `Hf`, `V`, `Co`, `Cr`, `Fe`, `Mn`, `Ni`
"""

INPUT_XLSX = "TCAM_solver_data.xlsx"
OUTPUT_CSV = "beamer/figures/data/BU_TCAM/TCAM_Alloy19.csv"
RAW_RESULTS_DIR = "beamer/figures/data/Alloy19"
CACHE_DIR = "cache"
TARGET_ALLOY = "19"
START_EXCEL_ROW = 18
END_EXCEL_ROW = 18 #inclusive ; None if you want to go to the end
EXPORT_LIQUIDUS_GR = False

THERMODYNAMIC_DATABASE = "TCHEA8"
SCHEIL_START_TEMPERATURE_K = 6000.0
SCHEIL_TEMPERATURE_STEP_K = 1.0
SCHEIL_TERMINATE_LIQUID_FRACTION = 0.01
SCHEIL_FINAL_TEMPERATURE_K = 298.15
N_NUMBER_OF_CORES = 10
LAYER_THICKNESS_M = 30.0e-6
BEAM_RADIUS_M = 4.0e-5
ABSORPTIVITY_PREFACTOR = 1.0
WAVELENGTH_NM = 1070.0
BEAM_QUALITY_FACTOR_M2 = 1.0
SUBSTRATE_HEIGHT_M = 0.40e-3 # 0.22e-3 default
GAS_PRESSURE_PA = 101325.0
BASE_PLATE_TEMPERATURE_K = 298.0
AMBIENT_TEMPERATURE_K = 296.15
RADIATION_EMISSIVITY = 0.8
CONVECTIVE_HEAT_TRANSFER_COEFFICIENT = 20.0
SMAGORINSKY_CONSTANT = 0.18
COMPOSITION_UNIT = CompositionUnit.MOLE_PERCENT
COMPOSITION_COLUMNS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V", "Co", "Cr", "Fe", "Mn", "Ni"]

COLUMN_MAP = {
    "Velocity (m/s)": "velocity_m_s",
    "Power (W)": "power_w",
    "Velocity_m_s": "velocity_m_s",
    "Power_w": "power_w",
}

IDENTITY_COLUMNS = ["Alloy", "Composition", "Classification", "velocity_m_s", "power_w", "Absorptivity"]
OUTPUT_COLUMNS = [
    "liquidus_temperature_k",
    "meltpool_width_m",
    "meltpool_depth_m",
    "meltpool_length_m",
    "meltpool_width_um",
    "meltpool_depth_um",
    "meltpool_length_um",
    "peak_temperature_k",
]
RAW_OUTPUT_COLUMNS = [
    "raw_result_directory",
    "raw_result_file",
    "raw_info_json",
    "raw_gr_csv",
    "gr_point_count",
]


def _format_composition_label(row, element_columns):
    if "Composition" in row and pd.notna(row["Composition"]):
        text = str(row["Composition"]).strip()
        if text:
            return text

    parts = []
    for element in element_columns:
        value = row.get(element)
        if pd.notna(value) and float(value) != 0.0:
            parts.append(f"{element}{float(value):g}")
    return "; ".join(parts)


def _row_composition_values(row, element_columns):
    """
    Read the independent composition columns without renormalizing the alloy.

    If the Excel values are already in mole percent, pass them through.
    If they look like fractions, convert them to mole percent because the
    script uses `CompositionUnit.MOLE_PERCENT`.
    """
    values = {}
    for element in element_columns:
        value = row.get(element)
        if pd.notna(value):
            values[element] = float(value)

    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("No positive composition values found.")

    if COMPOSITION_UNIT == CompositionUnit.MOLE_PERCENT:
        # Keep mole percent values as-is; only scale fractions into percent.
        if total <= 1.5:
            values = {element: value * 100.0 for element, value in values.items()}
    else:
        # Keep mole fraction values as-is; only scale percent into fraction.
        if total > 1.5:
            values = {element: value / 100.0 for element, value in values.items()}

    return values


def _build_scheil_options():
    return (
        ScheilOptions()
        .set_global_minimization_test_interval(10)
        .set_global_minimization_max_grid_points(2000)
        .set_liquid_phase("LIQUID")
        .set_gas_phase("GAS")
        .disable_equilibrium_solidification_calculation()
        .enable_approximate_driving_force_for_metastable_phases()
        .calculate_from_start_temperature()
        .set_temperature_step(SCHEIL_TEMPERATURE_STEP_K)
        .terminate_on_fraction_of_liquid_phase(
            SCHEIL_TERMINATE_LIQUID_FRACTION
        )
        .set_max_no_of_iterations(500)
        .set_required_accuracy(1.0e-6)
        .set_smallest_fraction(1.0e-12)
        .calculate_to_temperature_below_solidus(
            number_of_steps=50,
            final_temperature=SCHEIL_FINAL_TEMPERATURE_K,
        )
        .enable_evaporation_property_calculation()
    )


def _case_number(value):
    """Format a process value for a readable TCAM case-directory name."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Case-directory value must be finite, got {value}.")
    return f"{number:g}"


def _alloy_identifier(value):
    """Return a compact, filesystem-safe alloy identifier."""
    try:
        return _case_number(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        identifier = "".join(
            character
            if character.isalnum() or character in "-_."
            else "_"
            for character in text
        )
        if not identifier:
            raise ValueError("Alloy identifier cannot be empty.")
        return identifier


def _raw_result_directory(row_dict):
    return (
        Path(RAW_RESULTS_DIR)
        / (
            f"alloy{_alloy_identifier(row_dict['Alloy'])}"
            f"_{_case_number(row_dict['power_w'])}"
            f"_{_case_number(row_dict['velocity_m_s'])}"
            f"_row{int(row_dict['source_excel_row'])}"
        )
    )


def _archive_and_validate_result(result, destination):
    """
    Consolidate the native partitioned result into one standalone VTU.

    The output directory is built and validated in a temporary sibling
    directory before replacing any prior generated archive. Its normal
    contents are therefore only result.vtu and info.json.
    """
    source_result_file = Path(result.get_result_file_path()).resolve()
    source_directory = source_result_file.parent
    source_info_json = source_directory / "info.json"
    destination = Path(destination).resolve()

    if not source_result_file.is_file():
        raise FileNotFoundError(
            f"TC-Python returned a missing result file: {source_result_file}"
        )
    if not source_info_json.is_file():
        raise FileNotFoundError(
            f"TC-Python result has no info.json: {source_info_json}"
        )
    if source_directory == destination:
        raise ValueError("Raw result destination must differ from the cache.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_info_json.open(encoding="utf-8") as stream:
        info = json.load(stream)
    if info.get("mainFileName") != source_result_file.name:
        raise ValueError(
            "info.json identifies mainFileName="
            f"{info.get('mainFileName')!r}, expected "
            f"{source_result_file.name!r}."
        )

    import pyvista as pv

    root = pv.read(source_result_file)
    source = root.combine() if isinstance(root, pv.MultiBlock) else root
    if "subdomain_id" in source.array_names:
        source = source.threshold(
            value=3,
            scalars="subdomain_id",
            invert=True,
        )
    if "temperature" in source.cell_data:
        source = source.cell_data_to_point_data(pass_cell_data=True)
    if "temperature" not in source.point_data:
        raise ValueError(
            f"TC-Python mesh has no temperature data: {source_result_file}"
        )
    if source.n_points == 0 or source.n_cells == 0:
        raise ValueError(f"TC-Python mesh is empty: {source_result_file}")

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        temporary_result_file = temporary_directory / "result.vtu"
        temporary_info_json = temporary_directory / "info.json"
        source.save(temporary_result_file)

        info["nativeMainFileName"] = info["mainFileName"]
        info["mainFileName"] = temporary_result_file.name
        info["archiveFormat"] = "consolidated-vtu"
        with temporary_info_json.open("w", encoding="utf-8") as stream:
            json.dump(info, stream, indent=2, allow_nan=True)
            stream.write("\n")

        # Prove that the standalone file is readable before replacing an
        # earlier generated archive.
        compact_mesh = pv.read(temporary_result_file)
        if (
            compact_mesh.n_points != source.n_points
            or compact_mesh.n_cells != source.n_cells
        ):
            raise ValueError(
                "Consolidated VTU mesh size differs from the TC-Python mesh."
            )
        if "temperature" not in compact_mesh.point_data:
            raise ValueError(
                f"Consolidated mesh has no temperature data: "
                f"{temporary_result_file}"
            )

        if destination.exists():
            shutil.rmtree(destination)
        temporary_directory.rename(destination)
    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise

    archived_result_file = destination / "result.vtu"
    archived_info_json = destination / "info.json"

    return {
        "raw_result_directory": str(destination),
        "raw_result_file": str(archived_result_file),
        "raw_info_json": str(archived_info_json),
        "peak_temperature_k": float(compact_mesh["temperature"].max()),
    }


def _export_liquidus_gr(result, destination, liquidus_temperature_k):
    """
    Export TC-Python's built-in liquidus-front G and R evaluation.

    TC-Python returns [G, R, x, y, z], with G in K/m, R in m/s, and
    coordinates in m. The column names match the existing TCAM plotting
    workflow, but no ParaView gradient or contour-normal calculation is used.
    """
    values = result.get_thermal_gradient_and_solidification_rate()
    if len(values) != 5:
        raise ValueError(
            "Expected TC-Python G/R result [G, R, x, y, z], "
            f"received {len(values)} arrays."
        )

    lengths = [len(array) for array in values]
    if not lengths[0]:
        raise ValueError("TC-Python returned no liquidus-front G/R points.")
    if len(set(lengths)) != 1:
        raise ValueError(
            f"TC-Python returned inconsistent G/R array lengths: {lengths}"
        )

    gradient, solidification_rate, x, y, z = values
    gr = pd.DataFrame(
        {
            "Gradient_Magnitude": gradient,
            "R (m/s)": solidification_rate,
            "Points_0": x,
            "Points_1": y,
            "Points_2": z,
            "temperature": float(liquidus_temperature_k),
        }
    )
    numeric_values = gr.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("TC-Python liquidus-front G/R data is non-finite.")

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    gr_csv = destination / "TCAM_GR.csv"
    gr.to_csv(gr_csv, index=False)

    return {
        "raw_gr_csv": str(gr_csv),
        "gr_point_count": len(gr),
    }


def _run_steady_state(
    session,
    material_properties,
    velocity_m_s,
    power_w,
    raw_result_directory,
):
    heat_source = (
        HeatSource.gaussian_with_calculated_absorptivity(
            ABSORPTIVITY_PREFACTOR,
            WAVELENGTH_NM,
        )
        .set_power(float(power_w))
        .set_beam_radius(BEAM_RADIUS_M)
        .set_scanning_speed(float(velocity_m_s))
        .with_keyhole_model(
            KeyholeModel().set_beam_quality(BEAM_QUALITY_FACTOR_M2)
        )
    )
    numerical_options = (
        NumericalOptions()
        .set_number_of_cores(N_NUMBER_OF_CORES)
        .enable_petrov_galerkin()
        .disable_damping()
        .set_smagorinsky_constant(SMAGORINSKY_CONSTANT)
    )

    calc = (
        session.with_additive_manufacturing()
        .with_steady_state_calculation()
        .with_numerical_options(numerical_options)
        .with_material_properties(material_properties)
        .with_mesh(Mesh.fine())
        .set_gas_pressure(GAS_PRESSURE_PA)
        .set_base_plate_temperature(BASE_PLATE_TEMPERATURE_K)
        .set_ambient_temperature(AMBIENT_TEMPERATURE_K)
        .set_height(SUBSTRATE_HEIGHT_M)
        .set_layer_thickness(LAYER_THICKNESS_M)
        .enable_fluid_flow_marangoni()
        .disable_separate_materials()
        .with_top_boundary_conditions(
            TopBoundaryConditions()
            .set_radiation_emissivity(RADIATION_EMISSIVITY)
            .set_convective_heat_coefficient(CONVECTIVE_HEAT_TRANSFER_COEFFICIENT)
            .enable_evaporation()
        )
        .with_heat_source(heat_source)
    )

    result = calc.calculate()
    archived = _archive_and_validate_result(result, raw_result_directory)
    if EXPORT_LIQUIDUS_GR:
        gr_output = _export_liquidus_gr(
            result,
            archived["raw_result_directory"],
            material_properties.get_liquidus_temperature(),
        )
    else:
        gr_output = {
            "raw_gr_csv": None,
            "gr_point_count": 0,
        }

    return {
        "meltpool_width_m": result.get_meltpool_width(),
        "meltpool_depth_m": result.get_meltpool_depth(),
        "meltpool_length_m": result.get_meltpool_length(),
        "meltpool_width_um": result.get_meltpool_width() * 1.0e6,
        "meltpool_depth_um": result.get_meltpool_depth() * 1.0e6,
        "meltpool_length_um": result.get_meltpool_length() * 1.0e6,
        **archived,
        **gr_output,
    }


def _load_input_rows():
    df = pd.read_excel(INPUT_XLSX)
    df = df.rename(columns=COLUMN_MAP)
    df["source_excel_row"] = range(2, len(df) + 2)

    # Excel row numbers are 1-based and row 1 is assumed to be the header.
    # END_EXCEL_ROW is inclusive; None processes through the final row.
    if START_EXCEL_ROW > 2:
        df = df.loc[df["source_excel_row"] >= START_EXCEL_ROW].copy()
    if END_EXCEL_ROW is not None:
        if END_EXCEL_ROW < START_EXCEL_ROW:
            raise ValueError(
                f"END_EXCEL_ROW ({END_EXCEL_ROW}) must be greater than or "
                f"equal to START_EXCEL_ROW ({START_EXCEL_ROW})."
            )
        df = df.loc[df["source_excel_row"] <= END_EXCEL_ROW].copy()

    if "Alloy" not in df.columns:
        raise ValueError(f"Missing required column 'Alloy' in {INPUT_XLSX}.")
    if "velocity_m_s" not in df.columns or "power_w" not in df.columns:
        raise ValueError(
            f"Missing required process columns in {INPUT_XLSX}: "
            "'Velocity (m/s)'/'velocity_m_s' and "
            "'Power (W)'/'power_w'."
        )

    df = df.loc[df["Alloy"].astype(str) == TARGET_ALLOY].copy()
    if df.empty:
        raise ValueError(
            f"No rows for Alloy {TARGET_ALLOY} were found in {INPUT_XLSX}."
        )

    # Do not add zero-valued elements to the thermodynamic system. For Alloy 0
    # this reproduces the GUI system Ta-W-Hf-Mo instead of also selecting the
    # zero-valued Re, Nb, and V workbook columns.
    candidate_columns = [
        column for column in COMPOSITION_COLUMNS if column in df.columns
    ]
    element_columns = [
        column
        for column in candidate_columns
        if pd.to_numeric(df[column], errors="coerce").fillna(0.0).ne(0.0).any()
    ]
    if not element_columns:
        raise ValueError(
            f"{INPUT_XLSX} has no nonzero recognized composition columns: "
            f"{', '.join(COMPOSITION_COLUMNS)}"
        )

    return df.reset_index(drop=True), element_columns


def main():
    df, element_columns = _load_input_rows()
    output_rows = []

    with TCPython() as session:
        session.set_cache_folder(CACHE_DIR)

        # All filtered Alloy 0 rows have the same composition, so one immutable
        # system can be reused across the complete P-V batch.
        system = session.select_database_and_elements(
            THERMODYNAMIC_DATABASE,
            element_columns,
        ).get_system()

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            composition_label = _format_composition_label(
                row,
                element_columns,
            )

            try:
                if (
                    pd.isna(row_dict["velocity_m_s"])
                    or pd.isna(row_dict["power_w"])
                ):
                    raise ValueError("Missing power or velocity.")

                composition_values = _row_composition_values(
                    row,
                    element_columns,
                )
                dependent_element = max(
                    composition_values,
                    key=composition_values.get,
                )
                if not composition_label:
                    composition_label = "; ".join(
                        f"{element}{value:g}"
                        for element, value in composition_values.items()
                    )
                row_dict["Composition"] = composition_label

                scheil = (
                    system.with_scheil_calculation()
                    .with_calculation_type(
                        ScheilCalculationType.scheil_classic()
                    )
                    .set_composition_unit(COMPOSITION_UNIT)
                    .set_start_temperature(SCHEIL_START_TEMPERATURE_K)
                    .with_options(_build_scheil_options())
                    .enable_global_minimization()
                )
                for element in element_columns:
                    if element == dependent_element:
                        continue
                    scheil.set_composition(
                        element,
                        composition_values.get(element, 0.0),
                    )

                scheil_result = scheil.calculate()
                material_properties = MaterialProperties.from_scheil_result(
                    scheil_result,
                    interface_scattering_constant=4.0e-8,
                )
                material_properties.set_smoothing_for_all_properties(
                    Smoothing.LITTLE
                )

                am_output = _run_steady_state(
                    session,
                    material_properties,
                    row_dict["velocity_m_s"],
                    row_dict["power_w"],
                    _raw_result_directory(row_dict),
                )

                row_dict["liquidus_temperature_k"] = float(
                    material_properties.get_liquidus_temperature()
                )
                row_dict.update(am_output)

            except Exception as exc:
                print(
                    "Row failed for "
                    f"Alloy {row_dict.get('Alloy', '')}, "
                    f"P={row_dict.get('power_w', '')} W, "
                    f"V={row_dict.get('velocity_m_s', '')} m/s: {exc}"
                )
                for column in OUTPUT_COLUMNS + RAW_OUTPUT_COLUMNS:
                    row_dict[column] = float("nan")

            output_rows.append(row_dict)

    result = pd.DataFrame(output_rows)
    front_columns = ["Alloy", "Composition", "power_w", "velocity_m_s"]
    remaining_columns = [
        column for column in result.columns if column not in front_columns
    ]
    result = result[front_columns + remaining_columns]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print(
        result[
            [
                "Alloy",
                "power_w",
                "velocity_m_s",
                "meltpool_width_um",
                "meltpool_depth_um",
                "meltpool_length_um",
                "peak_temperature_k",
                "raw_result_file",
                "raw_gr_csv",
                "gr_point_count",
            ]
        ]
    )


if __name__ == "__main__":
    main()
