from pathlib import Path

import pandas as pd
from tc_python import *
from tc_python import CompositionUnit

"""
Batch AM steady-state example using TCHEA8.

What this script does:
- Reads alloy compositions and process conditions from an .xlsx file.
- Uses TCHEA8 + Scheil to derive material properties for each alloy row.
- Runs the AM steady-state calculation with fluid flow due to Marangoni effect.
- Does not enable separate powder material properties.
- Does not create any plots or figures.
- Writes one CSV with melt pool width, depth, length, and peak temperature.

Expected Excel layout:
- `Alloy` and optionally `Composition`
- Composition columns for the alloying elements used in the alloy family
  with `W` as the dependent element
- `Velocity (m/s)` or `velocity_m_s`
- `Power (W)` or `power_w`

Example element columns for this setup:
`W`, `Re`, `Nb`, `Ta`, `Mo`, `Hf`, `V`
"""

INPUT_XLSX = "TCAM_solver_data.xlsx"
OUTPUT_CSV = "CalcFiles/Test15/TCAM_Steady_TCHEA8_results_0and13.csv"
START_EXCEL_ROW = 2

THERMODYNAMIC_DATABASE = "TCHEA8"
SCHEIL_START_TEMPERATURE_K = 6000.0
SCHEIL_TEMPERATURE_STEP_K = 1.0
SCHEIL_TERMINATE_TEMPERATURE_K = 5500.0
N_NUMBER_OF_CORES = 6
DEPENDENT_ELEMENT = "W"

LAYER_THICKNESS_M = 30.0e-6
BEAM_RADIUS_M = 4.0e-5
ABSORPTIVITY_PREFactor = 1.0 #check
WAVELENGTH_NM = 1064.0
SUBSTRATE_HEIGHT_M = 2.0e-3
GAS_PRESSURE_PA = 1.0e5
BASE_PLATE_TEMPERATURE_K = 303.15
AMBIENT_TEMPERATURE_K = 296.15
RADIATION_EMISSIVITY = 0.8
CONVECTIVE_HEAT_TRANSFER_COEFFICIENT = 20.0
COMPOSITION_UNIT = CompositionUnit.MOLE_PERCENT
COMPOSITION_COLUMNS = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
INDEPENDENT_COMPOSITION_COLUMNS = [col for col in COMPOSITION_COLUMNS if col != DEPENDENT_ELEMENT]

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
        .set_global_minimization_test_interval(1)
        .set_global_minimization_max_grid_points(2000)
        .set_liquid_phase("LIQUID")
        .set_gas_phase("GAS")
        .enable_equilibrium_solidification_calculation()
        .enable_approximate_driving_force_for_metastable_phases()
        .calculate_from_start_temperature()
        .set_temperature_step(SCHEIL_TEMPERATURE_STEP_K)
        .terminate_on_temperature(SCHEIL_TERMINATE_TEMPERATURE_K)
        .set_max_no_of_iterations(500)
        .set_required_accuracy(1.0e-6)
        .set_smallest_fraction(1.0e-12)
        .calculate_to_temperature_below_solidus(number_of_steps=50, final_temperature=298.15)
        .enable_evaporation_property_calculation()
    )


def _result_peak_temperature(mesh):
    return float(mesh["temperature"].max())


def _run_steady_state(session, material_properties, velocity_m_s, power_w):
    heat_source = (
        HeatSource.gaussian_with_calculated_absorptivity(
            ABSORPTIVITY_PREFactor,
            WAVELENGTH_NM,
        )
        .set_power(float(power_w))
        .set_beam_radius(BEAM_RADIUS_M)
        .set_scanning_speed(float(velocity_m_s))
        .disable_keyhole_model()
    )

    calc = (
        session.with_additive_manufacturing()
        .with_steady_state_calculation()
        .with_numerical_options(NumericalOptions().set_number_of_cores(N_NUMBER_OF_CORES))
        .with_material_properties(material_properties)
        .with_mesh(Mesh.coarse())
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
    mesh = result.get_pyvista_mesh()

    return {
        "meltpool_width_m": result.get_meltpool_width(),
        "meltpool_depth_m": result.get_meltpool_depth(),
        "meltpool_length_m": result.get_meltpool_length(),
        "meltpool_width_um": result.get_meltpool_width() * 1.0e6,
        "meltpool_depth_um": result.get_meltpool_depth() * 1.0e6,
        "meltpool_length_um": result.get_meltpool_length() * 1.0e6,
        "peak_temperature_k": _result_peak_temperature(mesh),
    }


df = pd.read_excel(INPUT_XLSX)
df = df.rename(columns=COLUMN_MAP)

# Excel row numbers are 1-based and row 1 is assumed to be the header.
# Set START_EXCEL_ROW to the first data row you want to process.
if START_EXCEL_ROW > 2:
    df = df.iloc[START_EXCEL_ROW - 2 :].reset_index(drop=True)

if "Alloy" not in df.columns:
    raise ValueError(f"Missing required column 'Alloy' in {INPUT_XLSX}.")
if "velocity_m_s" not in df.columns or "power_w" not in df.columns:
    raise ValueError(
        f"Missing required process columns in {INPUT_XLSX}: 'Velocity (m/s)'/'velocity_m_s' and "
        "'Power (W)'/'power_w'."
    )

missing_composition_columns = [col for col in COMPOSITION_COLUMNS if col not in df.columns]
if missing_composition_columns:
    raise ValueError(
        f"{INPUT_XLSX} must contain composition columns: {', '.join(missing_composition_columns)}"
    )

element_columns = COMPOSITION_COLUMNS
if DEPENDENT_ELEMENT not in element_columns:
    raise ValueError(
        f"{INPUT_XLSX} must contain the dependent element column '{DEPENDENT_ELEMENT}'."
    )

output_rows = []

with TCPython() as session:
    session.set_cache_folder("cache")

    # Build the union of elements once. The same system is reused for every row.
    system = session.select_database_and_elements(THERMODYNAMIC_DATABASE, element_columns).get_system()

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        composition_label = _format_composition_label(row, element_columns)

        try:
            if pd.isna(row_dict["velocity_m_s"]) or pd.isna(row_dict["power_w"]):
                raise ValueError("Missing power or velocity.")

            composition_values = _row_composition_values(row, INDEPENDENT_COMPOSITION_COLUMNS)
            if not composition_label:
                composition_label = "; ".join(
                    f"{element}{value:g}" for element, value in composition_values.items()
                )
            row_dict["Composition"] = composition_label

            scheil = (
                system.with_scheil_calculation()
                .with_calculation_type(ScheilCalculationType.scheil_classic())
                .set_composition_unit(COMPOSITION_UNIT)
                .set_start_temperature(SCHEIL_START_TEMPERATURE_K)
                .with_options(_build_scheil_options())
                .enable_global_minimization()
            )
            for element in INDEPENDENT_COMPOSITION_COLUMNS:
                scheil.set_composition(element, composition_values.get(element, 0.0))

            scheil_result = scheil.calculate()
            material_properties = MaterialProperties.from_scheil_result(
                scheil_result,
                interface_scattering_constant=4.0e-8,
            )
            material_properties.set_smoothing_for_all_properties(Smoothing.MEDIUM)

            am_output = _run_steady_state(
                session,
                material_properties,
                row_dict["velocity_m_s"],
                row_dict["power_w"],
            )

            row_dict["liquidus_temperature_k"] = float(material_properties.get_liquidus_temperature())
            row_dict.update(am_output)

        except Exception as exc:
            print(f"Row failed for alloy '{row_dict.get('Alloy', '')}': {exc}")
            for col in OUTPUT_COLUMNS:
                row_dict[col] = float("nan")

        output_rows.append(row_dict)

result = pd.DataFrame(output_rows)

front_columns = ["Alloy", "Composition"]
remaining_columns = [col for col in result.columns if col not in front_columns]
result = result[front_columns + remaining_columns]

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False)

print(result[["Alloy", "meltpool_width_um", "meltpool_depth_um", "meltpool_length_um", "peak_temperature_k"]])
