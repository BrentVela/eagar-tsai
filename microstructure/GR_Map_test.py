from GR_Map import request_GR_Grid_from_excel
import matplotlib.pyplot as plt
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit these parameters before each run.
excel_path = REPO_ROOT / "effective_cp_data.xlsx"
row_index = 2  # EXCEL ROW - 2
element_cols = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
thermo_db = "TCHEA8"
kinetic_db = "MOBHEA3"
primary_phase = "BCC_B2"
interfacial_energy = 0.5
output_path = "CalcFiles/Bayesian_Data/New/TCAM/alloy0_250_0.5/GPR_outputs_5um/physics_based/GR_overlay_corrected2.png"
include_corrected_overlay = True
show_cet_lines = False
# TCAM ON TOP OF ET
# overlay_csv = [
#     "CalcFiles/Test16/250_0.5_v2/liquidus_GR_alloy0_250_0.5.csv",
#     "CalcFiles/Test17/TCAM_keyholing_alloy0_250_0.5/ParaView_data/threshold_data.csv",
# ]
# overlay_label = ["ET liquidus boundary", "TCAM liquidus boundary"]
# overlay_kwargs = [
#     {"c": "gold", "s": 4, "alpha": 0.35},
#     {"c": "lime", "s": 4, "alpha": 0.2},
# ]

# ET ON TOP OF TCAM
overlay_csv = [REPO_ROOT / "CalcFiles/Bayesian_Data/New/TCAM/alloy0_250_0.5/subdivide_data.csv"]
overlay_label = ["TCAM liquidus boundary"]
overlay_kwargs = [
    {"c": "lime", "s": 3, "alpha": 0.1},
]

if include_corrected_overlay:
    overlay_csv.append(
        REPO_ROOT
        / "CalcFiles/Bayesian_Data/New/TCAM/alloy0_250_0.5/GPR_outputs_5um/physics_based/GR_matched_points.csv"
    )
    overlay_label.append("Corrected liquidus points")
    overlay_kwargs.append({"c": "deepskyblue", "s": 3, "alpha": 0.1})

overlay_csv.append(
    REPO_ROOT
    / "CalcFiles/Bayesian_Data/New/ET/alloy0/250_0.5_5um/250_0.5_workflow/liquidus_GR_alloy0_250_0.5.csv"
)
overlay_label.append("ET liquidus boundary")
overlay_kwargs.append({"c": "gold", "s": 4, "alpha": 0.35})

# To overlay multiple CSVs, use lists with matching lengths:
# overlay_csv = [
#     "CalcFiles/Test17/GR_data_alloy0_250_0.5.csv",
#     "CalcFiles/Test16/125_0.2/liquidus_GR_alloy0_125_0.2.csv",
# ]
# overlay_label = ["Thermo-Calc AM liquidus points", "3D liquidus boundary"]
# overlay_kwargs = [
#     {"c": "gold", "s": 8, "alpha": 0.5},
#     {"c": "cyan", "s": 4, "alpha": 0.35},
# ]
disable_cache = False
show_plot = False


request_GR_Grid_from_excel(
    excel_path=excel_path,
    row_index=row_index,
    element_cols=element_cols,
    thermodynamic_database=thermo_db,
    kinetic_database=kinetic_db,
    primary_phase=primary_phase,
    interfacial_energy=interfacial_energy,
    show_plot=show_plot,
    disable_cache=disable_cache,
    disable_output=True,
    save_path=output_path,
    overlay_csv=overlay_csv,
    overlay_label=overlay_label,
    overlay_kwargs=overlay_kwargs,
    legend_loc="upper left",
    show_cet_lines=show_cet_lines,
)
