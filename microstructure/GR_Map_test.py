from GR_Map import request_GR_Grid_from_excel
import matplotlib.pyplot as plt

# Edit these parameters before each run.
excel_path = "et_custom_input_data.xlsx"
row_index = 6  # first alloy (row 2 in the spreadsheet)
element_cols = ["W", "Re", "Nb", "Ta", "Mo", "Hf", "V"]
thermo_db = "TCHEA8"
kinetic_db = "MOBHEA3"
primary_phase = "BCC_B2"
interfacial_energy = 0.5
output_path = "CalcFiles/Test16/125_0.2/GR_map_overlay_alloy0_125_0.2.png"
overlay_csv = "CalcFiles/Test16/125_0.2/liquidus_GR_alloy0_125_0.2.csv"
overlay_label = "3D liquidus boundary"
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
)
