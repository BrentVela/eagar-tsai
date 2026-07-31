from math import log10

from tc_python import *
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
import os


DEFAULT_PRIMARY_PHASE = "BCC_B2" 
DEFAULT_INTERFACIAL_ENERGY = 0.25 # Default, but will try to take value from spreadsheet
DEFAULT_CET_NUCLEATION_SITES = 2.0e15
DEFAULT_CET_NUCLEATION_UNDERCOOLING_K = 2.5
DEFAULT_CET_EQUIAXED_EXPONENT = 3.4


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _tc_python_cache_dir():
    return os.path.join(_repo_root(), "TC_Python", "caches", "GR_Map_cache")


def _cet_grid_npz_cache_dir():
    return os.path.join(_repo_root(), "TC_Python", "caches", "CET_grid_npz")


def _tc_python_calcfiles_dir():
    return os.path.join(_repo_root(), "TC_Python", "CalcFiles")


def _cet_grid_cache_metadata(
    thermodynamic_database,
    kinetic_database,
    element_names,
    solutes,
    primary_phase,
    interfacial_energy,
    nb_nucleations_site,
    nucleation_undercooling,
    equiaxed_exponent,
    composition_unit,
):
    return {
        "cache_format_version": 1,
        "thermodynamic_database": str(thermodynamic_database),
        "kinetic_database": str(kinetic_database),
        "elements": sorted(str(element) for element in element_names),
        "solutes": {
            str(element): float(amount)
            for element, amount in sorted(solutes.items())
        },
        "primary_phase": str(primary_phase),
        "interfacial_energy_j_m2": float(interfacial_energy),
        "nucleation_sites_m3": float(nb_nucleations_site),
        "nucleation_undercooling_k": float(nucleation_undercooling),
        "equiaxed_exponent": float(equiaxed_exponent),
        "composition_unit": str(composition_unit),
        "log10_r_grid": [-7.0, 1.0, 50],
        "log10_g_grid": [3.0, 10.0, 50],
    }


def _cet_grid_npz_path(metadata):
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_cet_grid_npz_cache_dir(), f"CET_grid_{digest}.npz")


def _load_cet_grid_npz(npz_path):
    if not os.path.isfile(npz_path):
        return None

    with np.load(npz_path, allow_pickle=False) as cached:
        required = {
            "r_grid",
            "g_grid",
            "equiaxed_fraction",
            "tip_radius",
            "dendrite_tip_undercooling",
            "nan_points",
            "metadata_json",
        }
        missing = required.difference(cached.files)
        if missing:
            raise ValueError(
                f"CET grid cache {npz_path} is missing arrays: {sorted(missing)}"
            )
        return (
            cached["r_grid"].copy(),
            cached["g_grid"].copy(),
            cached["equiaxed_fraction"].copy(),
            cached["tip_radius"].copy(),
            cached["dendrite_tip_undercooling"].copy(),
            cached["nan_points"].copy(),
        )


def _save_cet_grid_npz(
    npz_path,
    metadata,
    r_grid,
    g_grid,
    equiaxed_fraction,
    tip_radius,
    dendrite_tip_undercooling,
    nan_points,
):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    nan_points_array = np.asarray(nan_points, dtype=float)
    if nan_points_array.size == 0:
        nan_points_array = np.empty((0, 2), dtype=float)
    else:
        nan_points_array = nan_points_array.reshape(-1, 2)

    np.savez_compressed(
        npz_path,
        r_grid=np.asarray(r_grid, dtype=float),
        g_grid=np.asarray(g_grid, dtype=float),
        equiaxed_fraction=np.asarray(equiaxed_fraction, dtype=float),
        tip_radius=np.asarray(tip_radius, dtype=float),
        dendrite_tip_undercooling=np.asarray(
            dendrite_tip_undercooling,
            dtype=float,
        ),
        nan_points=nan_points_array,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True),
        ),
    )


def _plot_overlay_points(ax, overlay_csv, label=None, point_kwargs=None):
    overlay_df = pd.read_csv(overlay_csv)
    g_col = next(
        (column for column in ("G", "Gradient_Magnitude") if column in overlay_df.columns),
        None,
    )
    r_col = next(
        (column for column in ("R_x", "R", "R (m/s)") if column in overlay_df.columns),
        None,
    )
    if g_col is None:
        raise ValueError(
            f"{overlay_csv} is missing required column 'G' or 'Gradient_Magnitude'."
        )
    if r_col is None:
        raise ValueError(
            f"{overlay_csv} is missing required column 'R_x', 'R', or 'R (m/s)'."
        )

    r_values = overlay_df[r_col].to_numpy(dtype=float)
    g_values = overlay_df[g_col].to_numpy(dtype=float)
    finite_g = g_values[np.isfinite(g_values) & (g_values > 0)]
    if finite_g.size and np.nanmedian(finite_g) < 1.0e4:
        g_values = g_values * 1.0e6

    mask = np.isfinite(r_values) & np.isfinite(g_values) & (r_values > 0) & (g_values > 0)
    if not np.any(mask):
        raise ValueError(f"No plottable overlay points found in {overlay_csv}.")

    kwargs = {
        "s": 4,
        "c": "gold",
        "edgecolors": "none",
        "alpha": 0.35,
        "marker": "o",
        "zorder": 5,
    }
    if point_kwargs:
        kwargs.update(point_kwargs)

    ax.scatter(
        r_values[mask],
        g_values[mask],
        label=label or f"{os.path.basename(overlay_csv)} overlay",
        **kwargs,
    )


def _as_overlay_list(value, count, name):
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"{name} must have {count} entries when overlay_csv has {count} entries.")
        return list(value)
    return [value] * count


def _style_overlay_legend(legend, marker_size=48):
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])

    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([marker_size])
        if hasattr(handle, "set_edgecolors"):
            handle.set_edgecolors(["black"])
        if hasattr(handle, "set_linewidths"):
            handle.set_linewidths([0.8])


def _load_alloy_from_excel(excel_path, row_index, element_cols):
    df = pd.read_excel(excel_path)
    row = df.iloc[row_index]
    elements = {el: float(row[el]) for el in element_cols if el in df.columns}
    if not elements:
        raise ValueError(f"No element columns found in {excel_path}. Expected one of: {element_cols}")
    # Spreadsheet composition columns are mole/atomic fractions that sum to 1.
    elements = {el: val for el, val in elements.items() if not pd.isna(val)}
    elements = {key: v for key, v in sorted(elements.items(), key=lambda item: item[1], reverse=True)}
    dependent_element = list(elements.keys())[0]
    solutes = elements.copy()
    solutes.pop(dependent_element)
    return elements, solutes


def _load_liquidus_from_excel(excel_path, row_index):
    row = pd.read_excel(excel_path).iloc[row_index]
    for column in ("Liquidus (K)", "liquidus_temperature_k", "Liquidus"):
        if column in row.index and not pd.isna(row[column]):
            liquidus_temperature = float(row[column])
            if not math.isfinite(liquidus_temperature) or liquidus_temperature <= 1.0:
                raise ValueError(
                    f"Invalid liquidus temperature {row[column]!r} in column {column!r}."
                )
            return liquidus_temperature
    raise ValueError(
        f"Excel row {row_index} in {excel_path} has no liquidus temperature. "
        "Expected one of: 'Liquidus (K)', 'liquidus_temperature_k', or 'Liquidus'."
    )


def _load_interfacial_energy_from_excel(excel_path, row_index):
    """Return a spreadsheet interfacial energy, or None for an empty cell."""
    column = "Interfacial Energy (J/m^2)"
    row = pd.read_excel(excel_path).iloc[row_index]
    if column not in row.index or pd.isna(row[column]):
        return None

    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {column!r} value {row[column]!r} in Excel row {row_index}."
        ) from exc

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{column!r} must be a positive finite value in Excel row "
            f"{row_index}; got {row[column]!r}."
        )
    return value


def estimate_interfacial_energy(
    system,
    solutes,
    liquidus_temperature,
    primary_phase,
    composition_unit=CompositionUnit.MOLE_FRACTION,
):
    """Estimate liquid/primary-solid interfacial energy with Thermo-Calc."""
    calc = system.with_property_model_calculation("Interfacial Energy")
    calc.set_composition_unit(composition_unit)
    for element, amount in solutes.items():
        calc.set_composition(element, amount)

    evaluation_temperature = float(liquidus_temperature) - 1.0
    calc.set_temperature(evaluation_temperature)
    calc.set_argument("matrix", "LIQUID")
    calc.set_argument("precipitate", primary_phase)
    result = calc.calculate()
    interfacial_energy = float(result.get_value_of("interfacialEnergy"))
    if not math.isfinite(interfacial_energy) or interfacial_energy <= 0.0:
        raise ValueError(
            "The Thermo-Calc Interfacial Energy model returned "
            f"{interfacial_energy!r} J/m^2."
        )
    return interfacial_energy


def resolve_interfacial_energy(
    system,
    solutes,
    primary_phase,
    liquidus_temperature,
    interfacial_energy=None,
    composition_unit=CompositionUnit.MOLE_FRACTION,
    report=True,
):
    if interfacial_energy is not None:
        value = float(interfacial_energy)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("interfacial_energy must be a positive finite value.")
        if report:
            print(f"Using specified interfacial energy: {value:.6g} J/m^2")
        return value

    try:
        value = estimate_interfacial_energy(
            system,
            solutes,
            liquidus_temperature,
            primary_phase,
            composition_unit=composition_unit,
        )
    except Exception as exc:
        raise RuntimeError(
            "Thermo-Calc could not estimate the LIQUID/"
            f"{primary_phase} interfacial energy at "
            f"{float(liquidus_temperature) - 1.0:.6g} K. "
            "Supply an explicit fallback such as "
            f"--interfacial-energy {DEFAULT_INTERFACIAL_ENERGY:g}."
        ) from exc

    if report:
        print(
            "Estimated LIQUID/"
            f"{primary_phase} interfacial energy at "
            f"{float(liquidus_temperature) - 1.0:.6g} K: {value:.6g} J/m^2"
        )
    return value


def get_CET_grid(
    system,
    solutes,
    primary_phase=DEFAULT_PRIMARY_PHASE,
    interfacial_energy=DEFAULT_INTERFACIAL_ENERGY,
    nb_nucleations_site=DEFAULT_CET_NUCLEATION_SITES,
    nucleation_undercooling=DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
    equiaxed_exponent=DEFAULT_CET_EQUIAXED_EXPONENT,
    disable_output=True,
    composition_unit=CompositionUnit.MOLE_FRACTION,
):
    calc = system.with_property_model_calculation("CETPythonModel")
    if not disable_output:
        print('Calculator arguments:', list(calc.get_arguments()))
    calc.set_composition_unit(composition_unit)

    for element in solutes:
        calc.set_composition(element, solutes[element])

    logv_list = np.linspace(-7, 1, 50)
    logG_list = np.linspace(3, 10, 50)
    v = []
    G = []
    EF = []
    TR = []
    DTU = []
    nan_points = []
    calc.set_argument("primaryPhase", primary_phase)
    calc.set_argument("InterfacialEnergy", str(interfacial_energy))
    calc.set_argument("NbNucleationSites", str(nb_nucleations_site))
    calc.set_argument("NucleationUndercooling", str(nucleation_undercooling))
    calc.set_argument("EquiaxedExponent", str(equiaxed_exponent))
    calc.set_argument("Solve for", "Equiaxed fraction")

    for logv in logv_list:
        for logG in logG_list:
            calc.set_argument("Log10(v)", logv)
            calc.set_argument("TemperatureGradient", 10 ** logG)

            result = calc.calculate()
            equiaxed_fraction = result.get_value_of("fractionEquiaxedGrains")

            EF.append(equiaxed_fraction)
            TR.append(result.get_value_of("Tip radius"))
            DTU.append(result.get_value_of("Dendrite tip undercooling"))
            v.append(10 ** logv)
            G.append(10 ** logG)
            if not disable_output:
                print("CET for log(v):{:.2f}m/s and log(G):{:.2f}K/m: {:.4f}".format(logv, logG, equiaxed_fraction))

            # if the last G value outputted a nan value (found the planar region) we can break out of the loop and go to the next v value
            if math.isnan(equiaxed_fraction):
                nan_points.append((10 ** logv, 10 ** logG))
                if not disable_output:
                    print("Found planar region. Moving to next set of solidification rates.")

                remaining_list = [x for x in logG_list if x > logG]
                for logG in remaining_list:
                    EF.append(float('nan'))
                    TR.append(float('nan'))
                    DTU.append(float('nan'))
                    v.append(10 ** logv)
                    G.append(10 ** logG)
                    nan_points.append((10 ** logv, 10 ** logG))
                break
    return v, G, EF, TR, DTU, nan_points

def get_CET_lines(
    system,
    g_range,
    solutes,
    primary_phase=DEFAULT_PRIMARY_PHASE,
    interfacial_energy=DEFAULT_INTERFACIAL_ENERGY,
    nb_nucleations_site=DEFAULT_CET_NUCLEATION_SITES,
    nucleation_undercooling=DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
    equiaxed_exponent=DEFAULT_CET_EQUIAXED_EXPONENT,
    disable_output=True,
    composition_unit=CompositionUnit.MOLE_FRACTION,
    r_range=(-7, 1),
    r_count=50,
):
    # Interfacial energy Universal interatomic machine learning potentials
    calc = system.with_property_model_calculation("CETPythonModel")
    if not disable_output:
        print('Calculator arguments:', list(calc.get_arguments()))
    calc.set_composition_unit(composition_unit)

    for element in solutes:
        calc.set_composition(element, solutes[element])

    logv_list = np.linspace(r_range[0], r_range[1], r_count)
    v = []
    G = [[],[],[],[]]
    calc.set_argument("primaryPhase", primary_phase)
    calc.set_argument("InterfacialEnergy", str(interfacial_energy))
    calc.set_argument("NbNucleationSites", str(nb_nucleations_site))
    calc.set_argument("NucleationUndercooling", str(nucleation_undercooling))
    calc.set_argument("EquiaxedExponent", str(equiaxed_exponent))
    calc.set_argument("Solve for", "Thermal gradient")
    calc.set_argument("SolidificationGrowthRate", "log10(v)")
    calc.set_argument("Equiaxed fractions", "0.001 0.99")

    for logv in logv_list:
        calc.set_argument("Log10(v)", logv)
        result = calc.calculate()
        equiaxed_fraction = result.get_value_of("Thermal gradient")
        G[0].append(equiaxed_fraction["Equiaxed fraction=0.001"])
        G[1].append(equiaxed_fraction["Equiaxed fraction=0.99"])
        G[2].append(equiaxed_fraction["Planar front"])
        v.append(10 ** logv)

    return v, G

# Basic code to generate G-V map. Adjusts lower v bound to capture the start of the planar line.
def adaptive_CET_grid(filename, thermodynamic_database, kinetic_database, g_range, show_plot = False, disable_cache = False):
    nb_nucleations_site = DEFAULT_CET_NUCLEATION_SITES
    nucleation_undercooling = DEFAULT_CET_NUCLEATION_UNDERCOOLING_K
    equiaxed_exponent = DEFAULT_CET_EQUIAXED_EXPONENT

    # Start TC-Python session.
    with TCPython() as session:

        # Activate caching (recommended).
        if disable_cache:
            session.disable_caching()
        else:
            session.set_cache_folder(_tc_python_cache_dir())

        # Read parameters from properties file.
        df = pd.read_csv(os.path.join(_tc_python_calcfiles_dir(), filename))

        # Get elements and process them.
        elements = {key: df[key][0] * 100 for key in df.keys() if "PROP" not in key}
        elements = {key: v for key, v in sorted(elements.items(), key=lambda item: item[1], reverse=True)}
        dependent_element = list(elements.keys())[0]
        solutes = elements.copy()
        solutes.pop(dependent_element)

        # Create thermodynamic system.
        system = session.select_thermodynamic_and_kinetic_databases_with_elements(thermodynamic_database, kinetic_database, list(elements.keys())).get_system()

        # Start calculating region borders suing simple thermal gradient prediction.
        calc = system.with_property_model_calculation("CETPythonModel")
        for element in solutes:
            calc.set_composition(element, solutes[element])

        # Define x and y on a log scale.
        logG_list = np.arange(g_range[0], g_range[1], 0.1)
        logG_list = np.append(logG_list, g_range[1])

        # Set CET model parameters.
        calc.set_argument("primaryPhase", df["PROP Primary Phase"][0])
        calc.set_argument("InterfacialEnergy", str(df["PROP Interface Energy (J/m2)"][0]))
        calc.set_argument("NbNucleationSites", str(nb_nucleations_site))
        calc.set_argument("NucleationUndercooling", str(nucleation_undercooling))
        calc.set_argument("EquiaxedExponent", str(equiaxed_exponent))
        calc.set_argument("Solve for", "Thermal gradient")
        calc.set_argument("SolidificationGrowthRate", "log10(v)")
        calc.set_argument("Equiaxed fractions", "0.001 0.99")

        # We want to find the v where the planar line is equal to the lower gradient bound.
        # Initial v guess.
        low_v = -6
        low_g = 10**g_range[0]
        above, below = False, False
        # Increment up and down until lower v is confirmed.
        ## Maybe Bayesian Optimization here sometime?
        while not (above and below):
            calc.set_argument("Log10(v)", low_v)
            result = calc.calculate()
            equiaxed_fraction = result.get_value_of("Thermal gradient")
            if equiaxed_fraction["Planar front"] < low_g:
                below = True
                if not above:
                    low_v += 0.1
            elif equiaxed_fraction["Planar front"] > low_g:
                above = True
                low_v -= 0.1

        # Make array of v values to check.
        logV_list = np.arange(low_v, 0, 0.1)
        logV_list = np.append(logV_list, 0)
        ### Testing
        print(f"logv_list: {logV_list}")

        # Now we calculate the equiaxed fraction at each point.
        # Switch what we are solving for to make the grid.
        calc.set_argument("Solve for", "Equiaxed fraction")
        G = []
        V = []
        EF = []
        CS = []

        # Begin loop to evaluate each grid point.
        for i in range(len(logV_list)):
            for logG in logG_list:
                V.append(10 ** logV_list[i])
                G.append(10 ** logG)
                calc.set_argument("Log10(v)", logV_list[i])
                calc.set_argument("TemperatureGradient", 10 ** logG)

                result = calc.calculate()
                equiaxed_fraction = result.get_value_of("fractionEquiaxedGrains")

                EF.append(equiaxed_fraction)
                CS.append(0)

                # If the last G value output a nan value (found the planar region) we can break out of the loop and go to the next v value.
                if math.isnan(equiaxed_fraction):

                    remaining_list = [x for x in logG_list if x > logG]
                    for logG in remaining_list:
                        EF.append(float('nan'))
                        CS.append(3)
                        V.append(10 ** logV_list[i])
                        G.append(10 ** logG)
                    break

        EF = [-1 if math.isnan(value) else value for value in EF]

        if show_plot or save_path:
            norm = colors.Normalize(vmin=-1.1, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic_r)  # Choose any matplotlib colormap

            fig, axs = plt.subplots(1, 1)
            grid = axs.scatter(solidification_rate_CET, thermal_gradient_CET, c=[mapper.to_rgba(val) for val in equiaxed_fractions_CET], marker='s', s=90)
            axs.plot(solidification_rate_CET, lines[0],color='limegreen', linewidth=3)
            axs.plot(solidification_rate_CET, lines[1],color='orange', linewidth=3)
            axs.plot(solidification_rate_CET, lines[2],color='dodgerblue', linewidth=3)
            axs.set_ylim(5*10**3,1.8*10**9)
            axs.set_xscale('log')
            axs.set_yscale('log')
            axs.set_xlabel('Solidification Rate, R (m/s)')
            axs.set_ylabel('Thermal Gradient, G (K/m)')
            fig.colorbar(mapper, ax=axs, label='Equiaxed Fraction')

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
        return (np.array(solidification_rate_CET), np.array(thermal_gradient_CET), np.array(equiaxed_fractions_CET))


def request_GR_Grid_from_excel(
    excel_path,
    row_index,
    element_cols,
    thermodynamic_database,
    kinetic_database,
    primary_phase=DEFAULT_PRIMARY_PHASE,
    interfacial_energy=None,
    nb_nucleations_site=DEFAULT_CET_NUCLEATION_SITES,
    nucleation_undercooling=DEFAULT_CET_NUCLEATION_UNDERCOOLING_K,
    equiaxed_exponent=DEFAULT_CET_EQUIAXED_EXPONENT,
    show_plot=False,
    disable_cache=False,
    disable_output=True,
    save_path=None,
    overlay_csv=None,
    overlay_label=None,
    overlay_kwargs=None,
    legend_loc="upper center",
    show_cet_lines=False,
    report_interfacial_energy=True,
    return_interfacial_energy=False,
):
    elements, solutes = _load_alloy_from_excel(excel_path, row_index, element_cols)
    liquidus_temperature = _load_liquidus_from_excel(excel_path, row_index)
    spreadsheet_interfacial_energy = None
    if interfacial_energy is None:
        spreadsheet_interfacial_energy = _load_interfacial_energy_from_excel(
            excel_path,
            row_index,
        )
        interfacial_energy = spreadsheet_interfacial_energy
        if report_interfacial_energy and interfacial_energy is not None:
            print(
                "Using spreadsheet interfacial energy: "
                f"{interfacial_energy:.6g} J/m^2"
            )

    with TCPython() as session:
        if disable_cache:
            session.disable_caching()
        else:
            session.set_cache_folder(_tc_python_cache_dir())

        system = session.select_thermodynamic_and_kinetic_databases_with_elements(
            thermodynamic_database, kinetic_database, list(elements.keys())
        ).get_system()
        resolved_interfacial_energy = resolve_interfacial_energy(
            system,
            solutes,
            primary_phase,
            liquidus_temperature,
            interfacial_energy=interfacial_energy,
            composition_unit=CompositionUnit.MOLE_FRACTION,
            report=(
                report_interfacial_energy
                and spreadsheet_interfacial_energy is None
            ),
        )

        grid_metadata = _cet_grid_cache_metadata(
            thermodynamic_database=thermodynamic_database,
            kinetic_database=kinetic_database,
            element_names=elements.keys(),
            solutes=solutes,
            primary_phase=primary_phase,
            interfacial_energy=resolved_interfacial_energy,
            nb_nucleations_site=nb_nucleations_site,
            nucleation_undercooling=nucleation_undercooling,
            equiaxed_exponent=equiaxed_exponent,
            composition_unit=CompositionUnit.MOLE_FRACTION,
        )
        grid_npz_path = _cet_grid_npz_path(grid_metadata)
        cached_grid = None if disable_cache else _load_cet_grid_npz(grid_npz_path)

        if cached_grid is not None:
            (
                solidification_rate_CET,
                thermal_gradient_CET,
                equiaxed_fractions_CET,
                tip_radius_CET,
                dendrite_tip_undercooling_CET,
                cached_nan_points,
            ) = cached_grid
            nan_points = [tuple(point) for point in cached_nan_points]
            print(f"Loaded CET grid NPZ: {grid_npz_path}")
        else:
            (
                solidification_rate_CET,
                thermal_gradient_CET,
                equiaxed_fractions_CET,
                tip_radius_CET,
                dendrite_tip_undercooling_CET,
                nan_points,
            ) = get_CET_grid(
                system,
                solutes,
                primary_phase=primary_phase,
                interfacial_energy=resolved_interfacial_energy,
                nb_nucleations_site=nb_nucleations_site,
                nucleation_undercooling=nucleation_undercooling,
                equiaxed_exponent=equiaxed_exponent,
                disable_output=disable_output,
                composition_unit=CompositionUnit.MOLE_FRACTION,
            )
            if not disable_cache:
                _save_cet_grid_npz(
                    grid_npz_path,
                    grid_metadata,
                    solidification_rate_CET,
                    thermal_gradient_CET,
                    equiaxed_fractions_CET,
                    tip_radius_CET,
                    dendrite_tip_undercooling_CET,
                    nan_points,
                )
                print(f"Wrote CET grid NPZ: {grid_npz_path}")

        line_r, lines = get_CET_lines(
            system,
            (3, 10),
            solutes,
            primary_phase=primary_phase,
            interfacial_energy=resolved_interfacial_energy,
            nb_nucleations_site=nb_nucleations_site,
            nucleation_undercooling=nucleation_undercooling,
            equiaxed_exponent=equiaxed_exponent,
            disable_output=disable_output,
        )

        for i, value in enumerate(equiaxed_fractions_CET):
            if math.isnan(value):
                equiaxed_fractions_CET[i] = float(-1)
        for i, value in enumerate(tip_radius_CET):
            if math.isnan(value):
                tip_radius_CET[i] = float(-1)
        for i, value in enumerate(dendrite_tip_undercooling_CET):
            if math.isnan(value):
                dendrite_tip_undercooling_CET[i] = float(-1)

        if show_plot or save_path:
            norm = colors.Normalize(vmin=-1.1, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic_r)

            fig, axs = plt.subplots(1, 1)
            axs.scatter(
                solidification_rate_CET,
                thermal_gradient_CET,
                c=[mapper.to_rgba(val) for val in equiaxed_fractions_CET],
                marker="s",
                s=90,
            )
            if show_cet_lines:
                axs.plot(line_r, lines[0], color="limegreen", linewidth=3)
                axs.plot(line_r, lines[1], color="orange", linewidth=3)
                axs.plot(line_r, lines[2], color="dodgerblue", linewidth=3)
            #axs.set_ylim(5*10**3, 1.8*10**9)
            axs.set_xlim(1e-7, 1e1)
            axs.set_ylim(1e3, 1e10)
            axs.set_xscale("log")
            axs.set_yscale("log")
            axs.set_xlabel("Solidification Rate, R (m/s)")
            axs.set_ylabel("Thermal Gradient, G (K/m)")
            fig.colorbar(mapper, ax=axs, label="Equiaxed Fraction")

            if overlay_csv:
                overlay_csvs = list(overlay_csv) if isinstance(overlay_csv, (list, tuple)) else [overlay_csv]
                overlay_labels = _as_overlay_list(overlay_label, len(overlay_csvs), "overlay_label")
                overlay_kwarg_sets = _as_overlay_list(overlay_kwargs, len(overlay_csvs), "overlay_kwargs")
                for csv_path, label, kwargs in zip(overlay_csvs, overlay_labels, overlay_kwarg_sets):
                    _plot_overlay_points(
                        axs,
                        overlay_csv=csv_path,
                        label=label,
                        point_kwargs=kwargs,
                    )
                legend = axs.legend(loc=legend_loc)
                _style_overlay_legend(legend)

            if nan_points:
                nan_r, nan_g = np.asarray(nan_points).T
                print(
                    "fractionEquiaxedGrains returned NaN at "
                    f"{len(nan_points)} grid points. "
                    f"R range: {nan_r.min():.3e} to {nan_r.max():.3e} m/s, "
                    f"G range: {nan_g.min():.3e} to {nan_g.max():.3e} K/m"
                )

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
        outputs = (
            np.array(solidification_rate_CET),
            np.array(thermal_gradient_CET),
            np.array(equiaxed_fractions_CET),
        )
        if return_interfacial_energy:
            return (*outputs, resolved_interfacial_energy)
        return outputs
