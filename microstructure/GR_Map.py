from math import log10

from tc_python import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
import os

def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _tc_python_cache_dir():
    return os.path.join(_repo_root(), "TC_Python", "caches", "GR_Map_cache")


def _tc_python_calcfiles_dir():
    return os.path.join(_repo_root(), "TC_Python", "CalcFiles")


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


def get_CET_grid(system, solutes, primary_phase="BCC_B2", interfacial_energy=0.5, nb_nucleations_site=2e15, nucleation_undercooling=2.5, equiaxed_exponent=3.4, disable_output = True, composition_unit=CompositionUnit.MOLE_FRACTION):
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
    primary_phase="BCC_B2",
    interfacial_energy=0.5,
    nb_nucleations_site=2e15,
    nucleation_undercooling=2.5, # default
    equiaxed_exponent=3.4, # default
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
    # Temporary defaults until we find a way to get these parameters.
    nb_nucleations_site = 2e15
    nucleation_undercooling = 2.5
    equiaxed_exponent = 3.4

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
    primary_phase="BCC_B2",
    interfacial_energy=0.5,
    show_plot=False,
    disable_cache=False,
    disable_output=True,
    save_path=None,
    overlay_csv=None,
    overlay_label=None,
    overlay_kwargs=None,
    legend_loc="upper center",
    show_cet_lines=False,
):
    elements, solutes = _load_alloy_from_excel(excel_path, row_index, element_cols)

    with TCPython() as session:
        if disable_cache:
            session.disable_caching()
        else:
            session.set_cache_folder(_tc_python_cache_dir())

        system = session.select_thermodynamic_and_kinetic_databases_with_elements(
            thermodynamic_database, kinetic_database, list(elements.keys())
        ).get_system()

        solidification_rate_CET, thermal_gradient_CET, equiaxed_fractions_CET, tip_radius_CET, dendrite_tip_undercooling_CET, nan_points = get_CET_grid(
            system,
            solutes,
            primary_phase=primary_phase,
            interfacial_energy=interfacial_energy,
            nb_nucleations_site=4.0E11,
            nucleation_undercooling=4.0,
            equiaxed_exponent=3.13,
            disable_output=disable_output,
        )

        line_r, lines = get_CET_lines(
            system,
            (3, 10),
            solutes,
            primary_phase=primary_phase,
            interfacial_energy=interfacial_energy,
            nb_nucleations_site=4.0E11,
            nucleation_undercooling=4.0,
            equiaxed_exponent=3.13,
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
        return (np.array(solidification_rate_CET), np.array(thermal_gradient_CET), np.array(equiaxed_fractions_CET))
