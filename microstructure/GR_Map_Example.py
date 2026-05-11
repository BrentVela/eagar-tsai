from tc_python import *
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp

def get_map(elements: dict[str, float], liquidus: float, thermodynamic_database: str, kinetic_database: str, v_range = (-6, 0), g_range = (4, 8), primary_phase: str = 'BCC', resolution: int  = 6):
    with TCPython() as session:
        session.disable_caching()

        # Get rid of disable caching and enable this for faster times when running the same system again.
        # session.set_cache_folder(f"TC_Python_cache")

        # Create thermodynamic system.
        system = session.select_thermodynamic_and_kinetic_databases_with_elements(thermodynamic_database, kinetic_database, list(elements.keys())).get_system()

        # Create calculators.
        cet_calc = system.with_property_model_calculation("CETPythonModel").set_composition_unit(CompositionUnit.MASS_FRACTION)
        interface_calc = (system.with_property_model_calculation("Interfacial Energy").set_composition_unit(CompositionUnit.MASS_FRACTION))
        for element in list(elements)[:-1]:
            cet_calc.set_composition(element, elements[element])
            interface_calc.set_composition(element, elements[element])

        # Use broken-bond model to approximate interface energy.
        interface_calc = interface_calc.set_temperature(liquidus - 1)
        interface_result = (
            interface_calc.set_argument("matrix", "LIQUID").
            set_argument("precipitate", primary_phase).
            calculate()
        )
        interface_energy = interface_result.get_value_of("interfacialEnergy")

        # Set CET model parameters.
        cet_calc.set_argument("primaryPhase", primary_phase)
        cet_calc.set_argument("InterfacialEnergy", interface_energy)
        cet_calc.set_argument("Solve for", "Equiaxed fraction")

        # Optional. These determine equiaxed fractions but require data found in literature.
        # calc.set_argument("NbNucleationSites", nb_nucleations_site)
        # calc.set_argument("NucleationUndercooling", nucleation_undercooling)
        # calc.set_argument("EquiaxedExponent", equiaxed_exponent)

        # Get range of values to evaluate
        logV_list = np.linspace(v_range[0], v_range[1], 2**resolution)
        logG_list = np.linspace(g_range[0], g_range[1], 2**resolution)
        EF = [[-1 for _ in range(len(logG_list))] for _ in range(len(logV_list))]

        # Begin loop to evaluate each grid point.
        for j in range(len(logV_list)):
            for i in range(len(logG_list)):
                # Set new arguments
                cet_calc.set_argument("Log10(v)", logV_list[j])
                cet_calc.set_argument("TemperatureGradient", 10 ** logG_list[i])

                # Calculate point
                result = cet_calc.calculate()
                equiaxed_fraction = result.get_value_of("fractionEquiaxedGrains")
                EF[j][i] = equiaxed_fraction

                # If the last G value output a nan value (found the planar region) we can break out of the loop.
                if math.isnan(equiaxed_fraction):
                    break

        # Set planar regions to -1
        EF = [[-1 if math.isnan(value) else value for value in EF_column] for EF_column in EF]


        return 10 ** logV_list, 10 ** logG_list, np.array(EF)



# Define parameters:
# Elements by weight fraction.
elements = {
    'Ni': 0.6,
    'Cu': 0.4,
}
# Thermo-Calc thermodynamic system.
thermo_database = "TCHEA8"
kinetic_database = "MOBHEA3"
# Liquidus temperature (K).
liquidus = 1620
# Velocity and gradient ranges in log10 scale.
v_range = (-6, 0)
g_range = (4, 8)
# Primary phase of solidification. Typical values are BCC_B2 BCC_A2 FCC_L12
primary_phase = 'FCC_L12'
# Map resolution.
resolution = 5

# Run Calculation
velocity_values, gradient_values, fraction_values = get_map(elements, liquidus, thermo_database, kinetic_database, v_range, g_range, primary_phase, resolution)

# Create SciPy grid interpolator
interpolator = sp.interpolate.RegularGridInterpolator((velocity_values, gradient_values), fraction_values, fill_value=-2, bounds_error=False)

# Interpolate data.
example_velocities = [1e-2, 10**-2.5, 1e-3]
example_gradients = [1e5, 10**5.5, 1e6]
equiaxed_fractions = interpolator(np.column_stack([example_velocities, example_gradients]))
print(f"Equiaxed Fractions: {equiaxed_fractions}")

# Create G-V matplotlib map
fig, axs = plt.subplots()
x, y = np.meshgrid(velocity_values, gradient_values, indexing="ij")
axs.pcolormesh(x, y, fraction_values, cmap="seismic_r", vmin=-1.1, vmax=1, shading="auto")
axs.scatter(example_velocities, example_gradients)
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlabel('Solidification Rate, R (m/s)')
axs.set_ylabel('Thermal Gradient, G (K/m)')
mapper = cm.ScalarMappable(norm=colors.Normalize(vmin=-1.1, vmax=1, clip=True), cmap="seismic_r") # Choose any matplotlib colormap
fig.colorbar(mapper, ax=axs, label='Equiaxed Fraction')
plt.show()