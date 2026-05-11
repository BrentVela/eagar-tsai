from tc_python import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.colors as colors
import matplotlib.cm as cm
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn_extra.cluster import KMedoids
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import itertools

def get_material_properties_from_scheil(system, solutes):
    scheil_options = (ScheilOptions().calculate_from_start_temperature()
                      .calculate_to_temperature_below_solidus()
                      .enable_evaporation_property_calculation())

    scheil_calculator = (system.with_scheil_calculation()
                         .with_options(scheil_options)
                         .set_start_temperature(5000.0)
                         .set_composition_unit(CompositionUnit.MASS_PERCENT))

    for element in solutes:
        scheil_calculator.set_composition(element, solutes[element])

    scheil_result = scheil_calculator.calculate()
    return MaterialProperties.from_scheil_result(scheil_result)


def get_CET_grid(system, solutes, primary_phase="FCC_L12", interfacial_energy=0.5, nb_nucleations_site=2e15, nucleation_undercooling=2.5, equiaxed_exponent=3.4):
    calc = system.with_property_model_calculation("CETPythonModel")
    print('Calculator arguments:', list(calc.get_arguments()))
    calc.set_composition_unit(CompositionUnit.MASS_PERCENT)

    for element in solutes:
        calc.set_composition(element, solutes[element])

    logv_list = np.linspace(-6,0,50)
    logG_list = np.linspace(4,9,50)
    v = []
    G = []
    EF = []
    calc.set_argument("primaryPhase", primary_phase)
    calc.set_argument("InterfacialEnergy", str(interfacial_energy))
    calc.set_argument("NbNucleationSites", str(nb_nucleations_site))
    calc.set_argument("NucleationUndercooling", str(nucleation_undercooling))
    calc.set_argument("EquiaxedExponent", str(equiaxed_exponent))
    calc.set_argument("Solve for", "Equiaxed fraction")

    for logv in logv_list:
        for logG in logG_list:
            calc.set_argument("Log10(v)", logv)
            calc.set_argument("TemperatureGradient", 10**logG)
            
            result = calc.calculate()
            equiaxed_fraction = result.get_value_of("fractionEquiaxedGrains")
            
            EF.append(equiaxed_fraction)
            v.append(10**logv)
            G.append(10**logG)
            print("CET for log(v):{:.2f}m/s and log(G):{:.2f}K/m: {:.4f}".format(logv, logG, equiaxed_fraction))
    
            # if the last G value outputted a nan value (found the planar region) we can break out of the loop and go to the next v value
            if math.isnan(equiaxed_fraction):
                print("Found planar region. Moving to next set of solidification rates.")
                
                remaining_list = [x for x in logG_list if x > logG]
                for logG in remaining_list:
                    EF.append(float('nan'))
                    v.append(10**logv)
                    G.append(10**logG)
                break
    return v, G, EF

with TCPython() as start:
    start.set_cache_folder(os.path.basename(__file__) + "_cache")

    thermodynamic_database = "TCHEA7"
    kinetic_database = "MOBHEA3"
    dependent_element = "Cu"
    solutes = {"Ni": 30}
    scan_speed = 1.0 #m/s

    elements = [dependent_element] + list(solutes.keys())
    system = start.select_thermodynamic_and_kinetic_databases_with_elements(thermodynamic_database, kinetic_database,  elements).get_system()

    solidification_rate_CET, thermal_gradient_CET, equiaxed_fractions_CET = get_CET_grid(system, solutes, primary_phase="FCC_L12", interfacial_energy=0.5, nb_nucleations_site=4.0E11,
                                                                   nucleation_undercooling=4.0, equiaxed_exponent=3.13)
    
    # Find nan values and replace with -1. This is where the CET model found planar growth
    for i, value in enumerate(equiaxed_fractions_CET):
        if math.isnan(value):
            equiaxed_fractions_CET [i] = float(-1) 
    
    
    
    # solidification_rate_CET = [x for i,x in enumerate(solidification_rate_CET) if i in not_nan_indices]
    # thermal_gradient_CET = [x for i,x in enumerate(thermal_gradient_CET) if i in not_nan_indices]
    
    norm = colors.Normalize(vmin=-1.1, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic_r)  # Choose any matplotlib colormap
    
    fig, axs = plt.subplots(1, 1)
    grid = axs.scatter(solidification_rate_CET, thermal_gradient_CET, c=[mapper.to_rgba(val) for val in equiaxed_fractions_CET], cmap="seismic_r", marker='s', s=90)
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Solidification Rate, R (m/s)')
    axs.set_ylabel('Thermal Gradient, G (K/m)')
    fig.colorbar(mapper, ax=axs, label='Equiaxed Fraction')
    fig.savefig('CET_GR_grid_Cu70Ni30.png', dpi=600, bbox_inches='tight')
    
    
    #%%
    # Create X values for training
    X = np.log10(np.array([solidification_rate_CET, thermal_gradient_CET]))
    X = np.transpose(X)
    
    y = equiaxed_fractions_CET
    
    # Define kernel
    # kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=.5, length_scale_bounds=(.1,2)) + WhiteKernel(noise_level=1e-2)
    kernel = Matern(length_scale=2.0, nu=0.5, length_scale_bounds=(0.1, 10))
    
    # Train GPR
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
    gpr.fit(X, y)
    
    # Training results
    y_train = np.array(y)
    y_pred = gpr.predict(X, return_std=False)
    
    # Error metrics
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    # mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100  # Only valid if y_train has no zeros
    
    # Print metrics
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    # print(f"MAPE: {mape:.2f}%")
    
    # Parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_pred, edgecolor='k', alpha=0.7)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot with Error Metrics')
    plt.grid(True)
    plt.axis('equal')
    
    # Add metrics as text to the plot
    metrics_text = (
        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        # f"MAPE = {mape:.1f}%"
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig('CET_GPR_parity_Cu70Ni30.png', dpi=600, bbox_inches='tight')
    
    
    # Prediction grid (also log scale)
    R_range = np.linspace(-6, 0, 250)
    G_range = np.linspace(4, 9, 250)
    X_pred_log = np.array(list(itertools.product(R_range, G_range)))
    y_pred, sigma = gpr.predict(X_pred_log, return_std=True)
    
    # Convert X axis back from log to linear for plotting
    X_pred_lin = 10**X_pred_log
    
    # Plot predictions
    fig, axs = plt.subplots(1, 1)
    grid = axs.scatter(X_pred_lin[:,0], X_pred_lin[:,1], c=mapper.to_rgba(y_pred) , cmap="seismic_r", marker='s', s=10)
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Solidification Rate, R (m/s)')
    axs.set_ylabel('Thermal Gradient, G (K/m)')
    fig.colorbar(mapper, ax=axs, label='Equiaxed Fraction')
    fig.savefig('CET_GR_GPR_Cu70Ni30.png', dpi=600, bbox_inches='tight')

    

    # mp = get_material_properties_from_scheil(system, solutes)

    # am_calc = (start.with_additive_manufacturing()
    #              .with_steady_state_calculation()
    #              .with_heat_source(HeatSource.gaussian_with_constant_absorptivity().with_keyhole_model(KeyholeModel())
    #                                          .set_beam_radius(45.0e-6)
    #                                          .set_absorptivity(36.0)
    #                                          .set_power(250)
    #                                          .set_scanning_speed(scan_speed))
    #              .with_material_properties(mp)
    #         .enable_fluid_flow_marangoni())

    # am_result = am_calc.calculate()

    # thermal_gradient_AM, solidification_rate_AM, x, y, z = am_result.get_thermal_gradient_and_solidification_rate()
    # thermal_gradient_AM_melting, melting_rate_AM, xm, ym, zm = am_result.get_thermal_gradient_and_melting_rate()

    # fig, axs = plt.subplots(1, 1)
    # axs.title.set_text("Columnar to Equiaxed Transition for IN718\n Thermal gradient and solidification rate from AM Steady-state simulation")
    # for label, thermal_gradient_CET in thermal_gradients_CET.items():
    #     axs.loglog(solidification_rate_CET, thermal_gradient_CET, label=label)

    # axs.loglog(solidification_rate_AM, thermal_gradient_AM, label='Thermal gradient and solidification rate from AM Steady-state simulation', marker='o', lw=0)
    # axs.set_xlabel('Solidification rate (m/s)')
    # axs.set_ylabel('Thermal Gradient (K/s) ')
    # axs.legend()

    # fig2 = plt.figure()
    # fig2.suptitle('From AM steady-state')
    # ax1 = fig2.add_subplot(121, projection='3d')
    # p3d1 = ax1.scatter(x, y, z, s=30, c=solidification_rate_AM, marker='o', cmap='jet')
    # ax1.title.set_text('Solidification rate(m/s)')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    # fig2.colorbar(p3d1, ax=ax1)

    # ax2 = fig2.add_subplot(122, projection='3d')
    # p3d2 = ax2.scatter(x, y, z, s=30, c=thermal_gradient_AM, marker='o', cmap='jet')
    # ax2.title.set_text('Thermal gradient (K/m)')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    # fig2.colorbar(p3d2, ax=ax2)

    # fig3 = plt.figure()
    # fig3.suptitle('From AM steady-state')
    # ax1 = fig3.add_subplot(121, projection='3d')
    # p3d1 = ax1.scatter(xm, ym, zm, s=30, c=melting_rate_AM, marker='o', cmap='jet')
    # ax1.title.set_text('Melting rate(m/s)')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    # fig3.colorbar(p3d1, ax=ax1)

    # ax2 = fig3.add_subplot(122, projection='3d')
    # p3d2 = ax2.scatter(xm, ym, zm, s=30, c=thermal_gradient_AM_melting, marker='o', cmap='jet')
    # ax2.title.set_text('Thermal gradient (K/m)')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    # fig3.colorbar(p3d2, ax=ax2)

    plt.show()
