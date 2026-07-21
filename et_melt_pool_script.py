# Import the required modules
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import *
import scipy.version
import ctypes
import time
from itertools import compress
import os.path as path
import concurrent.futures

#import resource
#import gc
#from pympler import muppy,summary

def _get_libpath():
    """Return the best available shared-library path for the integrand."""
    sciVer = scipy.version.version.split(".")
    osType = sys.platform

    if int(sciVer[1]) >= 1:
        if osType == 'darwin':   # macOS
            return 'libeagar_tsai_integrand.dylib'
        if osType.startswith('linux'):   # Linux
            return './libeagar_tsai_integrand.so'
        if (osType == 'win32') or (osType == 'cygwin') or (osType == 'msys'):
            return 'libeagar_tsai_integrand.dll'
    return 0


def _load_integrand(libpath):
    """Return the compiled integrand when available, otherwise the Python fallback."""
    if libpath == 0 or (isinstance(libpath, str) and not path.exists(libpath)):
        return eagar_tsai_integrand

    lib = ctypes.CDLL(libpath)
    func = lib.eagar_tsai_integrand
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_double)
    return func


def _build_coordinate_ranges(two_sigma, domain_m, spatial_res_m):
    """Build the x, y, and z coordinate ranges in metres."""
    xMin = round(-1.0 * two_sigma * 1.5, 5)
    xMax = domain_m[0]
    yMin = 0.0
    yMax = domain_m[1]
    zMin = -1.0 * domain_m[2]
    zMax = 0.0

    nx = int(np.round(abs(xMax - xMin) / spatial_res_m)) + 1
    ny = int(np.round(abs(yMax - yMin) / spatial_res_m)) + 1
    nz = int(np.round(abs(zMax - zMin) / spatial_res_m)) + 1

    return (
        np.linspace(xMin, xMax, nx),
        np.linspace(yMin, yMax, ny),
        np.linspace(zMin, zMax, nz),
    )


def _build_coordinate_ranges_from_limits(x_limits_m, y_limits_m, z_limits_m, spatial_res_m):
    """Build coordinate ranges from explicit axis limits in metres."""
    xMin, xMax = x_limits_m
    yMin, yMax = y_limits_m
    zMin, zMax = z_limits_m

    nx = int(np.round(abs(xMax - xMin) / spatial_res_m)) + 1
    ny = int(np.round(abs(yMax - yMin) / spatial_res_m)) + 1
    nz = int(np.round(abs(zMax - zMin) / spatial_res_m)) + 1

    return (
        np.linspace(xMin, xMax, nx),
        np.linspace(yMin, yMax, ny),
        np.linspace(zMin, zMax, nz),
    )


def _volume_limits_from_bounds(bounds, beam_two_sigma_m, spatial_res_um, padding_um=(100.0, 100.0, 50.0)):
    """Return cropped volume limits around the melt pool or beam if no melt pool exists."""
    pad_x = float(padding_um[0]) * 1.0e-6
    y_max = 200.0e-6
    z_extra = 5.0 * float(spatial_res_um) * 1.0e-6

    if bounds is None:
        half_span = max(2.0 * beam_two_sigma_m, 100.0e-6)
        return (
            (-half_span, half_span),
            (0.0, y_max),
            (-half_span, 0.0),
        )

    return (
        (bounds['x_min_m'] - pad_x, bounds['x_max_m'] + pad_x),
        (0.0, y_max),
        (bounds['z_min_m'] - z_extra, 0.0),
    )


def _export_temperature_volume_pyvista(nxrange, nyrange, nzrange, tvolume, out_dir, row_index):
    """
    Save the temperature volume as a PyVista-readable structured image grid.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: PyVista is not installed; skipping VTI export.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    x_um = nxrange * 1.0e6
    y_um = nyrange * 1.0e6
    z_um = nzrange * 1.0e6

    if x_um.size > 1:
        dx = float(x_um[1] - x_um[0])
    else:
        dx = 1.0
    if y_um.size > 1:
        dy = float(y_um[1] - y_um[0])
    else:
        dy = 1.0
    if z_um.size > 1:
        dz = float(z_um[1] - z_um[0])
    else:
        dz = 1.0

    grid = pv.ImageData(
        dimensions=(x_um.size, y_um.size, z_um.size),
        spacing=(dx, dy, dz),
        origin=(float(x_um[0]), float(y_um[0]), float(z_um[0])),
    )
    grid.point_data["Temperature_K"] = np.ascontiguousarray(tvolume).ravel(order="F")

    vti_path = path.join(out_dir, f'ET_3D_temperature_alloy{row_index}.vti')
    grid.save(vti_path)
    return vti_path


def _run_chunk(params):
    (
        chunk_num,
        data,
        domain_um,
        spatial_res_um,
        out_dir,
        heatmap_rows_set,
        heatmap_dir,
        volume_padding_um,
    ) = params
    data = data.reset_index(drop=True)

    # ========== SIMULATION ====================================================
    domain = np.array(domain_um, dtype='f8')   # microns, expanded if needed
    spatialRes = float(spatial_res_um)         # microns
    # ==========================================================================

    # ========== BEAM and MATERIAL =============================================
    beam1, mat1 = beamFromCSV(data)
    # ==========================================================================

    integrationWarning()

    runSize = np.size(beam1.v, axis=0)
    for i in np.arange(runSize):
        sim1 = simParam(domain, spatialRes)
        try:
            (
                data.at[i, 'melt_length'],
                data.at[i, 'melt_width'],
                data.at[i, 'melt_depth'],
                data.at[i, 'peakT'],
                data.at[i, 'minT'],
                melt_bounds,
            ) = eagarTsaiParam(beam1, mat1, sim1, i)

            orig_idx = int(data.at[i, '__orig_index'])
            if heatmap_rows_set is not None and orig_idx in heatmap_rows_set:
                nxrange_xy, nyrange_xy, tplanexy = temp_xy_from_row(
                    data.iloc[[i]],
                    domain_um=domain_um,
                    spatial_res_um=spatial_res_um,
                )
                save_xy_heatmap(
                    nxrange_xy,
                    nyrange_xy,
                    tplanexy,
                    out_dir=heatmap_dir,
                    row_index=orig_idx,
                    png_name=f'ET_xy_heatmap_alloy{orig_idx}.png',
                    xlim_um=(-120.0, 600.0), #For alloy 0: -120, 600
                    ylim_um=(-200.0, 200.0), #For alloy 0: -200, 200
                    t_melt=mat1.tMelt[i],
                    xlabel='x (travel direction, um)',
                    ylabel='y (transverse direction, um)',
                    cmap='inferno',
                    vmin=0.0,
                    vmax=12000.0, #Temperature limit
                )
                x_limits_m, y_limits_m, z_limits_m = _volume_limits_from_bounds(
                    melt_bounds,
                    beam1.twoSigma[i],
                    spatial_res_um,
                    padding_um=volume_padding_um,
                )
                nxrange_v, nyrange_v, nzrange_v, tvolume = temp_volume_from_row(
                    data.iloc[[i]],
                    x_limits_m=x_limits_m,
                    y_limits_m=y_limits_m,
                    z_limits_m=z_limits_m,
                    spatial_res_um=spatial_res_um,
                )
                save_temperature_volume_csv(
                    nxrange_v,
                    nyrange_v,
                    nzrange_v,
                    tvolume,
                    out_dir=heatmap_dir,
                    row_index=orig_idx,
                )
                _export_temperature_volume_pyvista(
                    nxrange_v,
                    nyrange_v,
                    nzrange_v,
                    tvolume,
                    out_dir=heatmap_dir,
                    row_index=orig_idx,
                )
        except Exception as exc:
            print(f'Error in chunk {chunk_num}: {exc}')

    # Add micron columns for convenience (meters -> microns)
    data['melt_length_um'] = data['melt_length'] * 1.0e6
    data['melt_width_um'] = data['melt_width'] * 1.0e6
    data['melt_depth_um'] = data['melt_depth'] * 1.0e6

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        data.to_csv(path.join(out_dir, f'ET_v3_OUT_{chunk_num}_alloy0.csv'), index=False)

    return data


def compute_melt_pool(
    data,
    domain_um=(1200.0, 1200.0, 1000.0),
    spatial_res_um=1.0,
    chunk_size=1,
    workers=None,
    out_dir=None,
    heatmap_rows=None,
    heatmap_dir=None,
    volume_padding_um=(100.0, 100.0, 50.0),
):
    """
    Compute melt pool dimensions for each row in a DataFrame.

    Required columns:
      Velocity_m/s, Power, Beam_diameter_m, Absorptivity,
      T_liquidus, thermal_cond_liq, Density_kg/m3, Cp_J/kg

    If ``heatmap_rows`` is provided, the selected rows also get a cropped 3D
    temperature CSV in ``x, y, z, T`` format, a PyVista ``.vti`` grid, and
    the full-range z=0 heatmap PNG.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    data = data.copy()
    data['__orig_index'] = data.index

    heatmap_rows_set = None
    if heatmap_rows is not None:
        heatmap_rows_set = set(int(x) for x in heatmap_rows)
        if heatmap_dir is None:
            heatmap_dir = out_dir if out_dir is not None else 'CalcFiles'

    chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]
    params = [
        (
            i,
            chunk,
            domain_um,
            spatial_res_um,
            out_dir,
            heatmap_rows_set,
            heatmap_dir,
            volume_padding_um,
        )
        for i, chunk in enumerate(chunks)
    ]


    if workers is None or workers <= 1:
        results = [_run_chunk(p) for p in params]
    else:
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            results = list(executor.map(_run_chunk, params))

    return pd.concat(results, ignore_index=True)


def beamFromCSV(data):
    """ Import beam parameters from csv file """
    #data = pd.read_csv(inputfile,header=0)

	# These we will iterate through
#    v = data['Velocity_m/s'] # m/s
#    P = data['P [W]'] # Watts
#    twoSigma = data["Beam_diameter_m"]  #m # meters
#    A = data['Absorptivity'] # >0, <1
#    tMelt = data['PROP LT (K)']
#    k = data['Prop Liquidus Thermal conductivity (W/(mK))']
#    rho = data["Prop RT Density (kg/m3)"]
#    cp = data['eff_Cp_JkgK']
    
    v = data['Velocity_m/s'] # m/s
    P = data['Power'] # Watts
    twoSigma = data["Beam_diameter_m"]  #m # meters
    A = data['Absorptivity'] # >0, <1
    tMelt = data['T_liquidus']
    #tSolidus = data['T_solidus']
    k = data['thermal_cond_liq']
    rho = data["Density_kg/m3"]
    cp = data['Cp_J/kg']
    
    # Create the beam object
    return beam(twoSigma,P,v,A), material(tMelt,k,rho,cp)

def eagarTsaiParam(beam,material,simParam,i):
    """ Function that runs each iteration of the EagarTsai Simulation """
    # This version of the code uses a re-formulation of EagarTsai
    # By Sasha Rubenchik - LLNL 2015
    
    # Unpack a few variables
    # Material
    #print(material.tMelt[:])
    tMelt = material.tMelt[i]
    k = material.k[i]
    rho = material.rho[i]
    cp = material.cp[i]
    alpha = k/(rho*cp)
    
    # Beam (this varies each run)
    P = beam.P[i]
    A = beam.A[i]
    v = beam.v[i]
    sigma = beam.sigma[i]
    
    # Simulation params
    delta = simParam.spatialRes
    nxrange, nyrange, nzrange = _build_coordinate_ranges(
        beam.twoSigma[i],
        simParam.domain,
        delta,
    )
    nx = nxrange.size
    ny = nyrange.size
    nz = nzrange.size
    xMax = simParam.domain[0]
    yMin = 0.0
    yMax = simParam.domain[1]
    zMin = -1.0 * simParam.domain[2]
    zMax = 0.0
    
    libpath = _get_libpath()

    # Run integral
    tplanexy, tplanexz = runIntegrate(nxrange,nyrange,nzrange,nx,ny,nz,alpha,sigma,k,v,A,P,libpath)

    # Find the peak temperature
    peakT = np.amax(tplanexy)
    minT = np.amin(tplanexy)
    # Now check to see if the peak temperature is hotter than Tmelt
    if peakT > tMelt:
        # Then there is a melt pool, find the size
        # Section to actually extract the metrics regarding melt pool
        # From the XY Plane
        # Want to find the length
        meltXInd = np.squeeze(np.where(tplanexy[0,:] > tMelt))
        melt_length = np.amax(nxrange[meltXInd])-np.amin(nxrange[meltXInd])
        melt_trail_length = abs(np.amin(nxrange[meltXInd]))
        
        # Now want to find the width + depth (can do it in same outer loop)
        yLength = 0
        zLength = 0
        zMinMelt = 0.0
        for i1 in np.arange(np.size(meltXInd,axis=0)):
            meltYInd = np.squeeze(np.where(tplanexy[:,meltXInd[i1]] > tMelt))
            tmpYLength = np.amax(nyrange[meltYInd]) - np.amin(nyrange[meltYInd])
            if tmpYLength > yLength:
                yLength = tmpYLength
            
            meltZInd = np.squeeze(np.where(tplanexz[:,meltXInd[i1]] > tMelt))
            tmpZLength = np.amax(nzrange[meltZInd]) - np.amin(nzrange[meltZInd])
            if tmpZLength > zLength:
                zLength = tmpZLength
                zMinMelt = np.amin(nzrange[meltZInd])
		
        # Test to see if the domain is the correct size
        if np.isclose(np.amax(nxrange[meltXInd]),xMax):
            # Then the x domain is not long enough
            print ("The x domain (length) is not large enough, increasing size and re-running")
            simParam.domain[0] += sigma
            # Re-run the analysis
            # melt_width, melt_depth, melt_length = eagarTsaiParam(beam,material,simParam,i)
            melt_length, melt_width, melt_depth, peakT, minT, melt_bounds = eagarTsaiParam(beam,material,simParam,i)
        elif np.isclose(yLength,abs(yMax - yMin)):
            # Then the y domain is not large enough
            print ("The y domain (width) is not large enough, increasing size and re-running")
            simParam.domain[1] += sigma
            # Re-run the analysis
            melt_length, melt_width, melt_depth, peakT, minT, melt_bounds = eagarTsaiParam(beam,material,simParam,i)
        elif np.isclose(zLength,abs(zMax-zMin)):
            print ("The z domain (depth) is not large enough, increasing size and re-running")
            simParam.domain[2] += sigma
            # Re-run the analysis
            melt_length, melt_width, melt_depth, peakT, minT, melt_bounds = eagarTsaiParam(beam,material,simParam,i)
        else:
            # All is good, the melt pool wasnt clipped in the domain
            # Return the values
            melt_width = yLength * 2
            melt_depth = zLength
            melt_bounds = {
                'x_min_m': float(np.amin(nxrange[meltXInd])),
                'x_max_m': float(np.amax(nxrange[meltXInd])),
                'y_max_m': float(yLength),
                'z_min_m': float(zMinMelt),
                'z_max_m': 0.0,
            }
    else:
        # Then there is no melt pool, return 0 lengths
        melt_width = 0.0
        melt_depth = 0.0
        melt_length = 0.0
        melt_bounds = None

    del tplanexy, tplanexz
    return melt_length, melt_width, melt_depth, peakT, minT, melt_bounds

def runIntegrate(nxrange,nyrange,nzrange,nx,ny,nz,alpha,sigma,k,v,A,P,libpath,z_plane_um=0.0):
    """ Function to run the integration """
    func = _load_integrand(libpath)

    # Set up invariant parameters
    # Define the starting temperature
    t0 = 300.0 # Kelvin
    Ts = (A*P)/(np.pi*(k/alpha)*np.sqrt(np.pi*alpha*v*(sigma**3)))
    p = alpha/(v*sigma)
    
    # Create the two temperature planes
    tplanexy = np.zeros(nx*ny,dtype='f8').reshape(ny,nx)
    tplanexz = np.zeros(nx*nz,dtype='f8').reshape(nz,nx)
    
    # Convert requested XY plane depth from microns to dimensionless z
    z_plane = (z_plane_um * 1.0e-6) / np.sqrt((alpha * sigma) / v)

    # Run the integration
    for i1 in np.arange(nx):
        x = nxrange[i1]/sigma # make dimensional
        for i2 in np.arange(ny):
            y = nyrange[i2]/sigma # make dimensional
            tmpTemp = quad(func,0.,np.inf,args=(x, y, z_plane, p))
            tplanexy[i2,i1] = t0 + Ts*tmpTemp[0]
        for i3 in np.arange(nz):
            z = nzrange[i3]/np.sqrt((alpha*sigma)/v) # make dimensional
            tmpTemp = quad(func,0.,np.inf,args=(x, 0.0, z, p))
            tplanexz[i3,i1] = t0 + Ts*tmpTemp[0]

    # Return the temperature planes
    return tplanexy, tplanexz

def save_xy_heatmap(
    nxrange,
    nyrange,
    tplanexy,
    out_dir,
    row_index,
    png_name,
    xlim_um,
    ylim_um,
    t_melt,
    mirror_y=True,
    xlabel='x (um)',
    ylabel='y (um)',
    cmap='inferno',
    vmin=0.0,
    vmax=15000.0,
):
    """
    Save an XY heatmap PNG from a temperature plane.
    """
    os.makedirs(out_dir, exist_ok=True)

    x_um = nxrange * 1.0e6
    y_um = nyrange * 1.0e6

    if mirror_y:
        y_um = np.concatenate((-y_um[1:][::-1], y_um))
        tplanexy = np.concatenate((tplanexy[1:][::-1, :], tplanexy), axis=0)

    try:
        import matplotlib.pyplot as plt
        #plt.figure()
        plt.figure(figsize=(8.4, 4.4))
        plt.pcolormesh(x_um, y_um, tplanexy, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title('Melt Pool Heatmap at z = 0 um, Alloy 0', fontsize=16)
        cbar = plt.colorbar()
        cbar.set_label('Temperature (K)', fontsize=14, labelpad=16)
        plt.contour(x_um, y_um, tplanexy, levels=[t_melt], colors='cyan', linewidths=1.0)
        plt.xlim(xlim_um)
        plt.ylim(ylim_um)
        plt.tight_layout()
        png_path = path.join(out_dir, png_name)
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f'Warning: failed to save PNG heatmap for row {row_index}: {exc}')

def temp_volume_from_row(
    row,
    domain_um=(1200.0, 1200.0, 1000.0),
    spatial_res_um=1.0,
    x_limits_m=None,
    y_limits_m=None,
    z_limits_m=None,
):
    """
    Compute the full 3D temperature volume for a single-row DataFrame.

    Returns (nxrange, nyrange, nzrange, tvolume) where coordinates are in meters
    and tvolume has shape (nx, ny, nz) in Kelvin.
    """
    beam1, mat1 = beamFromCSV(row)
    sim1 = simParam(np.array(domain_um, dtype='f8'), float(spatial_res_um))

    i = 0
    k = mat1.k[i]
    rho = mat1.rho[i]
    cp = mat1.cp[i]
    alpha = k/(rho*cp)

    P = beam1.P[i]
    A = beam1.A[i]
    v = beam1.v[i]
    sigma = beam1.sigma[i]

    if x_limits_m is None or y_limits_m is None or z_limits_m is None:
        nxrange, nyrange, nzrange = _build_coordinate_ranges(
            beam1.twoSigma[i],
            sim1.domain,
            sim1.spatialRes,
        )
    else:
        nxrange, nyrange, nzrange = _build_coordinate_ranges_from_limits(
            x_limits_m,
            y_limits_m,
            z_limits_m,
            sim1.spatialRes,
        )
    nx = nxrange.size
    ny = nyrange.size
    nz = nzrange.size

    func = _load_integrand(_get_libpath())

    t0 = 300.0
    Ts = (A * P) / (np.pi * (k / alpha) * np.sqrt(np.pi * alpha * v * (sigma ** 3)))
    p = alpha / (v * sigma)
    z_scale = np.sqrt((alpha * sigma) / v)

    tvolume = np.zeros((nx, ny, nz), dtype='f8')
    for ix in np.arange(nx):
        x = nxrange[ix] / sigma
        for iy in np.arange(ny):
            y = nyrange[iy] / sigma
            for iz in np.arange(nz):
                z = nzrange[iz] / z_scale
                tmpTemp = quad(func, 0.0, np.inf, args=(x, y, z, p))
                tvolume[ix, iy, iz] = t0 + Ts * tmpTemp[0]

    return nxrange, nyrange, nzrange, tvolume


def temp_xy_from_row(
    row,
    domain_um=(1200.0, 1200.0, 1000.0),
    spatial_res_um=1.0,
    z_plane_um=0.0,
):
    """
    Compute XY temperature plane for a single-row DataFrame.
    Returns (nxrange, nyrange, tplanexy) in meters/K.
    """
    beam1, mat1 = beamFromCSV(row)
    sim1 = simParam(np.array(domain_um, dtype='f8'), float(spatial_res_um))

    i = 0
    k = mat1.k[i]
    rho = mat1.rho[i]
    cp = mat1.cp[i]
    alpha = k / (rho * cp)

    P = beam1.P[i]
    A = beam1.A[i]
    v = beam1.v[i]
    sigma = beam1.sigma[i]

    nxrange, nyrange, nzrange = _build_coordinate_ranges(
        beam1.twoSigma[i],
        sim1.domain,
        sim1.spatialRes,
    )
    nx = nxrange.size
    ny = nyrange.size
    nz = nzrange.size

    func = _load_integrand(_get_libpath())

    t0 = 300.0
    Ts = (A * P) / (np.pi * (k / alpha) * np.sqrt(np.pi * alpha * v * (sigma ** 3)))
    p = alpha / (v * sigma)

    tplanexy = np.zeros((ny, nx), dtype='f8')
    z_plane = (z_plane_um * 1.0e-6) / np.sqrt((alpha * sigma) / v)

    for ix in np.arange(nx):
        x = nxrange[ix] / sigma
        for iy in np.arange(ny):
            y = nyrange[iy] / sigma
            tmpTemp = quad(func, 0.0, np.inf, args=(x, y, z_plane, p))
            tplanexy[iy, ix] = t0 + Ts * tmpTemp[0]

    return nxrange, nyrange, tplanexy


def save_temperature_volume_csv(
    nxrange,
    nyrange,
    nzrange,
    tvolume,
    out_dir,
    row_index,
):
    """
    Save the 3D temperature volume as CSV with columns x, y, z, T.
    """
    os.makedirs(out_dir, exist_ok=True)

    x_um = nxrange * 1.0e6
    y_um = nyrange * 1.0e6
    z_um = nzrange * 1.0e6

    xx, yy, zz = np.meshgrid(x_um, y_um, z_um, indexing='ij')
    df = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel(),
        'z': zz.ravel(),
        'T': tvolume.ravel(),
    })

    csv_path = path.join(out_dir, f'ET_3D_temperature_alloy{row_index}.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def eagar_tsai_integrand(t, x, y, z, p):
    # This is Sasha's formulation
    intpre = 1.0/((4*p*t + 1)*np.sqrt(t))
    intexp = (-(z**2)/(4*t))-(((y**2)+(x-t)**2)/(4*p*t + 1))
    return intpre * np.exp(intexp)

class simParam():
    def __init__(self,domain,spatialRes):
        """ Define a simulation parameter object """
        self.domain = domain / 1.e6 # Convert into meters
        self.spatialRes = spatialRes / 1.e6 # Convert into meters

class beam():
    def __init__(self,twoSigma,P,v,A):
        """ Define a beam object """
        self.twoSigma = twoSigma
        # The next sigma is an altered version for Sasha's ET re-interp
        self.sigma = np.sqrt(2.0) * (self.twoSigma / 2.0)
        self.P = P
        self.v = v
        self.A = A

class material():
    def __init__(self,tMelt,k,rho,cp):
        """ Define a material object """
        self.tMelt = tMelt
        self.k = k
        self.rho = rho
        self.cp = cp

def integrationWarning():
    """ Function that warns the user about the integration type """
    # We need to do some checking about which method to use
    # We need to see which version of scipy we are running
    # The compiled integral will only work on scipy 0.15.1 and later
    sciVer = scipy.version.version.split(".")
    
    # Find the platform type
    osType = sys.platform
    if int(sciVer[1]) >= 1:
        # Then we can use compiled code to run the integration
        if osType == 'darwin': # Then you're cool because that's a Mac!
            print ("Using compiled integration code, should be faster")
        elif osType in 'linux2': # It's linux, I guess that's OK
            print ("Using compiled integration code, should be faster")
        elif (osType=='win32') or (osType=='cygwin'):
            # Sorry, windows sucks, run the integration using normal interpreted code
            print ("Using interpreted integration code, will be slow")
            print ("Consider switching to a Mac or Linux")
    else:
        print ("Using old version of Python and SciPy")
        print ("Please consider switching to Python 2.7.10 and SciPy 0.15.1")
        print ("Using interpreted integration code, will be slow")

if __name__ == "__main__":

    ##########################################################################

    results_df = pd.read_excel('et_custom_input_data.xlsx')
#    results_df['Velocity_m/s'] = results_df['v']
#    results_df['Power'] = results_df['P']
    results_df['T_liquidus'] = results_df['PROP LT (K)']
    #results_df['T_solidus'] = results_df['PROP ST (K)']
    results_df['thermal_cond_liq'] = results_df['PROP LT THCD (W/(mK))']
    results_df['Density_kg/m3'] = results_df['PROP RT Density (kg/m3)']
    results_df['Cp_J/kg'] =  results_df['PROP LT C (J/(kg K))']
    results_df['Beam_diameter_m'] = results_df['Beam Diam (m)']
    #
    elements =	['W',	'Re',	'Nb'	,'Ta',	'Mo',	'Hf',	'V']
    savename = 'prop_out'
    ##########################################################################

    # Only run one Excel row
    row_num = results_df.iloc[[10]] # iloc[[EXCEL_ROW# - 2]]
    compute_melt_pool(
        row_num, #results_df if you want all the alloys
        chunk_size=1,
        workers=10,
        out_dir='beamer/figures',
        heatmap_rows=[10],      # original row index from the Excel file (EXCEL_ROW# - 2). Selected rows get the 3D temperature CSV and heatmap PNG.
        heatmap_dir='beamer/figures',
    )


    #main()
    #sum1 = summary.summarize(muppy.get_objects())
    #summary.print_(sum1)
