import ctypes
import os
import os.path as path
import sys

import numpy as np
import pyvista as pv
import scipy.version
from scipy.integrate import quad


def eagar_tsai_integrand(t, x, y, z, p):
    intpre = 1.0 / ((4 * p * t + 1) * np.sqrt(t))
    intexp = (-(z**2) / (4 * t)) - (((y**2) + (x - t) ** 2) / (4 * p * t + 1))
    return intpre * np.exp(intexp)


def _get_libpath():
    sci_ver = scipy.version.version.split(".")
    os_type = sys.platform

    if int(sci_ver[1]) >= 1:
        if os_type == "darwin":
            return "libeagar_tsai_integrand.dylib"
        if os_type.startswith("linux"):
            return "./libeagar_tsai_integrand.so"
        if os_type in ("win32", "cygwin", "msys"):
            return "libeagar_tsai_integrand.dll"
    return None


def _load_integrand():
    libpath = _get_libpath()
    if not libpath or not path.exists(libpath):
        return eagar_tsai_integrand

    lib = ctypes.CDLL(libpath)
    func = lib.eagar_tsai_integrand
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_double)
    return func


def _axis_range_um(axis_limits_um, spatial_res_um):
    axis_min, axis_max = axis_limits_um
    n_points = int(np.round(abs(axis_max - axis_min) / spatial_res_um)) + 1
    return np.linspace(axis_min, axis_max, n_points)


def _temperature_volume(
    power_w,
    scan_speed_m_s,
    beam_diameter_m,
    absorptivity,
    thermal_conductivity_w_mk,
    density_kg_m3,
    heat_capacity_j_kgk,
    x_um,
    y_um,
    z_um,
):
    alpha = thermal_conductivity_w_mk / (density_kg_m3 * heat_capacity_j_kgk)
    sigma = np.sqrt(2.0) * (beam_diameter_m / 2.0)

    t0 = 300.0
    ts = (absorptivity * power_w) / (
        np.pi
        * (thermal_conductivity_w_mk / alpha)
        * np.sqrt(np.pi * alpha * scan_speed_m_s * (sigma**3))
    )
    p = alpha / (scan_speed_m_s * sigma)
    z_scale = np.sqrt((alpha * sigma) / scan_speed_m_s)
    func = _load_integrand()

    x_m = x_um * 1.0e-6
    y_m = y_um * 1.0e-6
    z_m = z_um * 1.0e-6

    tvolume = np.zeros((x_um.size, y_um.size, z_um.size), dtype="f8")
    for ix, x_val_m in enumerate(x_m):
        x = x_val_m / sigma
        for iy, y_val_m in enumerate(y_m):
            y = y_val_m / sigma
            for iz, z_val_m in enumerate(z_m):
                z = z_val_m / z_scale
                integral, _ = quad(func, 0.0, np.inf, args=(x, y, z, p))
                tvolume[ix, iy, iz] = t0 + ts * integral

    return tvolume


def _save_vti(output_vti, x_um, y_um, z_um, temperature_k):
    output_vti = path.abspath(output_vti)
    os.makedirs(path.dirname(output_vti), exist_ok=True)

    dx = float(x_um[1] - x_um[0]) if x_um.size > 1 else 1.0
    dy = float(y_um[1] - y_um[0]) if y_um.size > 1 else 1.0
    dz = float(z_um[1] - z_um[0]) if z_um.size > 1 else 1.0

    grid = pv.ImageData(
        dimensions=(x_um.size, y_um.size, z_um.size),
        spacing=(dx, dy, dz),
        origin=(float(x_um[0]), float(y_um[0]), float(z_um[0])),
    )
    grid.point_data["Temperature_K"] = np.ascontiguousarray(temperature_k).ravel(order="F")
    grid.save(output_vti)
    return output_vti


def export_eagar_tsai_vti(
    output_vti,
    power_w,
    scan_speed_m_s,
    beam_diameter_m,
    absorptivity,
    thermal_conductivity_w_mk,
    density_kg_m3,
    heat_capacity_j_kgk,
    x_limits_um,
    y_limits_um,
    z_limits_um,
    spatial_res_um=1.0,
):
    x_um = _axis_range_um(x_limits_um, spatial_res_um)
    y_um = _axis_range_um(y_limits_um, spatial_res_um)
    z_um = _axis_range_um(z_limits_um, spatial_res_um)

    temperature_k = _temperature_volume(
        power_w=power_w,
        scan_speed_m_s=scan_speed_m_s,
        beam_diameter_m=beam_diameter_m,
        absorptivity=absorptivity,
        thermal_conductivity_w_mk=thermal_conductivity_w_mk,
        density_kg_m3=density_kg_m3,
        heat_capacity_j_kgk=heat_capacity_j_kgk,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
    )
    return _save_vti(output_vti, x_um, y_um, z_um, temperature_k)


if __name__ == "__main__":
    vti_path = export_eagar_tsai_vti(
        output_vti="CalcFiles/Test16/125_0.2/ET_3D_temperature_alloy0_125_0.2.vti",
        power_w=125.0,
        scan_speed_m_s=0.2,
        beam_diameter_m=80.0e-6,
        absorptivity=0.59,
        thermal_conductivity_w_mk=23.75,
        density_kg_m3=18038.9,
        heat_capacity_j_kgk=251.6,
        x_limits_um=(-160.0, 380.0),
        y_limits_um=(0.0, 200.0),
        z_limits_um=(-60.0, 0.0),
        spatial_res_um=1.0,
    )
    print(f"Wrote {vti_path}")
