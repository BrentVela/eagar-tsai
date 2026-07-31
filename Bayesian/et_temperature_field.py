"""Efficient access to regular Eagar--Tsai temperature fields."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


TEMPERATURE_ARRAY_ALIASES = ("Temperature_K", "T_ET", "temperature")


@dataclass(frozen=True)
class ETTemperatureField:
    """A regular ET grid with coordinates in micrometers and temperature in K."""

    x_um: np.ndarray
    y_um: np.ndarray
    z_um: np.ndarray
    temperature_k: np.ndarray
    source: Path
    scan_direction_x_sign: int

    @property
    def axes(self):
        return self.x_um, self.y_um, self.z_um

    @property
    def temperature_min(self):
        return float(np.min(self.temperature_k))

    @property
    def temperature_max(self):
        return float(np.max(self.temperature_k))

    def interpolator(self, *, bounds_error=True):
        return RegularGridInterpolator(
            self.axes,
            self.temperature_k,
            method="linear",
            bounds_error=bounds_error,
            fill_value=None,
        )

    def yz_slice(
        self,
        x_um,
        *,
        y_limits_um=None,
        z_limits_um=None,
    ):
        """Return ``(actual_x, yy, zz, temperature)`` at the nearest x plane."""
        x_index = int(np.argmin(np.abs(self.x_um - float(x_um))))
        actual_x = float(self.x_um[x_index])
        tolerance = (
            abs(float(np.diff(self.x_um)[0])) * 0.5 + 1.0e-9
            if self.x_um.size > 1
            else 1.0e-9
        )
        if abs(actual_x - float(x_um)) > tolerance:
            raise ValueError(
                f"Requested x={x_um:g} um is outside the ET grid near "
                f"x={actual_x:g} um in {self.source}."
            )

        y_indices = np.flatnonzero(_axis_limit_mask(self.y_um, y_limits_um))
        z_indices = np.flatnonzero(_axis_limit_mask(self.z_um, z_limits_um))
        y = self.y_um[y_indices]
        z = self.z_um[z_indices]
        if y.size == 0 or z.size == 0:
            raise ValueError(
                f"Requested YZ limits select no ET points in {self.source}."
            )

        # Stored field order is (x, y, z); plotting expects (z, y).
        temperature = self.temperature_k[x_index][
            np.ix_(y_indices, z_indices)
        ].T
        yy, zz = np.meshgrid(y, z)
        return actual_x, yy, zz, temperature

    def roi_dataframe(self, limits_um):
        """Expand only a requested subvolume into x/y/z/T_ET table form."""
        x_indices = np.flatnonzero(
            _axis_limit_mask(self.x_um, limits_um.get("x"))
        )
        y_indices = np.flatnonzero(
            _axis_limit_mask(self.y_um, limits_um.get("y"))
        )
        z_indices = np.flatnonzero(
            _axis_limit_mask(self.z_um, limits_um.get("z"))
        )
        if not x_indices.size or not y_indices.size or not z_indices.size:
            raise ValueError(
                f"ET extrapolation ROI selected no points: {limits_um}"
            )

        x = self.x_um[x_indices]
        y = self.y_um[y_indices]
        z = self.z_um[z_indices]
        temperature = self.temperature_k[
            np.ix_(x_indices, y_indices, z_indices)
        ]
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return pd.DataFrame(
            {
                "x": xx.ravel(order="C"),
                "y": yy.ravel(order="C"),
                "z": zz.ravel(order="C"),
                "T_ET": temperature.ravel(order="C"),
            }
        )


def _axis_limit_mask(axis, limits):
    if limits is None:
        return np.ones(axis.size, dtype=bool)
    lower, upper = limits
    return (axis >= float(lower)) & (axis <= float(upper))


def _validate_field(field):
    expected_shape = (
        field.x_um.size,
        field.y_um.size,
        field.z_um.size,
    )
    if field.temperature_k.shape != expected_shape:
        raise ValueError(
            f"ET temperature shape {field.temperature_k.shape} does not "
            f"match coordinate shape {expected_shape} in {field.source}."
        )
    for name, axis in zip(("x", "y", "z"), field.axes):
        if axis.size < 2:
            raise ValueError(
                f"ET {name} axis needs at least two points in {field.source}."
            )
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(
                f"ET {name} coordinates must increase in {field.source}."
            )
    if not np.isfinite(field.temperature_k).all():
        raise ValueError(
            f"ET temperature contains non-finite values in {field.source}."
        )
    if field.scan_direction_x_sign not in (-1, 1):
        raise ValueError(
            "ET scan_direction_x_sign must be -1 or +1 in "
            f"{field.source}."
        )
    return field


def _load_vti(path):
    import pyvista as pv

    grid = pv.read(path)
    if not isinstance(grid, pv.ImageData):
        raise ValueError(
            f"Expected VTK ImageData in {path}, found {type(grid).__name__}."
        )

    array_name = next(
        (name for name in TEMPERATURE_ARRAY_ALIASES if name in grid.point_data),
        None,
    )
    if array_name is None:
        raise ValueError(
            f"No ET temperature point array found in {path}. Available "
            f"arrays: {sorted(grid.point_data.keys())}"
        )

    dimensions = tuple(int(value) for value in grid.dimensions)
    origin = np.asarray(grid.origin, dtype=float)
    spacing = np.asarray(grid.spacing, dtype=float)
    axes = tuple(
        origin[index] + spacing[index] * np.arange(dimensions[index])
        for index in range(3)
    )
    temperature = np.asarray(
        grid.point_data[array_name],
        dtype=float,
    ).reshape(dimensions, order="F")
    if "scan_direction_x_sign" in grid.field_data:
        scan_direction_x_sign = int(
            np.asarray(grid.field_data["scan_direction_x_sign"]).flat[0]
        )
    else:
        # VTI files written directly by eagar_tsai use +x for the trailing
        # wake, so the laser scans toward -x.
        scan_direction_x_sign = -1

    # Normalize negative VTK spacing to increasing interpolation axes.
    axes = list(axes)
    for index, axis in enumerate(axes):
        if axis.size > 1 and axis[1] < axis[0]:
            axes[index] = axis[::-1]
            temperature = np.flip(temperature, axis=index)

    return _validate_field(
        ETTemperatureField(
            x_um=np.asarray(axes[0]),
            y_um=np.asarray(axes[1]),
            z_um=np.asarray(axes[2]),
            temperature_k=temperature,
            source=path,
            scan_direction_x_sign=scan_direction_x_sign,
        )
    )


def _load_csv(path):
    table = pd.read_csv(path, usecols=["x", "y", "z", "T_ET"])
    axes = tuple(
        np.sort(table[column].unique()) for column in ("x", "y", "z")
    )
    shape = tuple(axis.size for axis in axes)
    if np.prod(shape) != len(table) or table.duplicated(
        ["x", "y", "z"]
    ).any():
        raise ValueError(
            f"ET CSV must contain one value at every Cartesian point: {path}"
        )

    ordered = table.sort_values(["x", "y", "z"], kind="stable")
    temperature = ordered["T_ET"].to_numpy(dtype=float).reshape(
        shape,
        order="C",
    )
    return _validate_field(
        ETTemperatureField(
            x_um=axes[0],
            y_um=axes[1],
            z_um=axes[2],
            temperature_k=temperature,
            source=path,
            # Legacy ET-prior CSV exports used +x as the scan direction.
            scan_direction_x_sign=1,
        )
    )


def load_et_temperature_field(path):
    """Load an ET ``.vti`` efficiently, with legacy ``.csv`` support."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vti":
        return _load_vti(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(
        f"Unsupported ET temperature field {path}. Use .vti or .csv."
    )
