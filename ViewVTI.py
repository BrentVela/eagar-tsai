import pyvista as pv

T_L = 3454.846466711204

grid = pv.read("CalcFiles/Test16/250_0.2/ET_3D_temperature_alloy0_250_0.2.vti")

surface = grid.extract_surface(algorithm="dataset_surface")
liquidus_line = surface.contour([T_L], scalars="Temperature_K").tube(radius=1.0)

plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    scalars="Temperature_K",
    cmap="inferno",
)
if liquidus_line.n_points:
    plotter.add_mesh(
        liquidus_line,
        color="cyan",
        line_width=4,
        render_lines_as_tubes=True,
    )
else:
    temp = grid.point_data["Temperature_K"]
    print(f"No T_L={T_L} K contour found. Temperature range is {temp.min():.3f} to {temp.max():.3f} K.")
plotter.add_axes()
plotter.show()
