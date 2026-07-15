import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Long-form data produced by PrintabilityMap.py
data = pd.read_csv('CalcFiles/Test6/printability_map_alloy0.csv')

# Pivot to grid: rows = Power, cols = Velocity
grid = data.pivot(index='Power', columns='Velocity_m/s', values='melt_depth_um')

p_vals = grid.index.astype(float).to_numpy()
v_vals = grid.columns.astype(float).to_numpy()
depth_um = grid.to_numpy()

out_dir = 'CalcFiles/Test6'
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(7,5))

# Match the reference heatmap look (blue -> green -> yellow -> red)
cmap = plt.get_cmap('jet')
norm = Normalize(vmin=0, vmax=200)

plt.pcolormesh(v_vals, p_vals, depth_um, shading='nearest', cmap=cmap, norm=norm)
plt.colorbar(label='Melt depth (um)')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Power (W)')
plt.title('Melt Depth Map vs. Power and Velocity (Alloy 0)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'melt_depth_map_alloy0_8.png'), dpi=200)
plt.close()
