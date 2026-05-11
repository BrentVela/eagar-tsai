import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

data_path = 'CalcFiles/Test7/printability_map_alloy0_chunk100.csv'
out_dir = 'CalcFiles/Test7'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Ensure expected columns exist
required = ['Velocity_m/s', 'Power', 'melt_depth_um', 'L/W', 'W/D']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing columns in {data_path}: {missing}')

# Conditions
cond_balling = df['L/W'] > 2.3 #turquoise OG: 2.9
cond_lof = df['melt_depth_um'] < 30.0 #light_pink OG: 30.0
cond_keyholing = df['W/D'] < 2.5 #sky_blue OG: 1.8

# Class priority: balling > lof > keyholing > defect free
df['class'] = 0
df.loc[cond_keyholing, 'class'] = 1
df.loc[cond_lof, 'class'] = 2
df.loc[cond_balling, 'class'] = 3

# Pivot to grid: rows = Power, cols = Velocity
grid = df.pivot(index='Power', columns='Velocity_m/s', values='class')
p_vals = grid.index.astype(float).to_numpy()
v_vals = grid.columns.astype(float).to_numpy()
classes = grid.to_numpy()

# Colormap: 0=white, 1=sky blue, 2=light pink, 3=turquoise
cmap = mcolors.ListedColormap(['white', 'skyblue', 'lightpink', 'turquoise'])
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

plt.figure(figsize=(7, 5))
plt.pcolormesh(v_vals, p_vals, classes, shading='nearest', cmap=cmap, norm=norm)
legend_handles = [
    Patch(facecolor='white', edgecolor='black', label='Defect-free'),
    Patch(facecolor='skyblue', edgecolor='black', label='Keyholing (W/D < 2.5)'),
    Patch(facecolor='turquoise', edgecolor='black', label='Balling (L/W > 2.3)'),
    Patch(facecolor='lightpink', edgecolor='black', label='Lack of Fusion (D < 30 um)')
]
plt.legend(handles=legend_handles, loc='lower right', frameon=True)
plt.xlabel('Scan speed, v (m/s)')
plt.ylabel('Laser power, P (W)')
plt.title('Printability Map (Alloy 0)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'printability_map_alloy0_chunk100.png'), dpi=200)
plt.close()
