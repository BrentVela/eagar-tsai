import os
import time
import threading
import numpy as np
import pandas as pd
from et_melt_pool_script import compute_melt_pool

# 1) Load base row for alloy 0 (adjust if your alloy identification differs)
df = pd.read_excel('et_input_data_example.xlsx')
df['T_liquidus'] = df['PROP LT (K)']
df['thermal_cond_liq'] = df['PROP LT THCD (W/(mK))']
df['Density_kg/m3'] = df['PROP RT Density (kg/m3)']
df['Cp_J/kg'] = df['PROP LT C (J/(kg K))']
df['Beam_diameter_m'] = df['Beam Diam (m)']

base = df.iloc[0].copy()  # alloy 0 baseline

# 2) Build grid with velocity outer, power inner
# v_vals = np.linspace(0.05, 2.0, 6)   # 6 points (0, 0.4, 0.8, 1.2, 1.6, 2.0)
# p_vals = np.linspace(50.0, 400.0, 6)  # 6 points (50, 120, 190, 260, 330, 400)
v_vals = np.linspace(0.025, 2.0, 80)      # 0.025 to 2 m/s
p_vals = np.linspace(50.0, 400.0, 71)   # 50 to 400 W

rows = []
for v in v_vals:
    for p in p_vals:
        r = base.copy()
        r['Velocity_m/s'] = v
        r['Power'] = p
        rows.append(r)


grid_df = pd.DataFrame(rows)

# If you want to see what the grid_df looks like: what data is being fed to compute_melt_pool
# grid_df.to_csv(
#     os.path.join('CalcFiles/Test6/printability_map_alloy0.csv'),
#     index=False
# )

# 3) Run melt pool model
start_time = time.time()
stop_event = threading.Event()

def _print_elapsed():
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f'Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}', flush=True)
        stop_event.wait(10.0)

timer_thread = threading.Thread(target=_print_elapsed, daemon=True)
timer_thread.start()

out = compute_melt_pool(
    grid_df,
    chunk_size=1,
    workers=12,          # increase if you want parallelism
    out_dir=None,
)

stop_event.set()
total_elapsed = time.time() - start_time
total_hours = int(total_elapsed // 3600)
total_minutes = int((total_elapsed % 3600) // 60)
total_seconds = int(total_elapsed % 60)
print(f'Total elapsed: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}', flush=True)

out_dir = 'CalcFiles/Test7'
os.makedirs(out_dir, exist_ok=True)

# Add ratio columns (melt geometry columns already exist in compute_melt_pool output)
out = out.copy()
out['W/D'] = out['melt_width_um'] / out['melt_depth_um']
out['L/W'] = out['melt_length_um'] / out['melt_width_um']

# Save selected columns
cols = [
    'Velocity_m/s',
    'Power',
    'melt_length_um',
    'melt_width_um',
    'melt_depth_um',
    'W/D',
    'L/W',
]

out[cols].to_csv(
    os.path.join(out_dir, 'printability_map_alloy0_chunk100.csv'),
    index=False
)
