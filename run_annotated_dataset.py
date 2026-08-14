"""
DeepSense 6G V2V Scenario 36 — Annotated Dataset Analysis & Visualization Runner.
Runs all cells from annotated_dataset_scenario36.ipynb directly in the terminal with live outputs and figures.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("DeepSense 6G Scenario 36: Annotated Dataset Walkthrough")
print("=" * 70)

# Cell 1: Load Pickle and CSV
print("\n[Step 1] Loading scenario36.p and scenario36.csv...")
with open("scenario36.p", "rb") as f:
    pdata = pickle.load(f)

csv_df = pd.read_csv("scenario36.csv")
print(f"   Pickle Keys: {len(pdata.keys())} keys")
print(f"   CSV Dimensions: {csv_df.shape[0]:,} rows x {csv_df.shape[1]} columns")

# Cell 2: Feature Inspection & 256 Power Matrix
print("\n[Step 2] Assembling Telemetry & 256-Element Power Matrix...")
df = pd.DataFrame({
    "abs_index": pdata["abs_index"],
    "seq_index": pdata["seq_index"],
    "unit1_gps1_lat": pdata["unit1_gps1_lat"],
    "unit1_gps1_lon": pdata["unit1_gps1_lon"],
    "unit2_gps1_lat": pdata["unit2_gps1_lat"],
    "unit2_gps1_lon": pdata["unit2_gps1_lon"],
    "unit1_overall-beam": pdata["unit1_overall-beam"]
})

pwr1 = np.vstack(pdata["unit1_pwr1"])
pwr2 = np.vstack(pdata["unit1_pwr2"])
pwr3 = np.vstack(pdata["unit1_pwr3"])
pwr4 = np.vstack(pdata["unit1_pwr4"])
pwr_all = np.hstack([pwr1, pwr2, pwr3, pwr4])

print(f"   Total Samples: {len(df):,}")
print(f"   Power Matrix Shape: {pwr_all.shape} across 4 subarrays (4 x 64 = 256 beams)")
print("\nFirst 5 Telemetry Rows:")
print(df.head(5).to_string())

# Cell 3: 12 Spatial Kinematics
print("\n[Step 3] Computing 12 Spatial Kinematic Features...")
rel_lat = df["unit2_gps1_lat"] - df["unit1_gps1_lat"]
rel_lon = df["unit2_gps1_lon"] - df["unit1_gps1_lon"]
distance = np.sqrt(rel_lat**2 + rel_lon**2) * 111000.0
bearing = np.arctan2(rel_lon, rel_lat).astype(np.float32)

kin_df = pd.DataFrame({
    "rel_lat": rel_lat,
    "rel_lon": rel_lon,
    "distance_m": distance,
    "bearing_rad": bearing,
    "sin_bearing": np.sin(bearing),
    "cos_bearing": np.cos(bearing),
    "bearing_rate": np.gradient(bearing),
    "distance_rate": np.gradient(distance),
})

print("\nSummary Statistics of Spatial Features:")
print(kin_df.describe().T[["mean", "std", "min", "50%", "max"]].to_string())

# Cell 4: Plotting 256-Element Beam Power Spectrum
print("\n[Step 4] Plotting 256-Beam Power Spectrum for Sample #100...")
sample_idx = 100
pwr_sample_linear = pwr_all[sample_idx]
pwr_sample_db = 10.0 * np.log10(pwr_sample_linear + 1e-12)
true_best_beam = int(df["unit1_overall-beam"].iloc[sample_idx])

os.makedirs("pdf_charts_annotated", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)
ax.bar(range(256), pwr_sample_db, color='#2563EB', width=0.8, alpha=0.85, label='Measured Beam Power (dBm)')
ax.axvline(true_best_beam, color='red', linestyle='--', linewidth=2, label=f'Optimal Beam (Index {true_best_beam})')
ax.set_title(f'Sample #{sample_idx}: 256-Element Beam Power Profile Across 4 Subarrays', fontsize=11, fontweight='bold')
ax.set_xlabel('Beam Codebook Index (0 - 255)', fontsize=9, fontweight='bold')
ax.set_ylabel('Received Power (dBm)', fontsize=9, fontweight='bold')
ax.legend(loc='upper right', fontsize=8.5)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("pdf_charts_annotated/sample_100_spectrum.png", bbox_inches='tight')
plt.close()
print("   Saved plot to pdf_charts_annotated/sample_100_spectrum.png")

# Cell 5: Drive-Block Split & Trajectory Run
print("\n[Step 5] Partitioning by Drive (Zero-Leakage Architecture)...")
drives = df["seq_index"].unique()
rng = np.random.default_rng(42)
rng.shuffle(drives)

n_tr = int(len(drives) * 0.70)
n_ca = int(len(drives) * 0.15)

train_drives = set(drives[:n_tr])
calib_drives = set(drives[n_tr:n_tr + n_ca])
test_drives = set(drives[n_tr + n_ca:])

counts = [
    len(df[df["seq_index"].isin(train_drives)]),
    len(df[df["seq_index"].isin(calib_drives)]),
    len(df[df["seq_index"].isin(test_drives)])
]

print(f"   Train Set:       {len(train_drives)} drives | {counts[0]:,} samples ({100*counts[0]/len(df):.1f}%)")
print(f"   Calibration Set: {len(calib_drives)} drives | {counts[1]:,} samples ({100*counts[1]/len(df):.1f}%)")
print(f"   Test Set:        {len(test_drives)} drives | {counts[2]:,} samples ({100*counts[2]/len(df):.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
drive_1 = df[df["seq_index"] == df["seq_index"].iloc[0]]
ax1.plot(drive_1["unit1_gps1_lon"], drive_1["unit1_gps1_lat"], 'b-o', markersize=3, label='Unit 1 (Receiver)')
ax1.plot(drive_1["unit2_gps1_lon"], drive_1["unit2_gps1_lat"], 'r-s', markersize=3, label='Unit 2 (Transmitter)')
ax1.set_title("Vehicle Trajectory Run (Drive #1)", fontsize=10.5, fontweight='bold')
ax1.set_xlabel("Longitude (deg)", fontsize=8.5)
ax1.set_ylabel("Latitude (deg)", fontsize=8.5)
ax1.legend(fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.6)

bars = ax2.bar(['Train (70%)', 'Calib (15%)', 'Test (15%)'], counts, color=['#1E3A8A', '#0D9488', '#E11D48'], width=0.55)
ax2.set_title("Zero-Leakage Drive Partition Distribution", fontsize=10.5, fontweight='bold')
ax2.set_ylabel("Sample Count", fontsize=8.5)
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 300, f"{yval:,}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("pdf_charts_annotated/drive_partitions.png", bbox_inches='tight')
plt.close()
print("   Saved trajectory & split plot to pdf_charts_annotated/drive_partitions.png")

print("\n" + "=" * 70)
print("Analysis Completed Successfully!")
print("=" * 70)
