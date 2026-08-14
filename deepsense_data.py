"""
Load DeepSense V2V scenarios using the official pickle + CSV format.

Data layout matches the 2023 Beam Prediction Challenge baseline:
  https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline

Fast path (training/benchmark):
  - scenario36.p  (pickle dict with GPS, power, image paths keyed by abs_index)
  - deepsense_challenge2023_trainset.csv  (x1-x5 input windows, y1 output labels)

Slow path (competition test):
  - Individual .txt GPS files and image paths in CSV columns
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

N_ARR = 4
N_BEAMS = 64
N_BEAMS_TOTAL = N_ARR * N_BEAMS
X_SIZE = 5
IDX_COL_TRAIN = "unique_index"
IDX_COL_SCEN = "abs_index"


def load_scenario_pickle(pickle_path: str | Path) -> dict:
    """Load pre-built scenario dict (scenario36.p, etc.)."""
    with open(pickle_path, "rb") as fp:
        return pickle.load(fp)


def _lookup_by_abs_index(csv_dict: dict, abs_index: int) -> int:
    """Return row index in pickle arrays for a given abs_index."""
    matches = np.where(csv_dict[IDX_COL_SCEN] == abs_index)[0]
    if len(matches) == 0:
        raise KeyError(f"abs_index {abs_index} not found in scenario pickle")
    return int(matches[0])


def pickle_to_dataframe(
    csv_path: str | Path,
    pickle_path: str | Path,
    scenario_id: int,
    image_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Convert official DeepSense training CSV + scenario pickle into a flat
    DataFrame compatible with deepsense_beam_tracking.py.

    Each row is one prediction target (y1 timestamp) with:
      - unit1_lat/lon, unit2_lat/lon at output time
      - pwr1_0 ... pwr4_63 linear power values
      - true beam index (0-255)
      - image_path (front camera rgb5, relative or absolute)
      - drive_id (= scenario_id for block splitting)
    """
    csv_path = Path(csv_path)
    pickle_path = Path(pickle_path)
    csv_dict = load_scenario_pickle(pickle_path)
    df_train = pd.read_csv(csv_path)

    samples = np.where(df_train["scenario"] == scenario_id)[0]
    if len(samples) == 0:
        raise ValueError(f"No rows for scenario {scenario_id} in {csv_path}")

    if image_root is None:
        image_root = pickle_path.parent
    image_root = Path(image_root)

    power_cols = [f"pwr{a + 1}_{b}" for a in range(N_ARR) for b in range(N_BEAMS)]
    rows = []

    for train_idx in tqdm(samples, desc=f"Loading scenario {scenario_id}"):
        y_abs = int(df_train[f"y1_{IDX_COL_TRAIN}"].iloc[train_idx])
        y_row = _lookup_by_abs_index(csv_dict, y_abs)

        unit1 = csv_dict["unit1_gps1"][y_row]
        unit2 = csv_dict["unit2_gps1"][y_row]

        record = {
            "drive_id": scenario_id,
            "abs_index": y_abs,
            "unit1_lat": float(unit1[0]),
            "unit1_lon": float(unit1[1]),
            "unit2_lat": float(unit2[0]),
            "unit2_lon": float(unit2[1]),
            "true_beam": int(df_train["y1_unit1_overall-beam"].iloc[train_idx]),
        }

        for arr_idx in range(N_ARR):
            pwrs = csv_dict[f"unit1_pwr{arr_idx + 1}"][y_row]
            for beam_idx in range(N_BEAMS):
                record[f"pwr{arr_idx + 1}_{beam_idx}"] = float(pwrs[beam_idx])

        rgb_rel = csv_dict["unit1_rgb5"][y_row]
        record["image_path"] = str(rgb_rel)
        record["image_path_full"] = str(image_root / rgb_rel)

        rows.append(record)

    df = pd.DataFrame(rows)
    df["overall-beam"] = df["true_beam"]
    return df


def load_all_scenarios(
    csv_path: str | Path,
    pickle_dir: str | Path,
    scenarios: list[int] | None = None,
    image_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load and concatenate scenarios 36-39 (default)."""
    if scenarios is None:
        scenarios = [36, 37, 38, 39]

    pickle_dir = Path(pickle_dir)
    parts = []
    for scen in scenarios:
        pkl = pickle_dir / f"scenario{scen}" / f"scenario{scen}.p"
        if not pkl.exists():
            pkl = pickle_dir / f"scenario{scen}.p"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing pickle for scenario {scen}: tried {pkl}")
        parts.append(
            pickle_to_dataframe(
                csv_path=csv_path,
                pickle_path=pkl,
                scenario_id=scen,
                image_root=image_root or pkl.parent,
            )
        )
    return pd.concat(parts, ignore_index=True)


def get_power_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered pwr1_0 ... pwr4_63 column names present in df."""
    cols = [c for c in df.columns if c.startswith("pwr") and "_" in c]
    cols.sort(key=lambda c: (int(c.split("_")[0][3:]), int(c.split("_")[1])))
    return cols
