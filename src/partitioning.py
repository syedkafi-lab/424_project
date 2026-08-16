import os
import sys
import numpy as np
import pandas as pd

X_SIZE = 5  # 5 historical steps
Y_SIZE = 1  # 1 future target step

def create_sliding_window_sequences(df, x_size=X_SIZE, y_size=Y_SIZE):
    """
    Construct 5-step historical inputs -> 1-step target sequences grouped by trajectory run (`seq_index`).
    Returns DataFrame where each row contains the 5 historical indices and the 1 target index.
    """
    sequences = []
    # Ensure df is sorted by abs_index
    df = df.sort_values("abs_index").reset_index(drop=True)

    for seq_id, group in df.groupby("seq_index", sort=False):
        group_len = len(group)
        if group_len < x_size + y_size:
            continue
        
        group_indices = group.index.values
        group_abs = group["abs_index"].values
        group_times = group["timestamp"].values
        group_beams = group["unit1_overall-beam"].values if "unit1_overall-beam" in group else group["true_beam"].values

        for i in range(group_len - x_size - y_size + 1):
            x_idx = group_indices[i : i + x_size]
            y_idx = group_indices[i + x_size : i + x_size + y_size]
            
            record = {
                "seq_index": seq_id,
                "x_indices": list(x_idx),
                "y_index": int(y_idx[0]),
                "target_abs_index": int(group_abs[i + x_size]),
                "target_timestamp": str(group_times[i + x_size]),
                "target_beam": int(group_beams[i + x_size]),
            }
            sequences.append(record)

    seq_df = pd.DataFrame(sequences)
    print(f"Generated {len(seq_df):,} 5-past -> 1-future sequence windows across {seq_df['seq_index'].nunique()} trajectory runs.")
    return seq_df

def assign_leakage_free_splits(seq_df, raw_df, train_pct=0.55, val_pct=0.15, calib_pct=0.15, test_pct=0.15):
    """
    Assign trajectory runs (seq_index) to Train/Val/Calib/Test chronologically based on initial appearance.
    Enforces strict block separation: every trajectory run belongs to EXACTLY one split.
    """
    first_seen = raw_df.groupby("seq_index")["abs_index"].min().sort_values()
    run_order = first_seen.index.tolist()
    total_runs = len(run_order)

    n_train = int(np.round(train_pct * total_runs))
    n_val = int(np.round(val_pct * total_runs))
    n_calib = int(np.round(calib_pct * total_runs))
    n_test = total_runs - n_train - n_val - n_calib

    train_runs = set(run_order[:n_train])
    val_runs = set(run_order[n_train : n_train + n_val])
    calib_runs = set(run_order[n_train + n_val : n_train + n_val + n_calib])
    test_runs = set(run_order[n_train + n_val + n_calib :])

    def get_split(s_id):
        if s_id in train_runs:
            return "train"
        elif s_id in val_runs:
            return "val"
        elif s_id in calib_runs:
            return "calib"
        elif s_id in test_runs:
            return "test"
        return "unknown"

    seq_df["split"] = seq_df["seq_index"].apply(get_split)

    # Verification: Zero leakage
    check = seq_df.groupby("seq_index")["split"].nunique()
    assert (check == 1).all(), "Leakage Error: Some seq_index spans multiple splits!"

    print("\n[Split Assignment Summary]")
    for split_name in ["train", "val", "calib", "test"]:
        cnt = np.sum(seq_df["split"] == split_name)
        n_r = len(set(seq_df[seq_df["split"] == split_name]["seq_index"]))
        print(f"  {split_name.upper():5s}: {cnt:6,d} sequences ({cnt / len(seq_df) * 100:5.1f}%) across {n_r:3d} trajectory runs")

    return seq_df

def build_partition_manifest(data_root=".", output_dir="data/processed"):
    """Full partitioning execution creating split_manifest.csv."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(data_root, "scenario36.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "scenario36", "scenario36.csv")

    df = pd.read_csv(csv_path)
    seq_df = create_sliding_window_sequences(df)
    seq_df = assign_leakage_free_splits(seq_df, df)

    # Save manifest
    manifest_path = os.path.join(output_dir, "split_manifest.csv")
    # For storage, convert list of indices to string format
    seq_df_out = seq_df.copy()
    seq_df_out["x_indices_str"] = seq_df_out["x_indices"].apply(lambda x: ",".join(map(str, x)))
    seq_df_out.to_csv(manifest_path, index=False)
    print(f"\nSaved split manifest to {manifest_path}")

    return seq_df

if __name__ == "__main__":
    build_partition_manifest()
