# features.py
# --------------------------------
# Converts a session thermal CSV into a fixed-length
# time-aware feature vector using stimulus blocks
# Supports repeated stimulus loops
# --------------------------------

import pandas as pd
import numpy as np


# Stimulus blocks within ONE loop (seconds)
STIMULUS_BLOCKS = [
    ("baseline", 0.0, 3.0),
    ("structure", 4.0, 5.0),
    ("velocity", 6.0, 8.0),
    ("fire", 9.0, 11.0),
    ("abstract", 12.0, 24.0),
    ("shock", 25.0, 31.0),
]

LOOP_DURATION = 31.0  # seconds


def extract_features(session_csv_path):
    """
    Input:
        CSV with columns:
            timestamp (seconds)
            roi
            mean_temp
            std_temp

    Output:
        X: 1D numpy array
        feature_names
    """

    df = pd.read_csv(session_csv_path)

    features = []
    feature_names = []

    rois = sorted(df["roi"].unique())

    # Determine how many loops exist
    num_loops = int(df["timestamp"].max() // LOOP_DURATION) + 1

    for loop_idx in range(num_loops):
        loop_start = loop_idx * LOOP_DURATION
        loop_end = loop_start + LOOP_DURATION

        loop_df = df[
            (df["timestamp"] >= loop_start) &
            (df["timestamp"] < loop_end)
        ]

        for block_name, t_start, t_end in STIMULUS_BLOCKS:
            abs_start = loop_start + t_start
            abs_end = loop_start + t_end

            block_df = loop_df[
                (loop_df["timestamp"] >= abs_start) &
                (loop_df["timestamp"] <= abs_end)
            ]

            for roi in rois:
                roi_df = block_df[block_df["roi"] == roi]

                if roi_df.empty:
                    mean_val = 0.0
                    std_val = 0.0
                else:
                    mean_val = roi_df["mean_temp"].mean()
                    std_val = roi_df["std_temp"].mean()

                features.extend([mean_val, std_val])
                feature_names.extend([
                    f"loop{loop_idx+1}_{block_name}_{roi}_mean",
                    f"loop{loop_idx+1}_{block_name}_{roi}_std"
                ])

    X = np.array(features, dtype=float)
    return X, feature_names


# Sanity test
if __name__ == "__main__":
    X, names = extract_features("session_20260210_141330.csv")
    print("Feature vector length:", len(X))
    for n, v in zip(names, X):
        print(f"{n}: {v:.4f}")
