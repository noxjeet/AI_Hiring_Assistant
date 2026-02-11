# build_dataset.py
# --------------------------------
# Builds training dataset (X_train, Y_train)
# from session CSVs and questionnaire outputs
# --------------------------------

import os
import numpy as np
from features import extract_features


# -------- CONFIGURATION --------
SESSION_DIR = "data/sessions"
QUESTIONNAIRE_DIR = "data/questionnaires"

X_OUTPUT_PATH = "X_train.npy"
Y_OUTPUT_PATH = "Y_train.npy"
# --------------------------------


def build_dataset():
    X_list = []
    Y_list = []

    session_files = sorted([
        f for f in os.listdir(SESSION_DIR)
        if f.endswith(".csv")
    ])

    if not session_files:
        raise RuntimeError("No session CSV files found.")

    for session_file in session_files:
        session_id = os.path.splitext(session_file)[0]

        session_csv_path = os.path.join(SESSION_DIR, session_file)
        questionnaire_path = os.path.join(
            QUESTIONNAIRE_DIR,
            f"{session_id}.npy"
        )

        if not os.path.exists(questionnaire_path):
            print(f"[SKIP] No questionnaire for {session_id}")
            continue

        # -------- Extract X --------
        X, feature_names = extract_features(session_csv_path)

        # -------- Load Y --------
        Y = np.load(questionnaire_path)

        if Y.shape != (5,):
            raise ValueError(
                f"Invalid questionnaire shape for {session_id}: {Y.shape}"
            )

        X_list.append(X)
        Y_list.append(Y)

        print(f"[OK] Added session {session_id}")

    if not X_list:
        raise RuntimeError("No valid sessions found.")

    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)

    np.save(X_OUTPUT_PATH, X_train)
    np.save(Y_OUTPUT_PATH, Y_train)

    print("\nDataset built successfully")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("Saved to:")
    print(" ", X_OUTPUT_PATH)
    print(" ", Y_OUTPUT_PATH)


# -------- Run manually --------
if __name__ == "__main__":
    build_dataset()
