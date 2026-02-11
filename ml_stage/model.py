# model.py
# --------------------------------
# Multi-output MLP Regressor for OCEAN prediction
# --------------------------------

import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_model(X_train, Y_train, model_path="ocean_mlp_model.pkl"):
    """
    X_train: (n_samples, n_features)
    Y_train: (n_samples, 5)
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, Y_train)
    joblib.dump(model, model_path)

    print("MLP model saved to", model_path)


if __name__ == "__main__":
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")

    train_model(X_train, Y_train)
