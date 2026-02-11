# predict.py
# --------------------------------
# Predict OCEAN scores for a new session
# --------------------------------

import joblib
from features import extract_features


def predict_personality(session_csv, model_path="ocean_mlp_model.pkl"):
    model = joblib.load(model_path)

    X, feature_names = extract_features(session_csv)
    X = X.reshape(1, -1)

    prediction = model.predict(X)[0]

    ocean = {
        "Openness": prediction[0],
        "Conscientiousness": prediction[1],
        "Extraversion": prediction[2],
        "Agreeableness": prediction[3],
        "Neuroticism": prediction[4]
    }

    return ocean


if __name__ == "__main__":
    result = predict_personality("new_session.csv")
    print("Predicted Personality:")
    for k, v in result.items():
        print(k, ":", round(v, 2))
