import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "breast_cancer_svm_rbf.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_columns = bundle["feature_columns"]  # exact feature order used in training
target_mapping = bundle.get("target_mapping", {2: "benign", 4: "malignant"})


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_columns=feature_columns)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        missing = []

        for col in feature_columns:
            raw = request.form.get(col, "").strip()
            if raw == "":
                missing.append(col)
            else:
                values.append(float(raw))

        if missing:
            return render_template(
                "index.html",
                feature_columns=feature_columns,
                prediction_text=f"Missing value(s) for: {', '.join(missing)}"
            )

        X_input = pd.DataFrame([values], columns=feature_columns)

        pred = model.predict(X_input)[0]
        label = target_mapping.get(pred, str(pred))

        return render_template(
            "index.html",
            feature_columns=feature_columns,
            prediction_text=f"Prediction: {label}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            feature_columns=feature_columns,
            prediction_text=f"Error: {e}"
        )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
