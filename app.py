from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "breast_cancer_svm_rbf.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
target_mapping = bundle.get("target_mapping", {2: "benign", 4: "malignant"})


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_columns=feature_columns)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs in the same order as training
        values = []
        for col in feature_columns:
            v = request.form.get(col, "").strip()
            if v == "":
                return render_template("index.html",
                                       feature_columns=feature_columns,
                                       prediction_text=f"Missing value for: {col}")
            values.append(float(v))

        X_input = pd.DataFrame([values], columns=feature_columns)
        pred = model.predict(X_input)[0]
        label = target_mapping.get(pred, str(pred))

        return render_template("index.html",
                               feature_columns=feature_columns,
                               prediction_text=f"Prediction: {label}")

    except Exception as e:
        return render_template("index.html",
                               feature_columns=feature_columns,
                               prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
