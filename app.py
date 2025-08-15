from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model, scaler, and feature columns
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect input data from form
            input_data = []
            for feature in feature_columns:
                value = request.form.get(feature)
                if value is None:
                    value = 0
                input_data.append(float(value))

            # Convert to numpy array and scale
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Make prediction
            pred = model.predict(input_scaled)[0]
            prediction = "Approved ✅" if pred == 1 else "Rejected ❌"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, features=feature_columns)

if __name__ == "__main__":
    app.run(debug=True)
