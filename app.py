from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load model
model = joblib.load("roi_model.pkl")

@app.route('/')
def home():
    return "🏠 Real Estate ROI Predictor API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        total_area = float(data['Total_Area'])
        bhk = int(data['BHK'])
        city = data['City']

        input_df = pd.DataFrame({
            "Total_Area": [total_area],
            "BHK": [bhk],
            "City": [city]
        })

        future_price = model.predict(input_df)[0]
        estimated_rent = future_price * 0.025
        total_rent_income = estimated_rent * 5
        current_price = future_price / (1.08 ** 5)
        roi = ((future_price + total_rent_income - current_price) / current_price) * 100

        return jsonify({
            "future_price": round(future_price, 2),
            "estimated_annual_rent": round(estimated_rent, 2),
            "total_rent_income": round(total_rent_income, 2),
            "roi_percent": round(roi, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
