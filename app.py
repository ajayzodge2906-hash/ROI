from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the ROI model
roi_model = joblib.load("roi_model.pkl")

# Define the features used in model
FEATURES = ['Total_Area', 'Baths', 'Parking', 'Lift', 'Security', 'Gym', 'Garden']

@app.route('/')
def home():
    return "🏡 Real Estate ROI Advisor API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        location = data.get('Location')
        area = float(data.get('Total_Area'))
        baths = int(data.get('Baths', 2))
        years = int(data.get('Years', 5))
        mode = data.get('Mode', 'both').lower()

        # Amenities - default to False if not provided
        parking = 1 if data.get('Parking') else 0
        lift = 1 if data.get('Lift') else 0
        security = 1 if data.get('Security') else 0
        gym = 1 if data.get('Gym') else 0
        garden = 1 if data.get('Garden') else 0

        # Create input vector
        input_data = pd.DataFrame([[
            area, baths, parking, lift, security, gym, garden
        ]], columns=FEATURES)

        # Predict ROI %
        roi_percent = roi_model.predict(input_data)[0]
        roi_percent = round(float(roi_percent), 2)

        # Calculate rent and future price
        estimated_annual_rent = round((roi_percent / 100) * area * 70, 2)  # Assume ₹70/sqft rent
        total_rent_income = round(estimated_annual_rent * years, 2)
        future_price = round(area * 4000 * (1 + roi_percent / 100), 2)  # Assume ₹4000/sqft base price

        # Build response
        response = {}

        if mode == "rent" or mode == "both":
            response["estimated_annual_rent"] = estimated_annual_rent
            response["total_rent_income"] = total_rent_income

        if mode == "sell" or mode == "both":
            response["future_price"] = future_price

        if mode == "buy":
            response["future_price"] = future_price
            response["estimated_annual_rent"] = estimated_annual_rent
            response["total_rent_income"] = total_rent_income
            response["profit_potential"] = round(future_price + total_rent_income, 2)

        response["roi_percent"] = roi_percent
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
