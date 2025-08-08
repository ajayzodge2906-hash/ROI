from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load models from 'models/' directory
model_dir = "models"
roi_model = joblib.load(os.path.join(model_dir, "roi_model.pkl"))
price_model = joblib.load(os.path.join(model_dir, "price_model.pkl"))
rent_model = joblib.load(os.path.join(model_dir, "rent_model.pkl"))
future_model = joblib.load(os.path.join(model_dir, "future_model.pkl"))

# Define label encoders with consistent mapping used during training
furnishing_encoder = LabelEncoder()
furnishing_encoder.classes_ = np.array(['Furnished', 'Semi-Furnished', 'Unfurnished'])

city_tier_encoder = LabelEncoder()
city_tier_encoder.classes_ = np.array([1, 2, 3])

@app.route('/')
def home():
    return "Welcome to the Real Estate ROI Advisor API"

@app.route('/predict', methods=['POST'])
def predict_roi():
    data = request.get_json()
    try:
        furnishing_encoded = furnishing_encoder.transform([data['Furnishing']])[0]
        city_tier_encoded = city_tier_encoder.transform([data['City_Tier']])[0]

        features = [
            float(data['Price']), float(data['Rent']), float(data['Maintenance']),
            float(data['Tax']), float(data['Misc']), float(data['Area_sqft']),
            int(data['Bedrooms']), int(data['Floor_Number']),
            furnishing_encoded, city_tier_encoded, int(data['Parking_Space']),
            float(data['Proximity_to_School']), float(data['Proximity_to_Hospital']),
            float(data['Proximity_to_Metro']), float(data['Green_Score']),
            float(data['Noise_Level'])
        ]

        features = np.array(features).reshape(1, -1)
        roi_value = roi_model.predict(features)[0]
        roi_percent = round(roi_value / float(data['Price']) * 100, 2)

        if roi_percent > 7:
            recommendation = "BUY"
        elif roi_percent >= 4:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"

        return jsonify({
            "roi_prediction": round(roi_value, 2),
            "roi_percent": roi_percent,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict/price', methods=['POST'])
def predict_price():
    data = request.get_json()
    try:
        furnishing_encoded = furnishing_encoder.transform([data['Furnishing']])[0]
        city_tier_encoded = city_tier_encoder.transform([data['City_Tier']])[0]

        features = [
            float(data['Area_sqft']), int(data['Bedrooms']), int(data['Floor_Number']),
            furnishing_encoded, city_tier_encoded, int(data['Parking_Space']),
            float(data['Proximity_to_School']), float(data['Proximity_to_Hospital']),
            float(data['Proximity_to_Metro']), float(data['Green_Score']), float(data['Noise_Level'])
        ]
        features = np.array(features).reshape(1, -1)
        price = price_model.predict(features)[0]
        return jsonify({"estimated_price": round(price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict/rent', methods=['POST'])
def predict_rent():
    data = request.get_json()
    try:
        furnishing_encoded = furnishing_encoder.transform([data['Furnishing']])[0]
        city_tier_encoded = city_tier_encoder.transform([data['City_Tier']])[0]

        features = [
            float(data['Area_sqft']), int(data['Bedrooms']), int(data['Floor_Number']),
            furnishing_encoded, city_tier_encoded, int(data['Parking_Space']),
            float(data['Proximity_to_School']), float(data['Proximity_to_Hospital']),
            float(data['Proximity_to_Metro']), float(data['Green_Score']), float(data['Noise_Level'])
        ]
        features = np.array(features).reshape(1, -1)
        rent = rent_model.predict(features)[0]
        return jsonify({"estimated_rent": round(rent, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict/future', methods=['POST'])
def predict_future_price():
    data = request.get_json()
    try:
        features = [
            float(data['Current_Price']), float(data['Green_Score']),
            float(data['Noise_Level'])
        ]
        features = np.array(features).reshape(1, -1)
        future_price = future_model.predict(features)[0]
        appreciation_percent = round(((future_price - float(data['Current_Price'])) / float(data['Current_Price'])) * 100, 2)
        return jsonify({
            "future_estimated_price": round(future_price, 2),
            "appreciation_percent": appreciation_percent
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
