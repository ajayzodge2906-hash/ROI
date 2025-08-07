from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load models
price_model = pickle.load(open("models/price_model.pkl", "rb"))
rent_model = pickle.load(open("models/rent_model.pkl", "rb"))

# Location encoder categories (manually extracted during training)
location_categories = [
    'Andheri East, Mumbai', 'Andheri West, Mumbai', 'Borivali, Mumbai',
    'Bandra, Mumbai', 'Dadar, Mumbai', 'Goregaon, Mumbai',
    'Juhu, Mumbai', 'Kandivali, Mumbai', 'Kurla, Mumbai', 'Malad, Mumbai',
    'Mulund, Mumbai', 'Powai, Mumbai', 'Thane, Mumbai', 'Vile Parle, Mumbai'
]

# Function to encode location manually
def encode_location(location):
    encoded = [1 if loc == location else 0 for loc in location_categories]
    return encoded

# Function to calculate amenities score
def calculate_amenities_score(data):
    amenities = ['Parking', 'Lift', 'Security', 'Gym', 'Garden']
    score = sum([1 for amenity in amenities if data.get(amenity, False)])
    return score

@app.route('/')
def home():
    return jsonify({'message': '🏠 Real Estate ROI Predictor API is live!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        location = data['Location']
        total_area = float(data['Total_Area'])
        baths = int(data.get('Baths', 2))
        years = int(data.get('Years', 5))
        mode = data.get('Mode', 'both').lower()

        if location not in location_categories:
            return jsonify({'error': f"Unsupported location: {location}"}), 400

        # Encode location
        encoded_location = encode_location(location)

        # Amenities score
        amenities_score = calculate_amenities_score(data)

        # Final feature vector
        features = [total_area, baths, amenities_score] + encoded_location
        features_array = np.array(features).reshape(1, -1)

        result = {'mode': mode}

        # Common predictions
        current_price = price_model.predict(features_array)[0]
        estimated_annual_rent = rent_model.predict(features_array)[0]
        total_rent_income = estimated_annual_rent * years
        future_price = current_price * (1.06 ** years)  # 6% annual appreciation

        # ROI calculation
        roi = ((total_rent_income) / current_price) * 100
        total_return = (future_price - current_price) + total_rent_income
        roi_total = (total_return / current_price) * 100

        # Handle modes
        if mode == "rent":
            result.update({
                "estimated_annual_rent": round(estimated_annual_rent, 2),
                "total_rent_income": round(total_rent_income, 2),
                "roi_percent": round(roi, 2)
            })

        elif mode == "sell":
            result.update({
                "current_price": round(current_price, 2),
                "future_price": round(future_price, 2)
            })

        elif mode == "rent_only":
            result.update({
                "estimated_annual_rent": round(estimated_annual_rent, 2)
            })

        elif mode == "buy":
            result.update({
                "current_price": round(current_price, 2),
                "future_price": round(future_price, 2),
                "estimated_annual_rent": round(estimated_annual_rent, 2),
                "total_rent_income": round(total_rent_income, 2),
                "total_return": round(total_return, 2),
                "roi_percent": round(roi_total, 2)
            })

        elif mode == "both":
            result.update({
                "current_price": round(current_price, 2),
                "future_price": round(future_price, 2),
                "estimated_annual_rent": round(estimated_annual_rent, 2),
                "total_rent_income": round(total_rent_income, 2),
                "roi_percent": round(roi, 2)
            })

        else:
            return jsonify({"error": f"Invalid mode: {mode}. Choose from rent, sell, both, rent_only, buy."}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
