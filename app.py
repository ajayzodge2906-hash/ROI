from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

roi_model = pickle.load(open('models/roi_model.pkl', 'rb'))
price_model = pickle.load(open('models/price_model.pkl', 'rb'))
rent_model = pickle.load(open('models/rent_model.pkl', 'rb'))
future_model = pickle.load(open('models/future_model.pkl', 'rb'))


@app.route('/')
def home():
    return jsonify({'message': 'Real Estate ROI Advisor API is running ✅'})

# ROI Prediction
@app.route('/predict', methods=['POST'])
def predict_roi():
    data = request.json

    try:
        features = np.array([
            float(data['Price']),
            float(data['Rent']),
            float(data['Maintenance']),
            float(data['Tax']),
            float(data['Misc']),
            float(data['Area_sqft']),
            int(data['Bedrooms']),
            float(data['Location_Score'])
        ]).reshape(1, -1)

        prediction = roi_model.predict(features)[0]
        return jsonify({'roi_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Price Estimation
@app.route('/predict/price', methods=['POST'])
def predict_price():
    data = request.json

    try:
        features = np.array([
            float(data['Area_sqft']),
            int(data['Bedrooms']),
            float(data['Location_Score'])
        ]).reshape(1, -1)

        prediction = price_model.predict(features)[0]
        return jsonify({'estimated_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rent Estimation
@app.route('/predict/rent', methods=['POST'])
def predict_rent():
    data = request.json

    try:
        features = np.array([
            float(data['Area_sqft']),
            int(data['Bedrooms']),
            float(data['Location_Score'])
        ]).reshape(1, -1)

        prediction = rent_model.predict(features)[0]
        return jsonify({'estimated_rent': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Future Price Estimation
@app.route('/predict/future', methods=['POST'])
def predict_future_price():
    data = request.json

    try:
        features = np.array([
            float(data['Area_sqft']),
            int(data['Bedrooms']),
            float(data['Location_Score'])
        ]).reshape(1, -1)

        prediction = future_model.predict(features)[0]
        return jsonify({'future_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


