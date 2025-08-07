from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
roi_model = pickle.load(open('models/roi_model.pkl', 'rb'))
price_model = pickle.load(open('models/price_model.pkl', 'rb'))
rent_model = pickle.load(open('models/rent_model.pkl', 'rb'))
current_price_model = pickle.load(open('models/current_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return "🏠 Real Estate ROI Advisor API is running."

@app.route('/predict_roi', methods=['POST'])
def predict_roi():
    data = request.get_json()
    try:
        features = np.array([
            data['price'],
            data['rent'],
            data['maintenance'],
            data['tax'],
            data['misc']
        ]).reshape(1, -1)

        prediction = roi_model.predict(features)[0]
        return jsonify({'roi_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    try:
        features = np.array([
            data['area_sqft'],
            data['bedrooms'],
            data['location_score']
        ]).reshape(1, -1)

        prediction = price_model.predict(features)[0]
        return jsonify({'price_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_rent', methods=['POST'])
def predict_rent():
    data = request.get_json()
    try:
        features = np.array([
            data['area_sqft'],
            data['bedrooms'],
            data['location_score']
        ]).reshape(1, -1)

        prediction = rent_model.predict(features)[0]
        return jsonify({'rent_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_current_price', methods=['POST'])
def predict_current_price():
    data = request.get_json()
    try:
        features = np.array([
            data['area_sqft'],
            data['bedrooms'],
            data['location_score']
        ]).reshape(1, -1)

        prediction = current_price_model.predict(features)[0]
        return jsonify({'current_price_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
