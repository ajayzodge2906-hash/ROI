from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models (must be in the same directory on Render)
roi_model = pickle.load(open('roi_model.pkl', 'rb'))
price_model = pickle.load(open('price_model.pkl', 'rb'))
rent_model = pickle.load(open('rent_model.pkl', 'rb'))
future_model = pickle.load(open('future_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict_roi():
    data = request.json
    features = np.array([
        data['price'], data['rent'], data['area'], data['bathrooms'],
        data['maintenance'], data['tax'], data['misc'], data['location_score']
    ]).reshape(1, -1)

    prediction = roi_model.predict(features)[0]
    return jsonify({'roi_prediction': round(prediction, 2)})

@app.route('/predict/price', methods=['POST'])
def predict_price():
    data = request.json
    features = np.array([
        data['area'], data['bathrooms'], data['location_score']
    ]).reshape(1, -1)

    price = price_model.predict(features)[0]
    return jsonify({'estimated_price': round(price, 2)})

@app.route('/predict/rent', methods=['POST'])
def predict_rent():
    data = request.json
    features = np.array([
        data['price'], data['location_score'], data['area']
    ]).reshape(1, -1)

    rent = rent_model.predict(features)[0]
    return jsonify({'estimated_rent': round(rent, 2)})

@app.route('/predict/future', methods=['POST'])
def predict_future_price():
    data = request.json
    features = np.array([
        data['price'], data['location_score']
    ]).reshape(1, -1)

    future_price = future_model.predict(features)[0]
    return jsonify({'future_price': round(future_price, 2)})

if __name__ == '__main__':
    app.run(debug=True)
