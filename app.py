from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ----------- Load local models ----------- #
with open('roi_model.pkl', 'rb') as f:
    roi_model = pickle.load(f)

with open('price_model.pkl', 'rb') as f:
    price_model = pickle.load(f)

with open('rent_model.pkl', 'rb') as f:
    rent_model = pickle.load(f)

with open('future_model.pkl', 'rb') as f:
    future_model = pickle.load(f)

# ----------- Prediction Route ----------- #
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mode = data.get('mode', 'roi')

    try:
        if mode == 'roi':
            features = [
                float(data['Price']),
                float(data['Rent']),
                float(data['Maintenance']),
                float(data['Tax']),
                float(data['Misc']),
                float(data['Total_Area']),
                int(data['Bedrooms']),
                float(data['Location_Score'])
            ]
            model = roi_model

        elif mode == 'price':
            features = [
                float(data['Total_Area']),
                int(data['Bedrooms']),
                float(data['Location_Score'])
            ]
            model = price_model

        elif mode == 'rent':
            features = [
                float(data['Price']),
                float(data['Total_Area']),
                int(data['Bedrooms']),
                float(data['Location_Score'])
            ]
            model = rent_model

        elif mode == 'future':
            features = [
                float(data['Price']),
                float(data['Location_Score'])
            ]
            model = future_model

        else:
            return jsonify({'error': 'Invalid prediction mode'}), 400

        prediction = model.predict([features])[0]
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------- Home Route ----------- #
@app.route('/')
def home():
    return '🏠 Real Estate ROI Advisor API is up and running!'

# ----------- Run Flask App ----------- #
if __name__ == '__main__':
    app.run(debug=True)
