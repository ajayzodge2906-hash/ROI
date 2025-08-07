from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model and feature columns
model = joblib.load('roi_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

@app.route('/')
def home():
    return '🏠 Real Estate ROI Predictor API is live!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract input values
        location = data['Location']
        total_area = data['Total_Area']
        baths = data['Baths']

        # Create DataFrame with correct structure
        input_dict = {
            'Total_Area': [total_area],
            'Baths': [baths],
            **{col: [0] for col in feature_columns if col.startswith("Location_")}
        }

        location_col = f'Location_{location}'
        if location_col in feature_columns:
            input_dict[location_col] = [1]  # One-hot encode the matching location

        input_df = pd.DataFrame(input_dict)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict price per sqft
        price_per_sqft = model.predict(input_df)[0]

        # ROI Calculation
        future_price = total_area * price_per_sqft
        estimated_annual_rent = future_price * 0.025  # 2.5% rent yield
        total_rent_income = estimated_annual_rent * 5  # for 5 years
        roi = (total_rent_income / future_price) * 100

        return jsonify({
            'future_price': round(future_price, 2),
            'estimated_annual_rent': round(estimated_annual_rent, 2),
            'total_rent_income': round(total_rent_income, 2),
            'roi_percent': round(roi, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

