from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
def load_tabular_model():
    return load_model(r"tabular.pkl")

# Expected features from the trained model
expected_features = [
    'service', 'cleanliness', 'value', 'location', 'sleep_quality',
    'rooms', 'check_in_front_desk', 'bussiness_service', 'date_stayed', 'via_mobile'
]

# Load model
model = load_tabular_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = {
            'service': float(request.form['service']),
            'cleanliness': float(request.form['cleanliness']),
            'value': float(request.form['value']),
            'location': float(request.form['location']),
            'sleep_quality': float(request.form['sleep_quality']),
            'rooms': float(request.form['rooms']),
            'check_in_front_desk': float(request.form['check_in_front_desk']),
            'bussiness_service': float(request.form['bussiness_service']),
            'date_stayed': float(request.form['date_stayed']),
            'via_mobile': float(request.form['via_mobile'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Validate input features
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default value 0

        # Ensure correct column order
        input_df = input_df[expected_features]

        # Predict with the model
        input_array = input_df.to_numpy(dtype=np.float32)
        prediction = model.predict(input_array)
        overall_rating = round(prediction[0][0] * 5, 2)  # Convert to a 5-star scale

        return jsonify({"input": input_data, "predicted_rating": overall_rating})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)