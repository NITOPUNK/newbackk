from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the saved model
MODEL_PATH = 'best_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file {MODEL_PATH} not found. Please ensure the file exists in the correct directory.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not all(key in data for key in ['CALC_DISTANCE', 'DURATION_MIN']):
            return jsonify({
                'error': 'Missing required fields: CALC_DISTANCE and DURATION_MIN'
            }), 400

        # Create DataFrame from input data
        input_data = pd.DataFrame({
            'CALC_DISTANCE': [float(data['CALC_DISTANCE'])],
            'DURATION_MIN': [float(data['DURATION_MIN'])]
        })
        
        # Ensure column names are clean
        input_data.columns = input_data.columns.str.strip()
        
        # Make prediction
        predicted_battery_used = model.predict(input_data)[0]
        
        # Return prediction
        return jsonify({
            'predicted_battery_used': float(predicted_battery_used)
        }), 200
    
    except ValueError as ve:
        return jsonify({
            'error': f'Invalid input data: {str(ve)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running and model is loaded'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)