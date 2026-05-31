from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import pandas as pd
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and encoders
model = joblib.load('flight_delay_predictor.pkl')
one_hot_encoders = joblib.load('one_hot_encoders.pkl')  # Load pre-fitted encoders
model_feature_names = joblib.load('model_feature_names.pkl')  # Load feature names used during training

# Login credentials
users = {'admin': 'password123'}

# Helper function to convert time string (HH:MM) to minutes
def time_to_minutes(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except ValueError:
        return None  # Return None if the time format is invalid

# Function to validate the input data
def validate_input(data):
    # Check if all required fields are provided
    required_fields = ['aircraft_type', 'origin_airport', 'destination_airport', 'departure_weekday', 'departure_time', 'airline']
    for field in required_fields:
        if not data.get(field):
            return False, f"Field '{field.replace('_', ' ').title()}' is required."

    # Check if departure time is valid
    departure_time = data['departure_time']
    if time_to_minutes(departure_time) is None:
        return False, "Departure Time should be in HH:MM format."

    # Validate 'Airline' and 'Destination Airport' against the categories in the encoder
    airline = data['airline']
    destination_airport = data['destination_airport']
    
    if airline not in one_hot_encoders['Airline'].categories_[0]:
        return False, f"Invalid airline: {airline}. Please select a valid airline."

    if destination_airport not in one_hot_encoders['Destination Airport'].categories_[0]:
        return False, f"Invalid destination airport: {destination_airport}. Please select a valid airport."

    return True, "Valid data."

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        session['logged_in'] = True
        return redirect(url_for('predict_page'))
    return "Invalid username or password"

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    if request.method == 'POST':
        # Collect input data
        input_data = {
            'aircraft_type': request.form['aircraft_type'],
            'origin_airport': request.form['origin_airport'],
            'destination_airport': request.form['destination_airport'],
            'departure_weekday': request.form['departure_weekday'],
            'departure_time': request.form['departure_time'],
            'arrival_time': request.form.get('arrival_time', ''),  # Optional
            'airline': request.form['airline']
        }

        # Validate the input data
        is_valid, message = validate_input(input_data)
        if not is_valid:
            return message  # Return plain message instead of JSON

        # Initialize encoded features
        encoded_features = []

        # One-hot encode categorical features
        for column, encoder in one_hot_encoders.items():
            if column in input_data and input_data[column] in encoder.categories_[0]:
                encoded_data = encoder.transform([[input_data[column]]]).flatten()
                encoded_features.extend(encoded_data)
            else:
                # Add zeros for unseen or missing categories
                encoded_features.extend([0] * len(encoder.categories_[0]))

        # Add departure time in minutes
        encoded_features.append(time_to_minutes(input_data['departure_time']))

        # Align encoded features to model_feature_names
        feature_vector = pd.DataFrame([encoded_features], columns=model_feature_names)
        for col in model_feature_names:
            if col not in feature_vector:
                feature_vector[col] = 0  # Add missing columns with 0s
        feature_vector = feature_vector[model_feature_names]  # Reorder columns

        # Make prediction
        prediction = model.predict(feature_vector)
        if prediction[0] == 0:
            result_message = "Congrats! Your flight is on time."
        else:
            result_message = "Oops! Your flight might be delayed."

        return result_message  # Return plain message instead of JSON

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
