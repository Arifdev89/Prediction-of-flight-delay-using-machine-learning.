import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the training data
train_data = pd.read_csv('flight_data.csv')

# Initialize the one-hot encoders for each categorical feature
one_hot_encoders = {
    'Aircraft Type': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    'Origin Airport': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    'Destination Airport': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    'Airline': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    'Departure Weekday': OneHotEncoder(handle_unknown='ignore', sparse_output=False)
}

# Fit the one-hot encoders to the training data
for column, encoder in one_hot_encoders.items():
    encoder.fit(train_data[[column]])

# Save the encoders to a file
joblib.dump(one_hot_encoders, 'one_hot_encoders.pkl')

print("One-hot encoders saved successfully.")
