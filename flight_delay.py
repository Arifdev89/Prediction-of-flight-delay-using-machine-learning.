import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('flight_data.csv')

# Function to convert time string to minutes
def time_to_minutes(time_str):
    # Convert 'HH:MM' format to total minutes
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Convert 'Departure Time' and 'Arrival Time' columns to minutes
data['Departure Time'] = data['Departure Time'].apply(time_to_minutes)
data['Arrival Time'] = data['Arrival Time'].apply(time_to_minutes)

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
categorical_cols = ['Aircraft Type', 'Origin Airport', 'Destination Airport', 'Departure Weekday', 'Airline']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Create a new column for flight delays (1: delayed, 0: on-time)
# We assume a flight is delayed if the arrival time is greater than the departure time
data['Delayed'] = (data['Arrival Time'] > data['Departure Time']).astype(int)

# Drop unnecessary columns for training (e.g., Arrival Time as target is already used)
X = data.drop(columns=['Arrival Time', 'Delayed'])

# Target variable: 'Delayed'
y = data['Delayed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the trained model to a file for later use
joblib.dump(model, 'flight_delay_predictor.pkl')

# If you want to load and use the saved model:
# loaded_model = joblib.load('flight_delay_predictor.pkl')
# prediction = loaded_model.predict(new_data)
