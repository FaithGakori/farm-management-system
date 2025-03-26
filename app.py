from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

#load data
df = pd.read_excel("Price Prediction.xls")

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Label encode categorical features
categorical_columns = ['Market', 'Commodity', 'County']
label_encoders = {}

for col in categorical_columns:
    label_encoders[col] =LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

@app.route('/')
def home():
    return render_template('prices.html')

# Function to get user input
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    market = request.form['Market']
    commodity = request.form['Commodity']
    supply_volume = request.form['Supply_Volume']
    county = request.form['County']
    #day = int(request.form['Day'])
    #month = int(request.form['Month'])
    #year = int(request.form['Year'])
    date_str = request.form['date']  # Get the date string (e.g., '2025-03-06')
    year, month, day = map(int, date_str.split('-'))  # Split and convert to integers

# Convert categorical inputs to label-encoded form using the loaded encoders
    market_encoded = label_encoders['Market'].transform([market])[0]
    commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]
    county_encoded = label_encoders['County'].transform([county])[0]

# Create a DataFrame for the input data (ensure it has the same structure as your training data)
    input_data = pd.DataFrame([[market_encoded, commodity_encoded, supply_volume, county_encoded, day, month, year]], 
                              columns=['Market', 'Commodity', 'Supply Volume', 'County', 'Day', 'Month', 'Year'])
    
# make prediction using the trained model
    predicted_value = model.predict(input_data)

# render the results on html page
    return render_template('prices.html', prediction_text=f"Ksh {predicted_value[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get user input from the form
#     market = int(request.form['Market'])
#     commodity = int(request.form['Commodity'])
#     supply_volume = int(request.form['Supply_Volume'])
#     county = int(request.form['County'])
#     day = int(request.form['Day'])
#     month = int(request.form['Month'])
#     year = 2024  # Fixed year for predictions

#     # Create input data for the model
#     input_data = pd.DataFrame({
#         'Market': [market],
#         'Commodity': [commodity],
#         'Supply Volume': [supply_volume],
#         'County': [county],
#         'Day': [day],
#         'Month': [month],
#         'Year': [year]
#     })

#     # Make a prediction
#     prediction = model.predict(input_data)

#     # Render the result on the HTML page
#     return render_template('prices.html', prediction_text=f"${prediction[0]:.2f}")

# if __name__ == '__main__':
#     app.run(debug=True)
