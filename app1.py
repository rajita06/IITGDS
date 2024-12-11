import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Generate example data
data = pd.DataFrame({
    'size': np.random.randint(500, 5000, 100),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100),
    'price': np.random.randint(100000, 1000000, 100)
})

# Encode categorical data
location_map = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
data['location'] = data['location'].map(location_map)

X = data[['size', 'location']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the trained model
def predict_price(size, location):
    location_encoded = location_map.get(location, -1)
    if location_encoded == -1:
        raise ValueError("Invalid location. Choose from: Urban, Suburban, Rural")
    
    input_data = np.array([[size, location_encoded]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the house details to predict the price.")

# Input fields
size = st.number_input("Size of the house (in sq. ft):", min_value=500, max_value=5000, step=100)
location = st.selectbox("Location:", ["Urban", "Suburban", "Rural"])

if st.button("Predict Price"):
    try:
        prediction = predict_price(size, location)
        st.success(f"Predicted price for a house with size {size} in {location}: ${prediction:,.2f}")
    except ValueError as e:
        st.error(str(e))
