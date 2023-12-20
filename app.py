import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
model = load_model("ModelDL.h5")

# Function to preprocess input data
def preprocess_input(data):
    # Your preprocessing steps here
    # For example, normalizing the data using StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

# Function to make predictions
def predict_price(model, input_data, timesteps):
    # Repeat the input data to fill the required number of timesteps
    repeated_input = np.repeat(input_data, timesteps, axis=0)
    # Reshape input data to match the model's expected input shape
    input_data = np.reshape(repeated_input, (1, timesteps, input_data.shape[1]))
    # Make predictions
    prediction = model.predict(input_data)
    return prediction.flatten()[0]

# Streamlit app
def main():
    st.title("Real Estate Price Prediction")

    # Input form for user to enter features
    bed = st.slider("Number of Bedrooms", 1, 10, 3)
    bath = st.slider("Number of Bathrooms", 1, 10, 2)
    area = st.slider("Floor Area (in square meters)", 50, 500, 100)

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'area': [area]
    })

    # Preprocess the input data
    processed_input = preprocess_input(user_input.values)

    # Specify the number of timesteps (should match the value used during training)
    timesteps = 25

    # Make predictions using the loaded model
    prediction = predict_price(model, processed_input, timesteps)

    # Display the prediction
    st.subheader("Predicted Price:")
    st.write(f"Rp {prediction:,.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
