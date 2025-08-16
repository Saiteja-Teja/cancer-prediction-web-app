import numpy as np
import pickle
import streamlit as st

# Load the trained model
with open("trained_model.sav", "rb") as f:
    loaded_model = pickle.load(f)

# Prediction function
def cancer_data(input_data):
    input_array = np.asarray(input_data)
    reshaped = input_array.reshape(1, -1)  # reshape for single prediction
    prediction = loaded_model.predict(reshaped)
    
    if prediction[0] == 0:
        return "✅ The person does NOT have breast cancer"
    else:
        return "⚠️ The person has Breast Cancer"

# Streamlit app
def main():
    st.title("🩺 Cancer Prediction Web App")
    st.write("Enter patient details to predict breast cancer")

    # User input
    mean_radius = st.number_input("Enter the mean radius value", min_value=0.0, format="%.5f")
    mean_texture = st.number_input("Enter the mean texture value", min_value=0.0, format="%.5f")
    mean_perimeter = st.number_input("Enter the mean perimeter value", min_value=0.0, format="%.5f")
    mean_area = st.number_input("Enter the mean area value", min_value=0.0, format="%.5f")
    mean_smoothness = st.number_input("Enter the mean smoothness value", min_value=0.0, format="%.5f")

    diagnosis = ""

    # Prediction button
    if st.button("🔍 Predict"):
        diagnosis = cancer_data([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness])
        st.success(diagnosis)

if __name__ == "__main__":
    main()
