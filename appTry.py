import streamlit as st
import pandas as pd
from PIL import Image

# Load dataset
file_path = 'clothes.csv'  # Path to your dataset
clothes_df = pd.read_csv(file_path)

# Streamlit webapp layout
st.title("StyleMate - An Outfit Recommender System")

# Gender selection
st.sidebar.header("User Preferences")
gender = st.sidebar.selectbox("Select Gender:", ["Male", "Female"])

# Options for user to choose from dataset columns
weather_options = clothes_df['Weather'].unique()
color_palette_options = clothes_df['Color Palette'].unique()
pattern_options = clothes_df['Pattern'].unique()
feeling_options = clothes_df['Feeling'].unique()
clothing_fit_options = clothes_df['Clothing Fit'].unique()
dress_name_options = clothes_df['Dress Name'].unique()

# User input fields
weather = st.sidebar.selectbox("Select Weather:", weather_options)
color_palette = st.sidebar.selectbox("Select Color Palette:", color_palette_options)
pattern = st.sidebar.selectbox("Select Pattern:", pattern_options)
feeling = st.sidebar.selectbox("How are you feeling?", feeling_options)
clothing_fit = st.sidebar.selectbox("Preferred Clothing Fit:", clothing_fit_options)
dress_name = st.sidebar.selectbox("Select Dress Type:", dress_name_options)

# Display an image placeholder for prediction result
image_placeholder = st.empty()

# Predict button
if st.sidebar.button("Predict"):
    # For simplicity, this code just displays a static image (you can replace this with model prediction code)
    # Example: Add logic here to pass the inputs to your model and get a predicted result
    st.write(f"Recommended outfit for a {feeling} feeling on a {weather} day:")
    
    # Display a sample image (Replace with the actual image based on prediction)
    sample_image_path = "sample_image.png"  # Replace with your model's predicted image path
    image = Image.open(sample_image_path)
    image_placeholder.image(image, caption=f"Recommended {dress_name}", use_column_width=True)

# Optionally, display the filtered options for debugging
st.sidebar.write("Selected Preferences:")
st.sidebar.write(f"Weather: {weather}")
st.sidebar.write(f"Color Palette: {color_palette}")
st.sidebar.write(f"Pattern: {pattern}")
st.sidebar.write(f"Feeling: {feeling}")
st.sidebar.write(f"Clothing Fit: {clothing_fit}")
st.sidebar.write(f"Dress Name: {dress_name}")