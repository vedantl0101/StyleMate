import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load pre-trained assets
model = load_model('m.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocess input
def preprocess_input(input_text, ps, stop_words, vectorizer):
    input_text = re.sub('[^a-zA-Z]', ' ', input_text).lower()
    input_text = ' '.join([ps.stem(word) for word in input_text.split() if word not in stop_words])
    # Filter words based on vectorizer vocabulary
    filtered_words = [word for word in input_text.split() if word in vectorizer.vocabulary_]
    if not filtered_words:
        return None  # Handle empty inputs gracefully
    return ' '.join(filtered_words)

# Path to images
image_folder1 = r"D:\ML Assignment\ML Project\StyleMate_\Male"

# Dictionary mapping dress names to local image file paths
male_dress_images = {
    "Suit": os.path.join(image_folder1, "Suit.jpg"),
    "T-Shirt": os.path.join(image_folder1, "T-Shirt.jpg"),
    "Sweater": os.path.join(image_folder1, "Sweater.jpg"),
    "Jacket": os.path.join(image_folder1, "Jacket.jpg"),
    "Jeans": os.path.join(image_folder1, "Jeans.jpg"),
    "Chinos": os.path.join(image_folder1, "Chinos.jpg"),
    "Shorts": os.path.join(image_folder1, "Shorts.jpg"),
    "Hoodie": os.path.join(image_folder1, "hoodie.jpg"),
    "Blazer": os.path.join(image_folder1, "Blazer.jpg"),
    "Polo Shirt": os.path.join(image_folder1, "Polo.jpg"),
    "Casual Shirt": os.path.join(image_folder1, "casual.jpg"),
    "Formal Shirt": os.path.join(image_folder1, "formal.jpg"),
    "Cardigan": os.path.join(image_folder1, "Cardigan.jpg"),
    "Denim Jacket": os.path.join(image_folder1, "denim.jpg"),
    "Bomber Jacket": os.path.join(image_folder1, "Bomber Jacket.jpg"),
    "Trench Coat": os.path.join(image_folder1, "trench.jpg"),
    "Tracksuit": os.path.join(image_folder1, "tracksuit.jpg"),
    "Sweatpants": os.path.join(image_folder1, "sweatpant.jpg"),
    "Vest": os.path.join(image_folder1, "vest.jpg")
}

# Streamlit App Configuration
st.set_page_config(page_title="StyleMate - Outfit Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>StyleMate - An Outfit Recommender System</h1>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("User Preferences")
weather = st.sidebar.selectbox("Select Weather:", ["Snowy", "Sunny", "Pleasant", "Windy"])
color_palette = st.sidebar.selectbox("Select Color Palette:", ["Neutral", "Earth", "Bright", "Pastel"])
pattern = st.sidebar.selectbox("Select Pattern:", ["Stripes", "Geometric", "Solid colors", "Floral"])
feeling = st.sidebar.selectbox("How are you feeling?", ["Trendy", "Unique", "Casual", "Sophisticated"])
clothing_fit = st.sidebar.selectbox("Preferred Clothing Fit:", ["Tight", "Standard", "Loose", "Oversized"])

# Display selected preferences
st.sidebar.write("### Selected Preferences:")
# st.sidebar.write(f"- Gender: {gender}")
st.sidebar.write(f"- Weather: {weather}")
st.sidebar.write(f"- Color Palette: {color_palette}")
st.sidebar.write(f"- Pattern: {pattern}")
st.sidebar.write(f"- Feeling: {feeling}")
st.sidebar.write(f"- Clothing Fit: {clothing_fit}")

if st.button("Predict Dress Name"):
    input_str = f"{weather},{color_palette},{pattern},{feeling},{clothing_fit}"
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')  # Retain 'not' for sentiment processing
    preprocessed_input = preprocess_input(input_str, ps, stop_words, vectorizer)
    
    if preprocessed_input is None:
        st.error("Input does not match training data vocabulary. Please revise your selections.")
    else:
        input_vector = vectorizer.transform([preprocessed_input]).toarray()
        prediction = model.predict(input_vector)
        predicted_label = np.argmax(prediction, axis=1)
        dress_name = label_encoder.inverse_transform(predicted_label)[0]
        st.write("Predicted Dress Name:", dress_name)
        
        if dress_name in male_dress_images:
            st.image(male_dress_images[dress_name], caption=dress_name)
        else:
            st.write("No image available for the predicted dress.")