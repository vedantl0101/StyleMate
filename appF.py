import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load pre-trained assets
model = load_model('neural_network_model.h5')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocess input
def preprocess_input(input_text, ps, stop_words):
    input_text = re.sub('[^a-zA-Z]', ' ', input_text).lower()
    input_text = ' '.join([ps.stem(word) for word in input_text.split() if word not in stop_words])
    return input_text

image_folder = r"D:\ML Assignment\ML Project\StyleMate_\Female"

# Dictionary mapping dress names to local image file paths
female_dress_images = {
    "Bodycon": os.path.join(image_folder, "bodycon.jpg"),
    "Long Sleeve": os.path.join(image_folder, "longsleeve.jpg"),
    "Tunic": os.path.join(image_folder, "Tunic.jpg"),
    "Peplum": os.path.join(image_folder, "Peplum.jpg"),
    "Sweater": os.path.join(image_folder, "Sweater.jpg"),
    "Wrap": os.path.join(image_folder, "Wrap.jpg"),
    "Maxi": os.path.join(image_folder, "Maxi.jpg"),
    "A-Line": os.path.join(image_folder, "ALine.jpg"),
    "T-Shirt": os.path.join(image_folder, "tshirt.jpg"),
    "Knit Sweater": os.path.join(image_folder, "knite sweater.jpg"),
    "Sheath": os.path.join(image_folder, "sheath.jpg"),
    "Shift": os.path.join(image_folder, "Shift.jpg"),
    "Fit and Flare": os.path.join(image_folder, "fitflare.jpg"),
    "Pencil": os.path.join(image_folder, "pencil.jpg"),
    "Boho Midi": os.path.join(image_folder, "boho midi.jpg"),
    "Sundress": os.path.join(image_folder, "sundress.jpg"),
    "Pinafore": os.path.join(image_folder, "Pinafore.jpg"),
    "Kaftan": os.path.join(image_folder, "kaftan.jpg"),
    "Turtleneck": os.path.join(image_folder, "Turtleneck.jpg"),
    "Shirt": os.path.join(image_folder, "shirt.jpg")
}


# Streamlit App
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
    stop_words.discard('not')
    preprocessed_input = preprocess_input(input_str, ps, stop_words)
    input_vector = vectorizer.transform([preprocessed_input]).toarray()
    prediction = model.predict(input_vector)
    predicted_label = np.argmax(prediction, axis=1)
    dress_name = label_encoder.inverse_transform(predicted_label)[0]
    st.write("Predicted Dress Name:", dress_name)
    if dress_name in female_dress_images:
        st.image(female_dress_images[dress_name], caption=dress_name)
    else:
        st.write("No image available for the predicted dress.")