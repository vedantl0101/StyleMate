import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
import pandas as pd

# Define paths for Male and Female assets
female_model_path = r"D:\ML Assignment\ML Project\StyleMate_\f.h5"
female_vectorizer_path = r"D:\ML Assignment\ML Project\StyleMate_\female_vectorizer.pkl"
female_label_encoder_path = r"D:\ML Assignment\ML Project\StyleMate_\female_label_encoder.pkl"
female_image_folder = r"D:\ML Assignment\ML Project\StyleMate_\Female"
female_ds_path = r"D:\ML Assignment\ML Project\StyleMate_\fm2.csv"

male_model_path = r"D:\ML Assignment\ML Project\StyleMate_\m.h5"
male_vectorizer_path = r"D:\ML Assignment\ML Project\StyleMate_\vectorizer.pkl"
male_label_encoder_path = r"D:\ML Assignment\ML Project\StyleMate_\label_encoder.pkl"
male_image_folder = r"D:\ML Assignment\ML Project\StyleMate_\Male"
male_ds_path = r"D:\ML Assignment\ML Project\StyleMate_\m4_cleaned.csv"

male_ds = pd.read_csv(male_ds_path)
female_ds = pd.read_csv(female_ds_path)


# Dictionaries mapping dress names to image file paths

female_dress_images = {
    "Bodycon": os.path.join(female_image_folder, "bodycon.jpg"),
    "Long Sleeve": os.path.join(female_image_folder, "longsleeve.jpg"),
    "Tunic": os.path.join(female_image_folder, "Tunic.jpg"),
    "Peplum": os.path.join(female_image_folder, "Peplum.jpg"),
    "Sweater": os.path.join(female_image_folder, "Sweater.jpg"),
    "Wrap": os.path.join(female_image_folder, "Wrap.jpg"),
    "Maxi": os.path.join(female_image_folder, "Maxi.jpg"),
    "A-Line": os.path.join(female_image_folder, "ALine.jpg"),
    "T-Shirt": os.path.join(female_image_folder, "tshirt.jpg"),
    "Knit Sweater": os.path.join(female_image_folder, "knite sweater.jpg"),
    "Sheath": os.path.join(female_image_folder, "sheath.jpg"),
    "Shift": os.path.join(female_image_folder, "Shift.jpg"),
    "Fit and Flare": os.path.join(female_image_folder, "fitflare.jpg"),
    "Pencil": os.path.join(female_image_folder, "pencil.jpg"),
    "Boho Midi": os.path.join(female_image_folder, "boho midi.jpg"),
    "Sundress": os.path.join(female_image_folder, "sundress.jpg"),
    "Pinafore": os.path.join(female_image_folder, "Pinafore.jpg"),
    "Kaftan": os.path.join(female_image_folder, "kaftan.jpg"),
    "Turtleneck": os.path.join(female_image_folder, "Turtleneck.jpg"),
    "Shirt": os.path.join(female_image_folder, "shirt.jpg")
}

male_dress_images = {
    "Suit": os.path.join(male_image_folder, "Suit.jpg"),
    "T-Shirt": os.path.join(male_image_folder, "T-Shirt.jpg"),
    "Sweater": os.path.join(male_image_folder, "Sweater.jpg"),
    "Jacket": os.path.join(male_image_folder, "Jacket.jpg"),
    "Jeans": os.path.join(male_image_folder, "Jeans.jpg"),
    "Chinos": os.path.join(male_image_folder, "Chinos.jpg"),
    "Shorts": os.path.join(male_image_folder, "Shorts.jpg"),
    "Hoodie": os.path.join(male_image_folder, "hoodie.jpg"),
    "Blazer": os.path.join(male_image_folder, "Blazer.jpg"),
    "Polo Shirt": os.path.join(male_image_folder, "Polo.jpg"),
    "Casual Shirt": os.path.join(male_image_folder, "casual.jpg"),
    "Formal Shirt": os.path.join(male_image_folder, "formal.jpg"),
    "Cardigan": os.path.join(male_image_folder, "Cardigan.jpg"),
    "Denim Jacket": os.path.join(male_image_folder, "denim.jpg"),
    "Bomber Jacket": os.path.join(male_image_folder, "Bomber Jacket.jpg"),
    "Trench Coat": os.path.join(male_image_folder, "trench.jpg"),
    "Tracksuit": os.path.join(male_image_folder, "tracksuit.jpg"),
    "Sweatpants": os.path.join(male_image_folder, "sweatpant.jpg"),
    "Vest": os.path.join(male_image_folder, "vest.jpg")
}
# Function to load assets based on gender
def load_assets(gender):
    if gender == "Male":
        model = load_model(male_model_path)
        with open(male_vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(male_label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        dataset = male_ds
        dress_images = male_dress_images
    else:  # Female
        model = load_model(female_model_path)
        with open(female_vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(female_label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        dataset = female_ds
        dress_images = female_dress_images
    return model, vectorizer, label_encoder, dress_images

# Preprocess input
def preprocess_input(input_text, ps, stop_words, vectorizer):
    input_text = re.sub('[^a-zA-Z]', ' ', input_text).lower()
    input_text = ' '.join([ps.stem(word) for word in input_text.split() if word not in stop_words])
    
    # Check for vocabulary attribute
    if hasattr(vectorizer, 'vocabulary_'):
        filtered_words = [word for word in input_text.split() if word in vectorizer.vocabulary_]
    else:
        raise ValueError("Loaded vectorizer does not have a vocabulary. Check the vectorizer file.")
    
    if not filtered_words:
        return None  # Handle empty inputs gracefully
    return ' '.join(filtered_words)

# Streamlit App
st.title("StyleMate - An Outfit Recommender System")
st.sidebar.header("User Preferences")

# Gender selection
gender = st.sidebar.selectbox("Select Gender:", ["Male", "Female"])
weather = st.sidebar.selectbox("Select Weather:", ["Snowy", "Sunny", "Pleasant", "Windy"])
color_palette = st.sidebar.selectbox("Select Color Palette:", ["Neutral", "Earth", "Bright", "Pastel"])
pattern = st.sidebar.selectbox("Select Pattern:", ["Stripes", "Geometric", "Solid colors", "Floral"])
feeling = st.sidebar.selectbox("How are you feeling?", ["Trendy", "Unique", "Casual", "Sophisticated"])
clothing_fit = st.sidebar.selectbox("Preferred Clothing Fit:", ["Tight", "Standard", "Loose", "Oversized"])

st.sidebar.write("### Selected Preferences:")
st.sidebar.write(f"- Gender: {gender}")
st.sidebar.write(f"- Weather: {weather}")
st.sidebar.write(f"- Color Palette: {color_palette}")
st.sidebar.write(f"- Pattern: {pattern}")
st.sidebar.write(f"- Feeling: {feeling}")
st.sidebar.write(f"- Clothing Fit: {clothing_fit}")

if st.button("Predict Outfit"):
    # Load assets dynamically based on gender
    model, vectorizer, label_encoder, dress_images = load_assets(gender)
    
    # Preprocess user input
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
        st.write(f"Predicted Dress Name ({gender}):", dress_name)
        
        if dress_name in dress_images:
            st.image(dress_images[dress_name], caption=dress_name)
        else:
            st.write("No image available for the predicted dress.")