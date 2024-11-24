import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score

# Initialize NLTK and download stopwords
nltk.download('stopwords')

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['T1'] = df['Weather'].str.cat(df['Color Palette'].astype(str), sep=',')
    df['T2'] = df['Pattern'].str.cat(df['Feeling'].astype(str), sep=',')
    df['T3'] = df['T1'].str.cat(df['T2'].astype(str), sep=',')
    df['T4'] = df['T3'].str.cat(df['Clothing Fit'].astype(str), sep=',')
    df = df.drop(columns=['Weather', 'Color Palette', 'Pattern', 'Feeling', 'Clothing Fit', 'T1', 'T2', 'T3'])
    df['Text'] = df['T4'].str.cat(df['Description'].astype(str), sep=',')
    df = df.drop(columns=['Description', 'T4'])
    return df

# Preprocess text data

nltk.download('stopwords')  # Ensure stopwords are downloaded

def preprocess_text(data):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))  # Use a set for faster lookup
    stop_words.discard('not')  # Retain "not" for sentiment
    corpus = []
    for text in data:
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
        corpus.append(text)
    return corpus


# Build and train the model
def build_and_train_model(X_train, y_train, X_test, y_test, output_dim):
    model = Sequential([
        Dense(units=128, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(units=64, activation='relu'),
        Dropout(0.3),
        Dense(units=output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])
    return model

# Predict function for real-time input
def predict_dress_name(input_text, vectorizer, model, label_encoder, ps, stop_words):
    input_text = re.sub('[^a-zA-Z]', ' ', input_text).lower()
    input_text = ' '.join([ps.stem(word) for word in input_text.split() if word not in stop_words])
    input_vector = vectorizer.transform([input_text]).toarray()
    prediction = model.predict(input_vector)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]

# Load dataset
# file_path = r'D:\ML Assignment\ML Project\Devil-Wears-Prada-main\Expanded_Female_Updated.csv'
# file_path = r'D:\ML Assignment\ML Project\Devil-Wears-Prada-main\Final_Updated_Female_Clothing_Fit.csv'
file_path = r'D:\ML Assignment\ML Project\Devil-Wears-Prada-main\Final_Expanded_Female_Clothing_Fit (1).csv'

df = load_and_preprocess_data(file_path)

# Process text and vectorize
corpus = preprocess_text(df['Text'])
vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:, 0].values)

# Save the label encoder and vectorizer
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
model = build_and_train_model(X_train, y_train, X_test, y_test, len(np.unique(y)))
model.save('neural_network_model.h5')

# Evaluate model accuracy
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_labels)

# Dictionary mapping dress names to image URLs
dress_images = {
    "Sheath": "https://i0.wp.com/fabrickated.com/wp-content/uploads/2015/01/sheath-dress-2.jpg",
    "Sweater": "https://www.lulus.com/images/product/xlarge/10839821_2196696.jpg?w=375&hdpi=1",
    "Turtleneck": "https://rukminim2.flixcart.com/image/850/1000/kihqz680-0/t-shirt/l/p/v/m-lady-high-black-39-lime-original-imafy9hhhmrtfkbu.jpeg?q=90&crop=false",
    "Maxi": "https://binfinite.in/cdn/shop/products/AA_c21d64c0-d27f-4331-9af9-0ea7f58bb065_1080x.jpg?v=1651586266",
    "Tunic": "https://cdn.shopify.com/s/files/1/0520/7395/5522/files/Chique_02561_360x.jpg?v=1710743487",
    "Shift": "https://assets.ajio.com/medias/sys_master/root/20230825/msmp/64e8b00fddf77915197c6140/-473Wx593H-466498789-black-MODEL.jpg",
    "Knit Sweater": "https://www.parents.com/thmb/P_GRQSTQzyPvBgT_SHC1ygbQvrE=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/anrabess-women-2023-fall-crewneck-long-sleeve-oversized-cable-knit-chunky-pullover-short-sweater-ffbff34c11664b9b842848a08f4054af.jpg",
    "Peplum": "https://m.media-amazon.com/images/I/61G-3cHhh5L._AC_UY1100_.jpg",
    "Boho Midi": "https://saffronthreadsclothing.com/cdn/shop/files/154_1d55281f-0886-4361-beb7-f18162e131be_864x1152.png?v=1710853990",
    "Wrap": "https://sassystripes.com/wp-content/uploads/2022/12/AHF9657.jpg",
    "Sundress": "https://m.media-amazon.com/images/I/81z7GNVCFuL._SY741_.jpg",
    "Fit and Flare": "https://i.etsystatic.com/5609612/r/il/26377e/2464428607/il_794xN.2464428607_sess.jpg",
    "Kaftan": "https://gillori.com/cdn/shop/products/GilloriKaftanDress.jpg?v=1634469459&width=1800",
    "Pencil": "https://imgs7.luluandsky.com/catalog/product/C/D/CDDE2930-BLACK_1.jpeg",
    "Bodycon": "https://assets.ajio.com/medias/sys_master/root/20230706/Dm2i/64a5c74deebac147fc509de9/-473Wx593H-466336756-black-MODEL.jpg",
    "A-Line": "https://rukminim2.flixcart.com/image/850/1000/l19m93k0/dress/y/1/9/m-bule-yellow-western-gurudev-fashion-original-imagcvckezdrfrjf.jpeg?q=90",
    "Shirt": "https://www.hancockfashion.com/cdn/shop/files/14053White_6_1800x1800.jpg?v=1698823457",
    "T-Shirt": "https://10d06a4d12b851f1b2d5-6729d756a2f36342416a9128f1759751.lmsin.net/1000010733006-Black-BLACK-1000010733006-8122021_01-1200.jpg",
    "Long Sleeve": "https://m.media-amazon.com/images/I/819Eb+tMZOL._AC_UY350_.jpg",
    "Pinafore": "https://10d06a4d12b851f1b2d5-6729d756a2f36342416a9128f1759751.lmsin.net/1000010733006-Black-BLACK-1000010733006-8122021_01-1200.jpg"
}

# Streamlit App
st.title("Dress Name Predictor")
st.write("Please select the following details to predict the dress name:")

weather_options = ["Snowy", "Sunny", "Pleasant", "Windy"]
color_palette_options = ["Earth", "Bright", "Pastel", "Neutral"]
pattern_options = ["Geometric", "Floral", "Solid colors", "Stripes"]
feeling_options = ["Unique", "Casual", "Sophisticated", "Trendy"]
clothing_fit_options = ["Oversized", "Loose", "Standard", "Tight"]

weather = st.radio("1 - What is the expected weather during the event?", weather_options, index=0)
color_palette = st.radio("2 - Choose a color palette:", color_palette_options, index=0)
pattern = st.radio("3 - Select a pattern:", pattern_options, index=0)
feeling = st.radio("4 - How do you want to feel in the outfit?", feeling_options, index=0)
clothing_fit = st.radio("5 - Choose a clothing fit:", clothing_fit_options, index=0)

if st.button("Predict Dress Name"):
    input_str = f"{weather},{color_palette},{pattern},{feeling},{clothing_fit}"
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    dress_name = predict_dress_name(input_str, vectorizer, model, label_encoder, ps, stop_words)
    st.write("Predicted Dress Name:", dress_name)
    if dress_name in dress_images:
        st.image(dress_images[dress_name], caption=dress_name)
    else:
        st.write("No image available for the predicted dress.")

if st.button("Evaluate Model Accuracy"):
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")


