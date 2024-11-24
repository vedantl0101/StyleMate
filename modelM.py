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

# Initialize NLTK
nltk.download('stopwords')

# Load and preprocess the cleaned dataset
file_path = r'D:\ML Assignment\ML Project\StyleMate_\m4_cleaned.csv'

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['T1'] = df['Weather'].str.cat(df['Color Palette'].astype(str), sep=',')
    df['T2'] = df['Pattern'].str.cat(df['Feeling'].astype(str), sep=',')
    df['T3'] = df['T1'].str.cat(df['T2'].astype(str), sep=',')
    df['T4'] = df['T3'].str.cat(df['Clothing Fit'].astype(str), sep=',')
    df['Text'] = df['T4'].str.cat(df['Description'].astype(str), sep=',')
    return df[['Text', 'Dress Name']]

data = load_and_preprocess_data(file_path)

# Preprocess text data
def preprocess_text(data):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')  # Retain "not" for sentiment
    corpus = []
    for text in data:
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
        corpus.append(text)
    return corpus

corpus = preprocess_text(data['Text'])
vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Dress Name'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

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

model = build_and_train_model(X_train, y_train, X_test, y_test, len(np.unique(y)))

# Save the model and assets
model.save('m.h5')
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model training complete and assets saved!")