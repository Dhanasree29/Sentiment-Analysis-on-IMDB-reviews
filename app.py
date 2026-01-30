import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load models and tokenizer once
@st.cache_resource  # Cache to prevent reloading on every run
def load_models():
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lr_model = pickle.load(open('lr_model.pkl', 'rb'))
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    lstm_model = load_model('lstm_model.h5')
    return vectorizer, lr_model, tokenizer, lstm_model

vectorizer, lr_model, tokenizer, lstm_model = load_models()

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment:")

user_input = st.text_area("Review", "Type your review here...")
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input
        processed_input = preprocess_text(user_input)
        
        # Logistic Regression prediction
        input_tfidf = vectorizer.transform([processed_input])
        lr_pred = lr_model.predict(input_tfidf)[0]
        lr_sentiment = "Positive" if lr_pred == 1 else "Negative"
        
        # LSTM prediction
        input_seq = tokenizer.texts_to_sequences([processed_input])
        input_pad = pad_sequences(input_seq, maxlen=200)
        lstm_pred = (lstm_model.predict(input_pad)[0] > 0.5).astype("int32")[0]
        lstm_sentiment = "Positive" if lstm_pred == 1 else "Negative"
        
        # Display results
        st.write("### Predictions")
        st.write(f"**Logistic Regression**: {lr_sentiment}")
        st.write(f"**LSTM**: {lstm_sentiment}")
    else:
        st.write("Please enter a review.")

st.write("Note: Models were trained on the IMDB dataset with positive/negative sentiment labels.")