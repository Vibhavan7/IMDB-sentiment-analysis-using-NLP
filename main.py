import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
## Mapping of word index back to words
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    # Add 3 to account for the 3 special tokens added during preprocessing
    # (padding, start of sequence, unknown)
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    ## convert string to list of integers
    words = text.lower().split()    
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st 
## Streamlit app
st.title("IMDB Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Enter Review", "")
if st.button("Classify"):
    preprocessed_text = preprocess_text(user_input)
    prediction = model.predict(preprocessed_text)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Probability: {prediction[0][0]}")
else:
    st.write("Enter a movie review")
