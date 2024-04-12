import streamlit as st
import numpy as np
import pickle

# Load pre-trained model and TF-IDF vectorizer
@st.cache_data()
def load_model():
    with open('svm_tfidf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return model, tfidf_vectorizer

# Function to predict sentiment using TF-IDF features
def predict_sentiment(text, model, tfidf_vectorizer):
    # Vectorize text using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])
    
    # Make predictions using the model
    sentiment_code = model.predict(text_vectorized)[0]
    
    # Map sentiment codes to labels
    if sentiment_code == 1:
        return "Positive"
    elif sentiment_code == 0:
        return "Neutral"
    elif sentiment_code == -1:
        return "Negative"

# Streamlit UI
st.title('Sentiment Analysis')

# Load pre-trained model and TF-IDF vectorizer
model, tfidf_vectorizer = load_model()

# Input text box
text_input = st.text_input('Enter a sentence:', '')

# Button to predict sentiment
if st.button('Predict'):
    if text_input:
        # Predict sentiment
        sentiment = predict_sentiment(text_input, model, tfidf_vectorizer)
        
        # Display sentiment
        st.write('Sentiment:', sentiment)
    else:
        st.write('Please enter a sentence.')
