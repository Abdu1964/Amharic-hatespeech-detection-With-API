import streamlit as st
from transformers import pipeline

# Load the model from Hugging Face
model_name = "abdaiyan/amharic-hate-detection"
classifier = pipeline("text-classification", model=model_name)

# Streamlit UI
st.title("Amharic Hate Speech Detection")
st.write("Enter a text to check if it's hate speech or not.")

user_input = st.text_area("Input Text", "")
if user_input:
    result = classifier(user_input)

    # Map the result label to 'Hate Speech' or 'Not Hate Speech'
    label = result[0]['label']
    confidence = result[0]['score']
    
    if label == 'LABEL_0':
        label = 'Not Hate Speech'
    elif label == 'LABEL_1':
        label = 'Hate Speech'

    st.write(f"Prediction: {label} with confidence: {confidence:.2f}")
