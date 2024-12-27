Amharic Hate Speech Detection with Streamlit
---------------------------------------------------
Overview
This application uses a Hugging Face model to detect hate speech in Amharic text. It provides a simple and interactive web interface powered by Streamlit, allowing users to input text and receive predictions on whether the text contains hate speech or not.

Features
Simple UI for users to input text.
Real-time classification using a Hugging Face model (abdaiyan/amharic-hate-detection).
Outputs predictions with confidence scores.
Accessible through any web browser.
Installation and Setup

Requirements
-------------------
Python 3.9 or later (ensure compatibility with the packages listed in requirements.txt).
The following Python libraries:
streamlit
transformers
torch
Steps to Set Up Locally
Clone the Repository

git clone https://github.com/<your-github-username>/<your-repository-name>.git
cd <your-repository-name>
Set Up a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies Use the requirements.txt file to install the necessary packages.

pip install -r requirements.txt
Run the Streamlit App Launch the application locally.
streamlit run app.py
Access the App

Open your browser and go to:
arduino
Copy code
http://localhost:8501
Files in the Repository
1. app.py
The main application file:

Imports necessary libraries (streamlit, transformers).
Loads the pre-trained Hugging Face model for Amharic hate speech detection.
Provides a user interface for inputting text and displays predictions.

2. requirements.txt
Contains the dependencies needed to run the app
streamlit
transformers
torch
How It Works
User Input:

Users can input Amharic text in the text area provided by the Streamlit app.
Model Prediction:

The app uses the Hugging Face pipeline for text-classification, loading the abdaiyan/amharic-hate-detection model.
The model predicts whether the text is hate speech or not hate speech, along with a confidence score.
Result Display:

The app displays the prediction (Hate Speech or Not Hate Speech) and its confidence score.
Code Snippet: app.py
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
Expected Outputs
Input: Amharic text, e.g., "እናንተ ደደብ ጋዜጠኞች ከመፃፋቹ በፊት ተማሩ መማር ሼም የለው በቅሎ አቶልድም ይላል ሳይንስ ሳይነስ ይድፋቹ"
Prediction: Not Hate Speech
Confidence Score: Confidence: 1
Future Enhancements
Add multilingual support for detecting hate speech in other languages.
Include visualization for confidence scores and trends.
Deploy the app on a public platform like Streamlit Cloud or AWS for wider accessibility.
Deployment
To make the app accessible to others:

Deploy it using Streamlit Cloud:
Push your repository to GitHub.
Connect it with Streamlit Cloud.
Alternatively, deploy it using Docker or cloud platforms like AWS, Heroku, or GCP.