import streamlit as st
from transformers import pipeline

# Load the model from Hugging Face
model_name = "abdaiyan/amharic-hate-detection"
classifier = pipeline("text-classification", model=model_name)

# Streamlit UI
st.set_page_config(
    page_title="Amharic Hate Speech Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# App title and description
st.title("🛡️ Amharic Hate Speech Detection")
st.markdown(
    """
    This tool uses advanced machine learning to detect whether an Amharic sentence is 
    **Hate Speech** or **Not Hate Speech**. Enter a sentence and click "Analyze" to see the prediction.
    """
)

# Input text area
user_input = st.text_area("Enter your text below:", height=100)

# Button to trigger prediction
if st.button("Analyze"):
    if user_input.strip():
        # Run the classification
        result = classifier(user_input)
        label = result[0]["label"]
        confidence = result[0]["score"]

        # Map the result label to human-readable text
        if label == "LABEL_0":
            label_text = "Not Hate Speech"
            color = "green"
        elif label == "LABEL_1":
            label_text = "Hate Speech"
            color = "red"

        # Display the result in a separate box
        st.markdown(
            f"""
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9;">
                <h3 style="color: {color}; text-align: center;">Prediction: {label_text}</h3>
                <p style="text-align: center; font-size: 18px;">Confidence Score: <b>{confidence:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a valid text to analyze.")
