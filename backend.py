from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the Hugging Face model
model_name = "abdaiyan/amharic-hate-detection"
classifier = pipeline("text-classification", model=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    # Run prediction
    result = classifier(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
