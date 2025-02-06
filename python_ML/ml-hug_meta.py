

from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)

# Define the model and tokenizer paths
model_name = ""  # Change this to your downloaded model

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Tokenizer and model loaded successfully.")

# Set the model to evaluation mode
model.eval()

@app.route('/')
def index():
    return jsonify({"message": "Model API running"})

# Define the API route for generating text
@app.route('/generate', methods=['POST'])
def generate():
    # Get the input prompt from the request
    data = request.json
    prompt = data.get('prompt', '')

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the model's response
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)

    # Decode the output tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
