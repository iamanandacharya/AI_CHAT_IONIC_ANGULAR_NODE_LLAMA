from flask import Flask, request, jsonify
import torch
import os
import sentencepiece as spm

app = Flask(__name__)

# Set paths
model_dir = ""
tokenizer_path = os.path.join(model_dir, "tokenizer.model")
model_path = os.path.join(model_dir, "consolidated.00.pth")

# Check tokenizer existence
print(f"Checking tokenizer path: {tokenizer_path}")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_path}")
# Load tokenizer
try:
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print("Tokenizer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer: {e}")

# Load model parameters
try:
    params = torch.load(model_path, map_location=torch.device("cpu"))
    from llama.model import LLaMAModel  # Ensure this is the correct import
    model = LLaMAModel(params)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Flask endpoint
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(input_ids)
        output_tokens = outputs[0].tolist()
        response = tokenizer.decode(output_tokens)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f"Failed to generate response: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
