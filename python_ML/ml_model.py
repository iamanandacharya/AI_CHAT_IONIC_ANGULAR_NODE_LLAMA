from fastapi import FastAPI
from transformers import LlamaForCausalLM, LlamaTokenizer
from pydantic import BaseModel

app = FastAPI()

# Load the LLaMA model and tokenizer
model_name = ""
tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_name)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(request: ChatRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt", max_length=128000, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
