from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn # Import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned model and tokenizer
# IMPORTANT: Ensure the 'finetuned_model' directory exists and contains
# the model and tokenizer files (e.g., config.json, pytorch_model.bin, tokenizer.json, etc.)
try:
    model_path = "./finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please ensure 'finetuned_model' directory exists and contains valid model files.")
    # Exit or handle the error gracefully if model loading is critical
    # For now, we'll let the app start but it won't be functional without the model.
    tokenizer = None
    model = None
    device = "cpu"


# Endpoint to handle questions
@app.post("/ask")
async def ask(question: str = Body(...)):
    if not tokenizer or not model:
        return {"answer": "Model is not loaded. Please check the server logs for errors."}

    try:
        # Format input using chat template
        messages = [{"role": "user", "content": question}]
        # `apply_chat_template` requires `add_generation_prompt=True` for proper formatting
        # when used for generation, especially with models that expect a specific turn structure.
        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and generate response
        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
        # Ensure that `pad_token_id` is set, especially if the model's tokenizer doesn't have it by default.
        # This is crucial for `model.generate` to handle variable input lengths.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id # Or a specific pad token if defined by the model
        
        outputs = model.generate(
            **inputs, 
            max_length=200, 
            num_return_sequences=1, 
            temperature=0.7, # Add temperature for more varied outputs
            do_sample=True,  # Enable sampling for temperature to take effect
            pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
        )
        # Decode the generated output, skipping special tokens
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process the answer to remove the original prompt if it's echoed
        # This is a common issue with chat templates and generation.
        # You might need to adjust this based on how your specific model's chat template works.
        # A simple approach is to find the last occurrence of the user's question and take everything after it.
        # This is a heuristic and might need refinement.
        if formatted_input in answer:
            answer = answer.split(formatted_input)[-1].strip()

        return {"answer": answer}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"answer": f"An error occurred while generating the response: {e}"}

# Serve the frontend
# Ensure that a 'static' directory exists in the same location as app.py
# and that 'index.html' is inside it.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found in 'static' directory.</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>An error occurred: {e}</h1>", status_code=500)

# This block ensures the app runs automatically when the script is executed
if __name__ == "__main__":
    # The host "0.0.0.0" makes the app accessible from other devices on the network.
    # For local development, "127.0.0.1" or "localhost" also works.
    uvicorn.run(app, host="0.0.0.0", port=8000)
