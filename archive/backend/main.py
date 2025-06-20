import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# PEFT is no longer imported here as we load a merged model

# --- Configuration ---
# Path to the directory containing your MERGED .safetensors model and tokenizer files
MERGED_MODEL_PATH = "merged_gemma3_medical\\model.safetensors" # Path relative to this main.py file

# --- Model Loading ---
model = None
tokenizer = None

app = FastAPI(title="Medical QA Gemma API (Merged Model)", version="0.2.0")

# --- CORS (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "null", # For local file:// access
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request and Response ---
class QuestionRequest(BaseModel):
    question: str
    max_new_tokens: int = 250

class AnswerResponse(BaseModel):
    question: str
    answer: str

# --- Model Loading Function (called at startup) ---
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer
    
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.join(script_dir, MERGED_MODEL_PATH)
    
    print(f"Attempting to load merged model and tokenizer from: {absolute_model_path}")

    if not os.path.isdir(absolute_model_path):
        error_msg = f"Model directory not found at {absolute_model_path}. Please ensure it contains the merged model files (e.g., model.safetensors, config.json, tokenizer.model)."
        print(error_msg)
        # Raise an error to prevent FastAPI from starting incorrectly if model is essential
        raise RuntimeError(error_msg)

    # Quantization config (optional, for saving memory)
    # If you have issues or enough VRAM, you can remove `quantization_config`
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
    )

    try:
        print(f"Loading tokenizer from: {absolute_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            absolute_model_path,
            trust_remote_code=True # Good practice, though often not needed for standard models
        )
        
        # Set pad_token to eos_token if not set (common for Gemma)
        if tokenizer.pad_token is None:
            print("Tokenizer `pad_token` not set. Setting it to `eos_token`.")
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Loading merged model from: {absolute_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            absolute_model_path,
            quantization_config=bnb_config, # Comment out if not using quantization
            device_map="auto",              # Automatically uses CUDA if available
            trust_remote_code=True,         # For custom model code, if any
            # torch_dtype=torch.bfloat16    # If not using BNB, consider this for memory/speed
        )
        
        model.eval() # Set model to evaluation mode

        print("Merged model and tokenizer loaded successfully.")
        
        # Optional: Test inference (uncomment to verify after loading)
        # test_prompt = "What is the capital of France?"
        # messages = [{"role": "user", "content": test_prompt}]
        # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        # outputs = model.generate(input_ids=inputs, max_new_tokens=20)
        # print("Test generation output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    except Exception as e:
        print(f"ERROR: Failed to load model or tokenizer from {absolute_model_path}: {e}")
        # Raising an exception here will prevent the server from starting if model loading fails.
        # This is generally better than letting it start in a broken state.
        model = None
        tokenizer = None
        raise RuntimeError(f"Failed to load model: {e}") from e


# --- API Endpoint (remains the same as previous version) ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        # Gemma IT models (and models fine-tuned from them) expect a specific chat format.
        messages = [
            {"role": "user", "content": request.question}
        ]
        # add_generation_prompt=True adds the <start_of_turn>model token
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        print(f"Generating response for: {request.question}")
        # print(f"Full prompt being sent to model: \n{prompt}") # For debugging

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Raw answer: {answer_text}")

        return AnswerResponse(
            question=request.question,
            answer=answer_text.strip()
        )

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for Merged Medical QA Gemma...")
    print(f"IMPORTANT: Ensure your MERGED_MODEL_PATH ('{MERGED_MODEL_PATH}') points to your directory containing model.safetensors and tokenizer files.")
    uvicorn.run(app, host="0.0.0.0", port=8000)