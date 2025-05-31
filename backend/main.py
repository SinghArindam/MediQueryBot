import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
# CRITICAL: Replace with the base model ID you used for fine-tuning
BASE_MODEL_ID = "google/gemma-2b-it" # OR "google/gemma-1.1-2b-it" or your specific base
ADAPTER_MODEL_PATH = "../fintuned_model" # Path relative to this main.py file

# --- Model Loading ---
model = None
tokenizer = None

app = FastAPI(title="Medical QA Gemma API", version="0.1.0")

# --- CORS (Cross-Origin Resource Sharing) ---
# Allows your frontend (served from a different port/origin) to communicate with the API
origins = [
    "http://localhost",
    "http://localhost:8000", # Default FastAPI
    "null", # For local file:// access (needed for standalone HTML)
    # Add any other origins if you deploy elsewhere
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
    max_new_tokens: int = 250 # Max tokens for the answer

class AnswerResponse(BaseModel):
    question: str
    answer: str
    # debug_prompt: str # Optional: to see the full prompt sent to the model

# --- Model Loading Function (called at startup) ---
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer
    print(f"Loading base model: {BASE_MODEL_ID}")

    # Quantization config (optional, for saving memory)
    # If you have issues or enough VRAM, you can remove `quantization_config`
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # Set pad_token to eos_token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config, # Comment out if not using quantization
            device_map="auto", # Automatically uses CUDA if available
            trust_remote_code=True, # Gemma might require this
        )
        
        print(f"Loading PEFT adapter from: {ADAPTER_MODEL_PATH}")
        # Ensure the adapter path is correct relative to this script
        script_dir = os.path.dirname(__file__)
        absolute_adapter_path = os.path.join(script_dir, ADAPTER_MODEL_PATH)
        
        if not os.path.exists(absolute_adapter_path):
            raise FileNotFoundError(f"Adapter model directory not found at {absolute_adapter_path}. Make sure the path is correct.")
            
        model = PeftModel.from_pretrained(base_model, absolute_adapter_path)
        model = model.merge_and_unload() # Optional: merge LoRA weights for faster inference, consumes more VRAM initially
        model.eval() # Set model to evaluation mode

        print("Model and tokenizer loaded successfully.")
        
        # Test inference (optional)
        # test_prompt = "What is diabetes?"
        # messages = [{"role": "user", "content": test_prompt}]
        # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        # outputs = model.generate(input_ids=inputs, max_new_tokens=20)
        # print("Test generation:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    except Exception as e:
        print(f"Error loading model: {e}")
        # You might want to raise the exception or handle it gracefully
        # For now, we'll let FastAPI start, but endpoints will fail.
        # raise RuntimeError(f"Failed to load model: {e}") from e
        model = None # Ensure model is None if loading fails
        tokenizer = None

# --- API Endpoint ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        # Gemma IT models expect a specific chat format.
        # Format for Gemma:
        # <start_of_turn>user
        # {your_question}<end_of_turn>
        # <start_of_turn>model
        # The model will continue from here.
        
        # Using apply_chat_template is the recommended way
        messages = [
            {"role": "user", "content": request.question}
        ]
        # add_generation_prompt=True adds the <start_of_turn>model token
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        print(f"Generating response for: {request.question}")
        # print(f"Full prompt being sent to model: \n{prompt}") # For debugging

        with torch.no_grad(): # Important for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True, # Set to False for deterministic output
                temperature=0.7, # Adjust for creativity vs. factuality
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id # Ensure pad_token_id is set
            )
        
        # Decode the generated tokens
        # outputs[0] contains the full sequence (prompt + generation)
        # We need to slice off the prompt part.
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Raw answer: {answer_text}")

        return AnswerResponse(
            question=request.question,
            answer=answer_text.strip()
            # debug_prompt=prompt # Optional
        )

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this from the `med_qa_gemma/backend/` directory
    # Or adjust paths if running from `med_qa_gemma/`
    print("Starting FastAPI server...")
    print("Model loading will happen on the first request or at startup.")
    print(f"IMPORTANT: Ensure your BASE_MODEL_ID ('{BASE_MODEL_ID}') is correct and ADAPTER_MODEL_PATH ('{ADAPTER_MODEL_PATH}') points to your LoRA files.")
    uvicorn.run(app, host="0.0.0.0", port=8000)