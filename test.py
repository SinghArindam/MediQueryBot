from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# Load tokenizer and model at startup
merged_model_path = "merged_gemma3_medical"  # Replace with your merged model path
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define request model
class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/query")
async def get_answer(query: QueryRequest):
    try:
        # Prepare input prompt
        prompt = f"Question: {query.question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)