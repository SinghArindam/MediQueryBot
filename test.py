from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Path to the merged fine-tuned model
merged_model_path = "merged_gemma3_medical"  # Replace with your merged model path

# Define quantization configuration (4-bit)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normalizing 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for efficiency
    bnb_4bit_use_double_quant=True  # Double quantization for better accuracy
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# Load merged model with quantization
model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    quantization_config=quant_config,
    device_map="auto",  # Automatically map to GPU/CPU
    torch_dtype=torch.float16
)

# Example medical QA query
query = "What are the common symptoms of Type 2 Diabetes?"

# Prepare input prompt (adjust based on your fine-tuning prompt format)
prompt = f"Question: {query}\nAnswer:"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,  # Adjust based on desired response length
    do_sample=True,  # Enable sampling for more natural responses
    temperature=0.7,  # Adjust for response creativity
    top_p=0.9  # Nucleus sampling
)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)