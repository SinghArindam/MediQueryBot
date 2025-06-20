from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def generate_medical_response(query: str, merged_model_path: str = "merged_gemma3_medical") -> dict:
    """
    Generates a medical question-answering response using a pre-trained model.

    Args:
        query (str): The medical question to be answered.
        merged_model_path (str): The path to the merged fine-tuned model.

    Returns:
        dict: A dictionary containing the generated response and the time taken.
    """
    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

    # Load merged model
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically map to GPU/CPU
    )

    # Prepare input prompt (adjust based on your fine-tuning prompt format)
    prompt = f"Question: {query}\nAnswer:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Adjust based on desired response length
        do_sample=True,  # Enable sampling for more natural responses
        temperature=0.7,  # Adjust for response creativity
        top_p=0.9  # Nucleus sampling
    )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    time_taken = end_time - start_time

    return {
        "response": response,
        "time_taken": f"responded in {time_taken:.2f} seconds"
    }

if __name__ == "__main__":
    # Example usage:
    medical_query = "What are the common symptoms of Type 2 Diabetes?"
    # Replace "merged_gemma3_medical" with the actual path to your merged model
    # For demonstration, let's assume it's in the current directory or a specified path.
    # If you don't have the model downloaded, this example will fail.
    # You would typically have downloaded and fine-tuned your model beforehand.
    
    # Placeholder for a valid model path. You need to provide your actual path.
    # For a real run, ensure 'merged_gemma3_medical' points to your model files.
    
    try:
        result = generate_medical_response(medical_query, merged_model_path="./merged_gemma3_medical/model.safetensors")
        print("Response:", result["response"])
        print(result["time_taken"])
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure 'merged_gemma3_medical' points to your actual fine-tuned model directory.")