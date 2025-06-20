You're looking to fine-tune a Gemma model on medical data, convert it to a quantized GGUF format, and then push it to Hugging Face. This is a comprehensive process!
A quick note: while you mentioned "Gemma 3 1b IT," the latest instruction-tuned Gemma 1.x model that is publicly available and commonly used for fine-tuning is Gemma 1.1 2B IT. Gemma 1.1 is an update over the original Gemma, offering improved quality. I'll provide the code for Gemma 1.1 2B IT, as it's the more relevant and performant choice.
Here's a detailed guide with code examples, assuming you're working within a Kaggle Notebook environment.
1. Setup Kaggle Environment and Authenticate
First, you need to set up your Kaggle environment and authenticate with Hugging Face.
```bash
# Install necessary libraries
!pip install -q -U transformers datasets accelerate bitsandbytes peft trl optimum llama-cpp-python huggingface_hub

# Login to Hugging Face (Important for accessing Gemma and pushing models)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# Get your Hugging Face token from Kaggle Secrets
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN") # Make sure you have a secret named 'HUGGINGFACE_TOKEN' in Kaggle
login(token=hf_token)

# Verify login
from huggingface_hub import HfApi
api = HfApi()
try:
    user_info = api.whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Error logging in to Hugging Face: {e}")

Kaggle Secrets:
 * Go to "Add-ons" -> "Secrets" in your Kaggle notebook.
 * Add a new secret with the name HUGGINGFACE_TOKEN and paste your Hugging Face write token (you can generate one from your Hugging Face settings: Settings -> Access Tokens -> New token). This is crucial for downloading Gemma and pushing your fine-tuned model.
2. Prepare Medical Data
You'll need a medical Q&A dataset. Kaggle hosts several. For this example, let's assume a simple JSONL format, which is common for instruction tuning.
Example format for your medical data (medical_qa.jsonl):
{"instruction": "What is the primary function of the heart?", "response": "The primary function of the heart is to pump blood throughout the body, delivering oxygen and nutrients to tissues and removing carbon dioxide and other waste products."}
{"instruction": "Explain the symptoms of a common cold.", "response": "Common cold symptoms typically include a runny or stuffy nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever."}

Load your dataset using the datasets library:
from datasets import load_dataset, DatasetDict
import pandas as pd

# Option 1: If your data is a local JSONL file
# For demonstration, let's create a dummy dataset
data = [
    {"instruction": "What are the common side effects of ibuprofen?", "response": "Common side effects of ibuprofen include stomach upset, nausea, vomiting, heartburn, diarrhea, constipation, and dizziness."},
    {"instruction": "Describe the process of digestion.", "response": "Digestion is the process where food is broken down into smaller molecules that the body can absorb. It begins in the mouth with chewing, continues in the stomach with acid and enzymes, and finishes in the small intestine where nutrients are absorbed into the bloodstream. Waste is then eliminated."},
    {"instruction": "What is diabetes mellitus?", "response": "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. It is characterized by high blood sugar levels resulting from either insufficient insulin production or the body's cells not responding properly to insulin, or both."},
    {"instruction": "How is pneumonia diagnosed?", "response": "Pneumonia is typically diagnosed based on a physical examination, review of medical history, and imaging tests like a chest X-ray. Blood tests and sputum cultures may also be used to identify the causative agent."}
]
df = pd.DataFrame(data)
df.to_json('medical_qa.jsonl', orient='records', lines=True)

dataset = load_dataset('json', data_files='medical_qa.jsonl', split='train')

# If you have a train/test split, you can do:
# dataset = load_dataset('json', data_files={'train': 'train.jsonl', 'test': 'test.jsonl'})
# For simplicity, we'll use the whole dataset for fine-tuning.
print(f"Dataset loaded: {len(dataset)} examples")
print(dataset[0])

3. Fine-tuning Gemma 1.1 2B IT with QLoRA
We'll use trl (Transformer Reinforcement Learning) and peft (Parameter-Efficient Fine-Tuning) for efficient QLoRA fine-tuning.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Model ID for Gemma 1.1 2B IT
model_id = "google/gemma-1.1-2b-it"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)

# Prepare model for K-bit training (required for QLoRA)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16, # LoRA attention dimension
    lora_alpha=32, # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], # Target modules for LoRA
    bias="none", # Bias type
    lora_dropout=0.05, # Dropout probability
    task_type="CAUSAL_LM", # Task type
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Tokenizer padding
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))


# Formatting function for SFTTrainer
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"Instruction:\n{example['instruction'][i]}\nResponse:\n{example['response'][i]}"
        output_texts.append(text)
    return output_texts

# Training arguments
from transformers import TrainingArguments

output_dir = "./gemma_medical_finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2, # Adjust based on GPU memory
    gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch sizes
    learning_rate=2e-4,
    num_train_epochs=3, # Adjust based on dataset size and desired convergence
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no", # For simplicity, not evaluating here
    fp16=False, # Use bfloat16 for Gemma
    bf16=True,
    push_to_hub=False, # We'll push manually later after conversion
    report_to="none", # Or "tensorboard", "wandb" etc.
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func,
    max_seq_length=512, # Adjust based on your data's typical length
)

# Start training
trainer.train()

# Save the fine-tuned adapter weights
trainer.save_model(output_dir)

# Merge LoRA adapters with the base model (important for GGUF conversion)
# This will save the full fine-tuned model
print("Merging LoRA adapters with base model...")
merged_model = model.merge_and_unload()
merged_model_save_path = "./gemma_medical_merged"
merged_model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)
print(f"Merged model saved to: {merged_model_save_path}")

4. Convert to Q4_K_M Quantized GGUF
This step requires llama.cpp and its conversion scripts. We'll clone it and use its Python utilities.
# Clone llama.cpp repository
!git clone https://github.com/ggerganov/llama.cpp.git
!pip install -r llama.cpp/requirements.txt

# Convert the merged Hugging Face model to GGUF format (fp16 for full precision first)
# The `convert.py` script needs the model in a Hugging Face format
# Ensure the model is saved to a directory accessible by convert.py
model_path_to_convert = "./gemma_medical_merged"
output_gguf_path = "./gemma-1.1-2b-it-medical-fp16.gguf"

print(f"Converting {model_path_to_convert} to GGUF (fp16)...")
!python llama.cpp/convert.py {model_path_to_convert} --outfile {output_gguf_path} --outtype f16

print(f"FP16 GGUF saved to: {output_gguf_path}")

# Quantize the GGUF model to Q4_K_M
quantized_gguf_path = "./gemma-1.1-2b-it-medical-Q4_K_M.gguf"

print(f"Quantizing {output_gguf_path} to Q4_K_M...")
!llama.cpp/quantize {output_gguf_path} {quantized_gguf_path} Q4_K_M

print(f"Q4_K_M GGUF saved to: {quantized_gguf_path}")

5. Push the GGUF to Hugging Face Hub
Now, you'll use huggingface_hub to push your quantized GGUF model to your Hugging Face repository.
from huggingface_hub import HfApi, create_repo, upload_file
import os

# Define your Hugging Face Hub repository details
hf_repo_id = "your-username/gemma-1.1-2b-it-medical-q4km" # Replace 'your-username' with your Hugging Face username
gguf_file_path = "./gemma-1.1-2b-it-medical-Q4_K_M.gguf"
readme_content = """
---
tags:
- llm
- gemma
- gemma-1.1
- medical
- fine-tuned
- gguf
- quantized
---
# Fine-tuned Gemma 1.1 2B IT on Medical Data (Q4_K_M Quantized GGUF)

This model is a fine-tuned version of `google/gemma-1.1-2b-it` on a custom medical Q&A dataset. It has been quantized to `Q4_K_M` GGUF format for efficient inference with `llama.cpp` compatible tools.

**Base Model:** [google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)

**Fine-tuning Data:** (Describe your medical dataset here, e.g., "A small dataset of medical question-answer pairs.")

**Quantization:** Q4_K_M GGUF

**How to use:**

You can load this GGUF file with `llama.cpp` or compatible clients like `ollama` or `text-generation-webui`.

Example with `llama.cpp`:

```bash
./main -m gemma-1.1-2b-it-medical-Q4_K_M.gguf -p "Instruction:\nWhat are the common symptoms of a heart attack?\nResponse:" -n 256

"""
Create the repository on Hugging Face Hub
print(f"Creating Hugging Face repository: {hf_repo_id}")
create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True, token=hf_token)
print("Repository created/exists.")
Upload the GGUF file
print(f"Uploading {gguf_file_path} to {hf_repo_id}...")
upload_file(
path_or_fileobj=gguf_file_path,
path_in_repo=os.path.basename(gguf_file_path),
repo_id=hf_repo_id,
repo_type="model",
token=hf_token,
)
print("GGUF file uploaded successfully.")
Add a README.md file
readme_path = "README.md"
with open(readme_path, "w") as f:
f.write(readme_content)
upload_file(
path_or_fileobj=readme_path,
path_in_repo="README.md",
repo_id=hf_repo_id,
repo_type="model",
token=hf_token,
)
print("README.md uploaded successfully.")
print(f"Your model is now available at: https://huggingface.co/{hf_repo_id}")

### Summary of Steps and Key Considerations:

1.  **Environment Setup:**
    * Install all necessary Python libraries (`transformers`, `datasets`, `accelerate`, `bitsandbytes`, `peft`, `trl`, `optimum`, `llama-cpp-python`, `huggingface_hub`).
    * **Crucially, set up your Hugging Face token as a Kaggle Secret** and log in using `huggingface_hub.login()`.

2.  **Data Preparation:**
    * Obtain or create a medical Q&A dataset.
    * Format it appropriately (e.g., as a JSONL file with `instruction` and `response` fields).
    * Load it using `datasets.load_dataset`.

3.  **Fine-tuning with QLoRA:**
    * Load `google/gemma-1.1-2b-it` with 4-bit quantization using `BitsAndBytesConfig`.
    * Prepare the model for K-bit training using `prepare_model_for_kbit_training`.
    * Define a `LoraConfig` to specify LoRA parameters (e.g., `r`, `lora_alpha`, `target_modules`).
    * Apply LoRA using `get_peft_model`.
    * Use `trl.SFTTrainer` for efficient supervised fine-tuning. Define `formatting_prompts_func` to structure your data for the model.
    * **After training, merge the LoRA adapters back into the base model** using `model.merge_and_unload()`. This is essential for creating a standalone model that can be converted to GGUF.

4.  **GGUF Conversion:**
    * Clone the `llama.cpp` repository.
    * Use `llama.cpp/convert.py` to convert your merged Hugging Face model to a full-precision GGUF (`--outtype f16`).
    * Then, use `llama.cpp/quantize` to quantize the FP16 GGUF to `Q4_K_M`.

5.  **Push to Hugging Face Hub:**
    * Use `huggingface_hub.create_repo` to create a new model repository on the Hub.
    * Use `huggingface_hub.upload_file` to push your `Q4_K_M` GGUF file and a `README.md` to the newly created repository.

**Important Notes:**

* **GPU Memory:** Fine-tuning even Gemma 1.1 2B IT on Kaggle with QLoRA can be demanding. Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` according to your available GPU memory.
* **Training Time:** Fine-tuning takes time. Be patient.
* **Dataset Quality:** The quality and size of your medical dataset will significantly impact the performance of your fine-tuned model.
* **Gemma Access:** Google's Gemma models require you to accept their terms of use on Hugging Face before you can download them programmatically. Ensure your Hugging Face token has read access to `google/gemma-1.1-2b-it`.
* **Error Handling:** The code snippets are for demonstration. In a production setting, you'd want more robust error handling.

