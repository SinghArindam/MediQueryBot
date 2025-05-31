import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# Sample medical Q&A data (replace with your own dataset for better results)
data = [
    {"prompts": "What is diabetes?", "responses": "Diabetes is a chronic condition characterized by high levels of sugar in the blood."},
    {"prompts": "How to treat a cold?", "responses": "Rest, drink plenty of fluids, and take over-the-counter medications."},
    {"prompts": "What are the symptoms of COVID-19?", "responses": "Common symptoms include fever, cough, and shortness of breath."},
    {"prompts": "How to prevent heart disease?", "responses": "Maintain a healthy diet, exercise regularly, and avoid smoking."},
]

# Load tokenizer and model
model_name = "google/gemma-3-1b-it"  # Instruction-tuned version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format data using chat template
formatted_data = []
for item in data:
    messages = [
        {"role": "user", "content": item["prompts"]},
        {"role": "assistant", "content": item["responses"]}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    formatted_data.append({"text": formatted_text})

dataset = Dataset.from_list(formatted_data)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
)

# Set up SFTTrainer for supervised fine-tuning
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./finetuned_model")
print("Model fine-tuning complete and saved to './finetuned_model'.")