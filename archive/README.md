# MediQueryBot - Medical Q&A with Fine-tuned Gemma 3 1B
Medical Query Bot for asking Simple Medical Queries. Custom Finetuned Gemma3 for medical dataset and implemented a frontend for smooth interaction.
This project fine-tunes the Gemma 3 1B model on medical Q&A data and provides a simple web interface using FastAPI and HTML/JavaScript.

**Note:** This is for educational purposes only and should not be used for actual medical advice.

## Prerequisites
- Python 3.8+
- A GPU with sufficient memory (e.g., 16GB+) for fine-tuning
- Internet access to download the base model

## Setup
1. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate
   ```
    On Windows:
   ```
    venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Fine-tune the model:**
   - Run the fine-tuning script:
     ```
     python finetune.py
     ```
   - This uses a sample dataset. Replace the `data` list in `finetune.py` with your own medical Q&A dataset for better results.

4. **Start the FastAPI server:**
   ```
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Interact with the application:**
   - Open your browser to `http://localhost:8000`.
   - Enter a medical question and click "Ask" to see the model's response.

## Directory Structure
- `finetune.py`: Fine-tuning script
- `app.py`: FastAPI backend
- `static/index.html`: Frontend interface


---
