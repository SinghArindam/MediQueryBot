# 🩺 MediQueryBot

Glass-morphic medical chatbot powered by **Custom Fine-Tuned (LoRA) LLM**  
– Multiple chats • Markdown answers • Collapsible reasoning • Latency footer

---

## ✨ Features
| Feature | Description |
| --- | --- |
| Custom Fine-Tuned (LoRA) LLM | Runs locally via `transformers` (CPU wheel, no API key). |
| Markdown rendering | Headlines, lists, code blocks, tables. |
| Multi-chat sidebar | Every chat stored as its own JSON file (`backend/chats/<id>.json`). |
| Reasoning reveal | Model’s `<think>` content collapses under “💭 Reasoning”. |
| Latency tracking | Response time is saved and shown. |
| AMOLED theme | pure-black UI with animated red→blue→green→gold halo. |

---

## 🏃‍♂️ Quick start (local)

```

git clone https://github.com/you/med-query-bot.git
cd med-query-bot

# build + run

docker build -t mediquery .
docker run -p 7860:7860 mediquery

# → http://localhost:7860

```

### Python dev mode

```

python -m venv .venv \&\& source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload

```
Open `frontend/index.html` directly or serve it with any static server.

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a new **“Docker”** Space.  
2. Push the whole repo (`git push`).  
3. The Space builds automatically; first boot downloads the ~1.2 GB model.

No tokens required; Custom Fine-Tuned (LoRA) LLM.

---

## 📂 Project structure
```

backend/         FastAPI + model loader
frontend/        HTML + Tailwind + JS (marked.js)
Dockerfile       Container for Spaces / local

```

---

## 🔧 Config & tips
* Change `MODEL_ID` in `backend/main.py` to any Custom Fine-Tuned (LoRA) LLM model that fits RAM.  
* Restrict CORS (`allow_origins`) before going public.  
* Each assistant turn is stored with `"latency": seconds` – useful for analytics.

---

Made with ❤️ by **Arindam Singh**