# syntax=docker/dockerfile:1

# ───────────────────────────────────────────────
# 1. Base image
# ───────────────────────────────────────────────
FROM python:3.11

# ───────────────────────────────────────────────
# 2. Create a non-root user – Hugging Face Spaces
#    recommends this pattern for security [2]
# ───────────────────────────────────────────────
RUN adduser --disabled-password --gecos '' user
USER user
WORKDIR /app

# ───────────────────────────────────────────────
# 3. Install Python dependencies
# ───────────────────────────────────────────────

COPY backend/requirements.txt ./requirements.txt

USER root 

# 3.1 · install normal deps from PyPI
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3.2 · install PyTorch CPU wheels from the dedicated index
RUN pip install --no-cache-dir \
     torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cpu \
 && pip cache purge

USER user

# ───────────────────────────────────────────────
# 4. Copy source (backend + frontend)
# ───────────────────────────────────────────────
COPY --chown=user . .

# ───────────────────────────────────────────────
# 5. Expose the port Spaces routes to the web
#    (must be 7860) [2][3]
# ───────────────────────────────────────────────
EXPOSE 7860

# ───────────────────────────────────────────────
# 6. Environment tweaks (optional)
# ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ───────────────────────────────────────────────
# 7. Launch FastAPI via Uvicorn
# ───────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
