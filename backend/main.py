"""
MediQueryBot · FastAPI backend
– Qwen3-0.6B (local, CPU/GPU)
– multi-chat persistence  (1 JSON per chat)
– latency saved and returned
"""

from __future__ import annotations
import json, uuid, asyncio, time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ───────────────────────────────────────────
# 1.  Load model
# ───────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-0.6B"
print("⏳ Loading Qwen3-0.6B …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
print("✅ Model ready")

SYS_PROMPT = (
    "You are MediQueryBot, a helpful medical assistant. "
    "Provide concise, evidence-based information. "
    "Always finish with: "
    "'This information is educational; not a substitute for professional advice.'"
)

@torch.inference_mode()
def qwen_chat(user: str) -> str:
    tmpl = tokenizer.apply_chat_template(
        [{"role":"system","content":SYS_PROMPT},
         {"role":"user",  "content":user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(tmpl, return_tensors="pt").to(model.device)
    ids = model.generate(
        **inputs,
        max_new_tokens=768,
        temperature=0.4,
        do_sample=True
    )
    gen_ids = ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ───────────────────────────────────────────
# 2.  Persistence helpers
# ───────────────────────────────────────────
ROOT      = Path(__file__).parent
CHAT_DIR  = ROOT / "chats"
CHAT_DIR.mkdir(exist_ok=True)
_lock = asyncio.Lock()

def _chat_path(cid:str)->Path: return CHAT_DIR / f"{cid}.json"

def _load(cid:str)->dict:
    fp=_chat_path(cid)
    if not fp.exists():
        raise HTTPException(404,"chat not found")
    with fp.open("r",encoding="utf-8") as f:
        return json.load(f)

def _save(data:dict)->None:
    with _chat_path(data["id"]).open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def _list_chats()->list[dict]:
    out=[]
    for fp in CHAT_DIR.glob("*.json"):
        with fp.open("r",encoding="utf-8") as f:
            j=json.load(f)
            title=next((m["content"] for m in j["messages"] if m["role"]=="user"),"Untitled")
            out.append({"id":j["id"],"created_at":j["created_at"],
                        "title":title[:40]+("…" if len(title)>40 else "")})
    out.sort(key=lambda x:x["created_at"], reverse=True)
    return out

# ───────────────────────────────────────────
# 3.  FastAPI app
# ───────────────────────────────────────────
app = FastAPI(title="MediQueryBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONT = ROOT.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONT), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONT / "index.html")

# pydantic
class Msg(BaseModel): message:str
class Info(BaseModel): id:str; created_at:str

# ── routes ─────────────────────────────────
@app.post("/chats", response_model=Info)
async def new_chat():
    cid=uuid.uuid4().hex[:8]
    data={"id":cid,"created_at":datetime.utcnow().isoformat(),"messages":[]}
    async with _lock: _save(data)
    return data

@app.get("/chats")
async def list_chats(): return _list_chats()

@app.get("/chats/{cid}/history")
async def history(cid:str=FPath(...,min_length=1)):
    async with _lock: return _load(cid)["messages"]

@app.post("/chats/{cid}/ask")
async def ask(msg:Msg, cid:str=FPath(...,min_length=1)):
    async with _lock: data=_load(cid)

    t0=time.perf_counter()
    answer=await asyncio.get_event_loop().run_in_executor(None, qwen_chat, msg.message)
    latency=round(time.perf_counter()-t0,1)

    async with _lock:
        data=_load(cid)
        data["messages"] += [
            {"role":"user","content":msg.message},
            {"role":"assistant","content":answer,"latency":latency}
        ]
        _save(data)

    return {"reply":answer,"latency":latency}
