## High-level Architecture

**Backend**

- Python + FastAPI (quick to scaffold, async friendly)
    
- MongoDB Atlas as the primary data store; use the official Motor async driver for non-blocking I/O1.
    
- Hugging Face transformers for LLM inference / fine-tuning workflows23.
    
- WebRTC + FastAPI WebSocket endpoints for real-time speech consultations.
    
- Background workers (Celery + Redis or FastAPI tasks) for heavy jobs: model fine-tuning, QR-code generation, BMI calculation, media transcoding.
    

**Frontend**

- React (or Next.js for SSR) → single codebase can serve web & desktop (Electron) later.
    
- Tailwind / Material UI for a clean, consistent look.
    
- Web Speech API for in-browser STT/TTS; fall back to Google Cloud Speech or Azure Speech for mobile/native.
    
- QuaggaJS / @zxing/library for QR scanning with the device camera.
    
- Socket.IO client for live voice sessions.
    

## MongoDB Collections & Indexes

|Collection|Mandatory Indexes|Purpose|
|---|---|---|
|users|`email (uniq)`, `phone (uniq)`, `userId (uniq)`|auth & profile look-ups|
|medical_cards|`userId (uniq)`|quick fetch of card at login / QR scan|
|chats|`userId`, `updatedAt` (TTL optional)|list user’s chats newest-first|
|messages|`chatId`, `createdAt`|infinite scroll in chat window|
|sessions|`token (uniq)`, `expiresAt` (TTL)|guest / PIN / QR sessions|
|models|`name (uniq)`|store fine-tuned model metadata|

## REST / WebSocket API Surface

|Method & Path|Auth|Description|
|---|---|---|
|POST /api/auth/register|none|create user, return JWT & generated medical card|
|POST /api/auth/login|none|email + password ⇒ JWT|
|POST /api/auth/qr-login|none|QR string ⇒ JWT|
|POST /api/auth/pin-login|none|`{userId, pin}` ⇒ JWT|
|GET /api/users/me|JWT|fetch profile & card|
|PATCH /api/users/me|JWT|update profile; BMI recalculated server-side|
|GET /api/users/me/avatar|public|serve profile image|
|POST /api/chats|JWT or guest|create (or resume) chat room|
|GET /api/chats|JWT|list user chats|
|GET /api/chats/:chatId|JWT|stream chat metadata|
|DELETE /api/chats/:chatId|JWT|remove chat|
|WS /ws/chats/:chatId|JWT or guest|bidirectional message stream (text & voice blobs)|
|POST /api/messages|JWT or guest|send text message → LLM reply|
|POST /api/messages/voice|JWT or guest|upload voice blob → STT → LLM → TTS|
|GET /api/models|JWT admin|list available fine-tuned models|
|POST /api/models/fine-tune|JWT admin|start background fine-tune job|
|GET /api/models/:id/status|JWT admin|polling endpoint|
|POST /api/qr|JWT|regenerate medical QR (returns PNG/SVG)|
|GET /api/news|public|optional health-blog sidebar feed|
|POST /api/sensors|JWT|future: push wearable data for RAG context|

(WS = WebSocket endpoint)

## Suggested Backend Tasks

- Scaffold FastAPI project: auth module, chat module, model module, util package for QR/BMI.
    
- Configure Motor client with env-based connection string to MongoDB Atlas1.
    
- Implement JWT auth middleware; optional refresh-token cookies.
    
- Write reusable Pydantic schemas for User, MedicalCard, Chat, Message.
    
- Integrate Hugging Face pipeline loader that can hot-swap between base and fine-tuned checkpoints23.
    
- Speech flow: voice blob → Whisper STT → chat completion → TTS (Coqui TTS or ElevenLabs API) → return audio.
    
- WebRTC signalling via FastAPI + Socket.IO for “phone-like” consultations.
    
- Celery worker for model training jobs; persist job state in MongoDB models collection.
    

## Suggested Frontend Tasks

- React router pages:
    
    - Landing / guest chat (V1)
        
    - Dashboard (V2) with call panel, chat pane, news card
        
    - Auth screens & QR-scanner (V3)
        
- Global context for auth token & selected model.
    
- Reusable ChatWindow component with left-aligned messages.
    
- Dropdown to switch models → emits REST call `/api/models/use`.
    
- VoiceChat component: WebRTC + Web Speech API fallback.
    
- Generate medical card view with printable QR.
    
- Store minimal data in IndexedDB for offline guest mode.
    

## External Services / SDKs to Wire Up

- Hugging Face Inference API or self-hosted transformers server23.
    
- Google or Azure Speech (if browser STT insufficient).
    
- Cloudinary/S3 for image uploads (avatars, QR PNGs).
    
- PubNub / Pusher (optional) if you prefer managed WebSocket infrastructure.
    

The endpoints above satisfy all user stories for V1–V3 while leaving space for sensor ingestion, face auth and more advanced RAG workflows later.

1. [programming.database_integration](https://www.perplexity.ai/search/programming.database_integration)
2. [programming.chatbots](https://www.perplexity.ai/search/programming.chatbots)
3. [programming.model_downloading](https://www.perplexity.ai/search/programming.model_downloading)