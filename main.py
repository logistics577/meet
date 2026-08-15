"""
Meet App - Production-style single-file FastAPI backend.

Features:
- SQLite persistence (users, meetings, messages)
- Email-based login (no password) -> bearer token
- Create / join meeting links
- WebSocket signaling for WebRTC (offer/answer/ICE) + live chat + join notifications
- STUN/TURN config endpoint
- Screen-share is handled client side (renegotiation over the same signaling channel)
- Chatbot popup (Groq llama-3.3-70b-versatile) - only unlocked for a specific email
- Custom logging middleware + CORS middleware
- Serves the frontend (index.html) at "/"

Run:
    pip install fastapi "uvicorn[standard]" websockets httpx --break-system-packages
    export GROQ_API_KEY=xxxx           # required for chatbot
    export TURN_URL=turn:your.turn.server:3478   # optional
    export TURN_USERNAME=xxxx                    # optional
    export TURN_CREDENTIAL=xxxx                  # optional
    python3 main.py
"""

import os
import time
import uuid
import asyncio
import sqlite3
import secrets
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
import httpx
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

load_dotenv()
# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DB_PATH = os.environ.get("MEET_DB_PATH", os.path.join(os.path.dirname(__file__), "meet.db"))
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Only this email gets the AI chatbot popup unlocked (per spec).
CHATBOT_ALLOWED_EMAIL = os.environ.get("CHATBOT_ALLOWED_EMAIL", "abc@gmail.com").lower()

TURN_URL = os.environ.get("TURN_URL", "")
TURN_USERNAME = os.environ.get("TURN_USERNAME", "")
TURN_CREDENTIAL = os.environ.get("TURN_CREDENTIAL", "")

# Max recording upload size (bytes) - keep SQLite BLOBs sane. Default 25MB.
MAX_RECORDING_BYTES = int(os.environ.get("MAX_RECORDING_BYTES", 25 * 1024 * 1024))

# Keep-alive: on platforms like Render's free tier, the service spins down after
# ~15 min with no inbound HTTP traffic. If SELF_PING_URL (or RENDER_EXTERNAL_URL,
# which Render sets automatically) is present, a background task pings /health
# on an interval so an *already-running* instance never goes idle-cold.
# IMPORTANT: this cannot wake an instance that has already spun down - only an
# external monitor (UptimeRobot, cron-job.org, etc.) hitting a public URL can do
# that. Use both for a live interview: this task keeps it warm, an external
# monitor guarantees the first request always lands on a warm instance.
SELF_PING_URL = os.environ.get("SELF_PING_URL") or os.environ.get("RENDER_EXTERNAL_URL", "")
KEEP_ALIVE_INTERVAL_SECONDS = int(os.environ.get("KEEP_ALIVE_INTERVAL_SECONDS", 600))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("meetapp")

# --------------------------------------------------------------------------
# DB layer
# --------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                host_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (host_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                sender_email TEXT,
                text TEXT NOT NULL,
                is_bot INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            );

            CREATE TABLE IF NOT EXISTS participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                user_id TEXT,
                name TEXT NOT NULL,
                email TEXT,
                joined_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                user_id TEXT,
                recorder_name TEXT,
                mime_type TEXT NOT NULL,
                audio_blob BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            );
            """
        )
    logger.info("SQLite ready at %s", DB_PATH)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------

def get_user_by_token(token: str) -> Optional[sqlite3.Row]:
    with get_db() as db:
        row = db.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
        return row


def require_user(authorization: Optional[str] = Header(None)) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


# --------------------------------------------------------------------------
# App + middleware
# --------------------------------------------------------------------------

app = FastAPI(
    title="Meet App",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Custom FastAPI middleware: logs every request with timing + status."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response


app.add_middleware(RequestLogMiddleware)


@app.on_event("startup")
async def on_startup():
    init_db()
    if SELF_PING_URL:
        asyncio.create_task(_keep_alive_loop())
        logger.info("Keep-alive enabled: pinging %s every %ss", SELF_PING_URL, KEEP_ALIVE_INTERVAL_SECONDS)
    else:
        logger.info("Keep-alive disabled (set SELF_PING_URL or RENDER_EXTERNAL_URL to enable)")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str


class CreateMeetingRequest(BaseModel):
    title: Optional[str] = "Untitled meeting"


class ChatbotRequest(BaseModel):
    meeting_id: str
    message: str


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------

@app.post("/api/auth/login")
def login(payload: LoginRequest):
    email = payload.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    first_name = email.split("@")[0].split(".")[0].split("+")[0]
    first_name = first_name.capitalize() if first_name else "Guest"

    with get_db() as db:
        existing = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            token = secrets.token_hex(16)
            db.execute("UPDATE users SET token = ? WHERE id = ?", (token, existing["id"]))
            user_id = existing["id"]
            name = existing["name"]
        else:
            user_id = str(uuid.uuid4())
            token = secrets.token_hex(16)
            name = first_name
            db.execute(
                "INSERT INTO users (id, email, name, token, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, name, token, now_iso()),
            )

    return {
        "token": token,
        "user_id": user_id,
        "email": email,
        "name": name,
        "first_name": name.split(" ")[0],
        "chatbot_unlocked": email == CHATBOT_ALLOWED_EMAIL,
    }


# --------------------------------------------------------------------------
# Meeting routes
# --------------------------------------------------------------------------

def short_id(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


@app.post("/api/meetings/create")
def create_meeting(payload: CreateMeetingRequest, user: sqlite3.Row = Depends(require_user)):
    meeting_id = short_id()
    with get_db() as db:
        # extremely unlikely collision, but guard anyway
        while db.execute("SELECT 1 FROM meetings WHERE id = ?", (meeting_id,)).fetchone():
            meeting_id = short_id()
        db.execute(
            "INSERT INTO meetings (id, title, host_id, created_at) VALUES (?, ?, ?, ?)",
            (meeting_id, payload.title or "Untitled meeting", user["id"], now_iso()),
        )
    return {
        "meeting_id": meeting_id,
        "title": payload.title or "Untitled meeting",
        "host_name": user["name"],
        "join_path": f"/room/{meeting_id}",
    }


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    with get_db() as db:
        meeting = db.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")
        host = db.execute("SELECT name FROM users WHERE id = ?", (meeting["host_id"],)).fetchone()
        participant_count = db.execute(
            "SELECT COUNT(*) AS c FROM participants WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()["c"]
    return {
        "meeting_id": meeting["id"],
        "title": meeting["title"],
        "host_name": host["name"] if host else "Unknown",
        "created_at": meeting["created_at"],
        "participant_count": participant_count,
    }


@app.get("/api/meetings/{meeting_id}/messages")
def get_messages(meeting_id: str):
    with get_db() as db:
        rows = db.execute(
            "SELECT sender_name, sender_email, text, is_bot, created_at "
            "FROM messages WHERE meeting_id = ? ORDER BY id ASC",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/ice-config")
def ice_config():
    """STUN/TURN servers for the browser's RTCPeerConnection."""
    servers = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    if TURN_URL:
        turn_entry = {"urls": [TURN_URL]}
        if TURN_USERNAME:
            turn_entry["username"] = TURN_USERNAME
        if TURN_CREDENTIAL:
            turn_entry["credential"] = TURN_CREDENTIAL
        servers.append(turn_entry)
    return {"iceServers": servers}


# --------------------------------------------------------------------------
# Call recordings (audio) - stored as BLOBs in SQLite
# --------------------------------------------------------------------------

@app.post("/api/recordings/upload")
async def upload_recording(
    meeting_id: str = Form(...),
    file: UploadFile = File(...),
    user: sqlite3.Row = Depends(require_user),
):
    with get_db() as db:
        meeting = db.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

    data = await file.read()
    if len(data) > MAX_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail="Recording too large")
    if not data:
        raise HTTPException(status_code=400, detail="Empty recording")

    with get_db() as db:
        cur = db.execute(
            "INSERT INTO recordings (meeting_id, user_id, recorder_name, mime_type, audio_blob, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (meeting_id, user["id"], user["name"], file.content_type or "audio/webm", data, len(data), now_iso()),
        )
        recording_id = cur.lastrowid

    logger.info("Stored recording %s for meeting %s (%d bytes)", recording_id, meeting_id, len(data))
    return {"recording_id": recording_id, "size_bytes": len(data)}


@app.get("/api/meetings/{meeting_id}/recordings")
def list_recordings(meeting_id: str, user: sqlite3.Row = Depends(require_user)):
    with get_db() as db:
        rows = db.execute(
            "SELECT id, recorder_name, mime_type, size_bytes, created_at FROM recordings "
            "WHERE meeting_id = ? ORDER BY id DESC",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/recordings/{recording_id}")
def get_recording(recording_id: int, user: sqlite3.Row = Depends(require_user)):
    with get_db() as db:
        row = db.execute(
            "SELECT audio_blob, mime_type FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")
    return Response(content=row["audio_blob"], media_type=row["mime_type"])


# --------------------------------------------------------------------------
# Health + keep-alive (see SELF_PING_URL note above)
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "time": now_iso()}


async def _keep_alive_loop():
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
            try:
                await client.get(SELF_PING_URL.rstrip("/") + "/health")
                logger.info("Keep-alive ping sent to %s", SELF_PING_URL)
            except Exception as e:
                logger.warning("Keep-alive ping failed: %s", e)


# --------------------------------------------------------------------------
# Chatbot (Groq) - only unlocked for CHATBOT_ALLOWED_EMAIL
# --------------------------------------------------------------------------

@app.post("/api/chatbot")
async def chatbot(payload: ChatbotRequest, user: sqlite3.Row = Depends(require_user)):
    if user["email"].lower() != CHATBOT_ALLOWED_EMAIL:
        raise HTTPException(status_code=403, detail="Chatbot is not available for this account")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server")

    with get_db() as db:
        db.execute(
            "INSERT INTO messages (meeting_id, sender_name, sender_email, text, is_bot, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (payload.meeting_id, user["name"], user["email"], payload.message, now_iso()),
        )
        history_rows = db.execute(
            "SELECT sender_name, text, is_bot FROM messages "
            "WHERE meeting_id = ? ORDER BY id DESC LIMIT 12",
            (payload.meeting_id,),
        ).fetchall()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise, friendly meeting assistant embedded inside a video call app. "
                "Help the user with quick answers, summaries, or meeting notes. Keep replies short."
            ),
        }
    ]
    for row in reversed(history_rows):
        role = "assistant" if row["is_bot"] else "user"
        messages.append({"role": role, "content": row["text"]})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.5, "max_tokens": 500},
            )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logger.error("Groq API error: %s", e.response.text)
        raise HTTPException(status_code=502, detail="Chatbot upstream error") from e
    except Exception as e:
        logger.exception("Chatbot failure")
        raise HTTPException(status_code=502, detail="Chatbot failed to respond") from e

    with get_db() as db:
        db.execute(
            "INSERT INTO messages (meeting_id, sender_name, sender_email, text, is_bot, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (payload.meeting_id, "Meet Assistant", None, reply, now_iso()),
        )

    return {"reply": reply}


# --------------------------------------------------------------------------
# WebSocket signaling + live chat + presence
# --------------------------------------------------------------------------

class RoomManager:
    """Tracks active WebSocket connections per meeting room (in-memory)."""

    def __init__(self):
        self.rooms: dict[str, dict[str, WebSocket]] = {}

    async def join(self, meeting_id: str, peer_id: str, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(meeting_id, {})[peer_id] = ws

    def leave(self, meeting_id: str, peer_id: str):
        peers = self.rooms.get(meeting_id)
        if peers and peer_id in peers:
            del peers[peer_id]
        if peers is not None and not peers:
            self.rooms.pop(meeting_id, None)

    def peers_in(self, meeting_id: str) -> dict[str, WebSocket]:
        return self.rooms.get(meeting_id, {})

    async def send_to(self, meeting_id: str, peer_id: str, message: dict):
        ws = self.rooms.get(meeting_id, {}).get(peer_id)
        if ws:
            await ws.send_json(message)

    async def broadcast(self, meeting_id: str, message: dict, exclude: Optional[str] = None):
        for pid, ws in list(self.peers_in(meeting_id).items()):
            if pid == exclude:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                pass


manager = RoomManager()


@app.websocket("/ws/{meeting_id}")
async def signaling_ws(websocket: WebSocket, meeting_id: str, token: str = "", name: str = "Guest"):
    user = get_user_by_token(token) if token else None
    display_name = user["name"] if user else (name or "Guest")
    email = user["email"] if user else None
    peer_id = uuid.uuid4().hex[:10]

    with get_db() as db:
        meeting = db.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if not meeting:
        await websocket.close(code=4404)
        return

    await manager.join(meeting_id, peer_id, websocket)

    with get_db() as db:
        db.execute(
            "INSERT INTO participants (meeting_id, user_id, name, email, joined_at) VALUES (?, ?, ?, ?, ?)",
            (meeting_id, user["id"] if user else None, display_name, email, now_iso()),
        )

    # Tell the newcomer who is already in the room
    existing_peers = [pid for pid in manager.peers_in(meeting_id) if pid != peer_id]
    await manager.send_to(
        meeting_id,
        peer_id,
        {"type": "welcome", "peer_id": peer_id, "existing_peers": existing_peers, "name": display_name},
    )

    # Notify everyone else that a new participant joined
    await manager.broadcast(
        meeting_id,
        {"type": "peer-joined", "peer_id": peer_id, "name": display_name},
        exclude=peer_id,
    )

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type in ("offer", "answer", "ice-candidate"):
                target = data.get("target")
                if target:
                    await manager.send_to(
                        meeting_id,
                        target,
                        {**data, "from": peer_id, "name": display_name},
                    )

            elif msg_type == "chat-message":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                with get_db() as db:
                    db.execute(
                        "INSERT INTO messages (meeting_id, sender_name, sender_email, text, is_bot, created_at) "
                        "VALUES (?, ?, ?, ?, 0, ?)",
                        (meeting_id, display_name, email, text, now_iso()),
                    )
                await manager.broadcast(
                    meeting_id,
                    {
                        "type": "chat-message",
                        "from": peer_id,
                        "name": display_name,
                        "text": text,
                        "created_at": now_iso(),
                    },
                )

            elif msg_type == "media-state":
                # e.g. mic muted, camera off, screen-sharing started/stopped
                await manager.broadcast(
                    meeting_id,
                    {"type": "media-state", "from": peer_id, "name": display_name, "state": data.get("state")},
                    exclude=peer_id,
                )

    except WebSocketDisconnect:
        pass
    finally:
        manager.leave(meeting_id, peer_id)
        await manager.broadcast(meeting_id, {"type": "peer-left", "peer_id": peer_id, "name": display_name})


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "zindex.html")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/room/{meeting_id}", response_class=HTMLResponse)
def serve_room(meeting_id: str):
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
