# Signal — Meet

A self-contained video meeting app: FastAPI + SQLite backend (`main.py`) and a
single-page vanilla JS frontend (`index.html`).

## Features
- Email-only login (no password) → bearer token, first name shown as the greeting.
- Dashboard: create a meeting link in one click, copy it, or join an existing one by code/link.
- Joining via a shared link asks for email first (also doubles as login).
- WebRTC video calls (mesh) with a WebSocket signaling channel: offer/answer/ICE relay,
  "X joined / left the meeting" notifications, live in-call chat.
- Camera/mic permission prompt, mute/camera toggle, screen sharing (swaps the outgoing
  video track live via `replaceTrack`, no renegotiation needed).
- STUN by default (Google's public STUN servers); set `TURN_URL` / `TURN_USERNAME` /
  `TURN_CREDENTIAL` env vars to add a TURN server for users behind strict NATs — required
  for reliable production use across networks.
- Floating AI chatbot popup (bottom-right) that only appears for one designated account
  (`CHATBOT_ALLOWED_EMAIL`, defaults to `abc@gmail.com`), backed by Groq's
  `llama-3.3-70b-versatile` model.
- Every meeting auto-records mixed audio (local mic + every remote participant, combined
  via the Web Audio API) and uploads it as a BLOB to SQLite when you leave.
- All meetings, participants, chat/bot messages, and recordings persisted in SQLite (`meet.db`).
- Custom request-logging middleware + CORS middleware.
- Self keep-alive task + `/health` endpoint to help avoid cold-start gaps on free hosting tiers.

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_key          # get one at console.groq.com
export CHATBOT_ALLOWED_EMAIL=abc@gmail.com  # optional, this is the default
export TURN_URL=turn:your.turn.server:3478  # optional, for production NAT traversal
export TURN_USERNAME=xxxx                   # optional
export TURN_CREDENTIAL=xxxx                 # optional
python3 main.py
```

Open `http://localhost:8000`, log in with any email, click **New meeting**, copy the
link, and open it in a second browser tab/window (or send it to someone else) to test
the call. Log in as `abc@gmail.com` in one of the tabs to see the AI chatbot icon.

## Recordings
Audio (not video) is recorded client-side: local mic + every remote participant's audio
are mixed with the Web Audio API into one track, captured with `MediaRecorder`, and
uploaded to `/api/recordings/upload` when the user clicks **Leave**. Fetch a meeting's
recordings with `GET /api/meetings/{id}/recordings` and download one with
`GET /api/recordings/{id}`. Known limitation: closing the tab / losing connection
*without* clicking Leave skips the upload — there's no `beforeunload` fallback yet
because reliably uploading an authenticated multipart blob on tab-close isn't
guaranteed by browsers (`sendBeacon` doesn't support custom headers). If you need that
covered, chunk-upload periodically during the call instead of only at the end.

## Cross-browser connectivity (Safari / Firefox)
If calls connect fine in Chrome but fail in Safari or Firefox, it's almost always a
missing TURN server, not a code bug. This app ships with Google's public STUN servers
only, which is enough on the same network or a permissive NAT — Safari and Firefox are
stricter about symmetric-NAT/relay requirements and will fail ICE silently without a
TURN relay. Get a TURN server (e.g. [Twilio TURN](https://www.twilio.com/docs/stun-turn),
[Cloudflare Calls](https://developers.cloudflare.com/calls/turn/), or self-hosted
[coturn](https://github.com/coturn/coturn)) and set `TURN_URL` / `TURN_USERNAME` /
`TURN_CREDENTIAL`. The frontend now logs each peer's `iceConnectionState` to the console
and shows a toast on `failed`, so you can confirm this is the cause.

## Keep-alive on Render (or any free/idle-spindown host)
Render's free tier spins a service down after ~15 minutes with no inbound HTTP traffic,
so the first request after idle time pays a slow cold-start. Two settings work together:
- `SELF_PING_URL` (or Render's own `RENDER_EXTERNAL_URL`, set automatically) makes the
  app ping its own `/health` every `KEEP_ALIVE_INTERVAL_SECONDS` (default 600s/10min) —
  this keeps an **already-running** instance from going idle.
- It **cannot** wake an instance that already spun down before anyone hit it — for an
  interview, also point a free external monitor (UptimeRobot, cron-job.org, Better
  Uptime) at your public `/health` URL on a 5–10 minute interval, so there's always
  outside traffic keeping it warm even between your own visits.

## Notes for taking this further
- The signaling server keeps peer connections in memory (`RoomManager`) — fine for a
  single server process; put it behind Redis pub/sub if you scale to multiple instances.
- Video is peer-to-peer mesh, which is simple but doesn't scale much past ~4-6
  participants per call; a real production build with many participants per room would
  route media through an SFU (e.g. LiveKit, mediasoup) instead.
- Add HTTPS/WSS (e.g. behind Caddy/Nginx, or Render's own HTTPS) before using this
  outside localhost — browsers require a secure context for camera/mic/screen-share
  access on any non-localhost, non-HTTPS origin. This is the other common cause of
  "works on my machine, fails elsewhere."
- Swap the plain bearer token for signed JWTs with expiry if this goes further than a demo.
- SQLite BLOBs are fine for a demo/interview-scale app; for real production recording
  volume, store audio in object storage (S3/R2) and keep only the URL in SQLite.
