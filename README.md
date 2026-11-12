# Face Recognition Service

A stateless FastAPI backend that provides face embedding extraction and 1:1 face verification for the **attendance-tam** attendance system. It is called server-to-server by the Next.js dashboard (via `lib/face-recognition.ts`) and never stores any data itself — the Next.js app's Prisma database is the single source of truth for employee face embeddings.

> **Important:** this service must never be exposed to the public internet. Run it on an internal/private network (`localhost` in development, a private network or VPC in production) that only the Next.js server can reach. The `X-Internal-Secret` header is a second layer of defense, not a substitute for network isolation.

## Model backend note

This service originally used DeepFace/TensorFlow. The prebuilt TensorFlow wheel on PyPI requires a CPU with AVX, and the production VPS runs on a QEMU virtual CPU with no AVX support at all, which made TensorFlow crash with `SIGILL` on import. The service was therefore moved to **ONNX Runtime + InsightFace** (`buffalo_l` model pack, 512‑dimension ArcFace/`w600k_r50` embeddings), since ONNX Runtime has no AVX requirement.

Consequences of this change:

- Embeddings are **not compatible** with the original DeepFace/ArcFace embeddings — any previously enrolled face data must be re-enrolled.
- **Anti-spoofing / liveness detection is currently disabled.** The original Fasnet/MiniFASNet liveness check requires PyTorch, which risks the same AVX crash, and no validated ONNX replacement is in place yet. Do not rely on this service to reject photos of a screen or printed photo until this is implemented.
- If this service is ever deployed to a host with AVX support, switching back to the DeepFace/TensorFlow stack is a reasonable option for built-in anti-spoofing and slightly higher accuracy (see git history for the pre-migration version of `main.py`).

## Features

- **`/embed`** — extract a 512-d face embedding from a single uploaded photo (used during employee enrollment).
- **`/verify`** — 1:1 verification: compare one attendance photo against the reference embedding(s) of a single logged-in employee, returning a match/no-match decision and the cosine distance.
- **`/health`** — unauthenticated health check.
- Fully stateless: photos are decoded in memory and discarded; nothing is written to disk.
- Shared-secret authentication (`X-Internal-Secret` header, compared with constant-time `hmac.compare_digest`) on every data-bearing endpoint.
- Model is loaded once at startup (via FastAPI's `lifespan`), not on the first request, so the first real attendance check isn't penalized with model-load latency.
- Thread-safe inference: a global lock serializes calls into the InsightFace/ONNX Runtime session, while endpoints are defined as sync `def` functions so FastAPI runs them in a thread pool instead of blocking the event loop.

## Tech stack

| Component | Library / Version |
| --- | --- |
| Web framework | FastAPI `0.115.0` |
| ASGI server | Uvicorn `0.30.6` |
| Face detection + embedding | InsightFace `1.0.1` (`buffalo_l` pack: SCRFD detector + ArcFace `w600k_r50` embedding) |
| Inference runtime | ONNX Runtime `1.23.2` (CPU execution provider) |
| Image decoding | OpenCV (`opencv-python-headless` `5.0.0.93`) |
| Numerics | NumPy `2.2.6` |
| Validation | Pydantic `2.9.2` |
| Config | `python-dotenv` `1.0.1` |

## Prerequisites

- Python 3.10+ (any version compatible with the pinned dependencies above)
- Internet access on first run, so InsightFace can download the `buffalo_l` model pack (~280 MB, cached afterwards in `~/.insightface/models/`)
- The same `FACE_API_SECRET` value configured in this service's `.env` and in the Next.js dashboard's `.env`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in FACE_API_SECRET with a random string — it must match
# FACE_API_SECRET in the Next.js dashboard's .env exactly
```

## Running

`.env` is loaded automatically via `python-dotenv` — no manual `export` needed on any platform.

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If the server fails to start with `FACE_API_SECRET belum di-set` (not set), check:

- That an actual `.env` file (not just `.env.example`) exists in this directory
- That `FACE_API_SECRET` inside it is non-empty

The `buffalo_l` model pack is downloaded automatically by InsightFace the first time it's used.

## Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `FACE_API_SECRET` | Yes | — | Shared secret required in the `X-Internal-Secret` header on every request to `/embed` and `/verify`. Must be identical to the value configured in the Next.js dashboard. The service refuses to start if this is unset. |
| `FACE_MATCH_THRESHOLD` | No | `0.68` | Maximum cosine distance for two embeddings to be considered the same face in `/verify`. This value was calibrated for the original DeepFace/ArcFace embeddings and has **not** been re-validated for InsightFace/`buffalo_l` — re-test with real data before relying on it in production. |
| `FACE_ANTI_SPOOFING` | No | `true` | Intended to toggle liveness detection on `/verify`. **Currently has no effect** — the ONNX/InsightFace backend has no liveness implementation yet, and the service logs a warning at startup if this is `true`. Do not treat this as an active security control. |

## API reference

All endpoints except `/health` require the header:

```
X-Internal-Secret: <FACE_API_SECRET>
```

Requests without this header, or with an incorrect value, receive `401 Unauthorized`.

### `GET /health`

Unauthenticated health check.

**Response `200`**

```json
{
  "status": "ok",
  "model": "buffalo_l",
  "anti_spoofing": false
}
```

### `POST /embed`

Extract a face embedding from a single photo. Used during employee enrollment — the caller (Next.js) is responsible for persisting the returned vector to the `FaceEmbedding` table.

**Request** — `multipart/form-data`

| Field | Type | Description |
| --- | --- | --- |
| `photo` | file | Image containing exactly one clearly visible face. If multiple faces are detected, the one with the highest detection confidence is used. |

```bash
curl -X POST http://localhost:8000/embed \
  -H "X-Internal-Secret: <your-secret>" \
  -F "photo=@/path/to/photo.jpg"
```

**Response `200`**

```json
{
  "status": "ok",
  "embedding": {
    "vector": [0.0123, -0.0456, "... 512 floats total"],
    "model": "buffalo_l"
  }
}
```

**Error responses**

| Status | Condition |
| --- | --- |
| `400` | Uploaded file is not a decodable image, or no face was detected in the photo. |
| `401` | Missing or incorrect `X-Internal-Secret` header. |

### `POST /verify`

1:1 verification — compares one attendance photo against the reference embedding(s) belonging to a single, already-identified employee (not a 1:N search across all employees).

**Request** — `multipart/form-data`

| Field | Type | Description |
| --- | --- | --- |
| `photo` | file | The attendance-check photo to verify. |
| `reference_vectors` | string (JSON) | A JSON-encoded array of embedding vectors (`number[][]`) for the employee being verified, fetched from Prisma by the Next.js caller. Even a single reference embedding must be wrapped in an array. |

```bash
curl -X POST http://localhost:8000/verify \
  -H "X-Internal-Secret: <your-secret>" \
  -F "photo=@/path/to/checkin-photo.jpg" \
  -F 'reference_vectors=[[0.0123, -0.0456, "..."]]'
```

The photo's embedding is compared against every vector in `reference_vectors` using cosine distance; the smallest distance is used to decide the result against `FACE_MATCH_THRESHOLD`.

**Response `200`**

```json
{
  "status": "match",
  "distance": 0.3421
}
```

`status` is either `"match"` or `"no_match"`. `distance` is the cosine distance (lower = more similar) between the submitted photo and the closest reference embedding, rounded to 4 decimal places.

**Error responses**

| Status | Condition |
| --- | --- |
| `400` | `reference_vectors` is missing, not valid JSON, not a non-empty array, or the photo has no detectable face / isn't a valid image. |
| `401` | Missing or incorrect `X-Internal-Secret` header. |

## Project structure

```
ai-model/
├── main.py             # FastAPI app: model loading, auth, /health, /embed, /verify
├── requirements.txt    # Pinned Python dependencies
├── .env.example         # Template for required environment variables
├── uploads/            # Present but unused — service is stateless, no files are persisted here
└── README.md
```

## Integration with the dashboard

The Next.js dashboard calls this service from `lib/face-recognition.ts`, pointing at it via a `FACE_API_URL` environment variable on the Next.js side (e.g. `http://localhost:8000` in development). Keep `FACE_API_SECRET` identical between this service's `.env` and the dashboard's `.env` — a mismatch will cause every request to fail with `401 Unauthorized`.

## Production deployment

- Deploy on a separate host/container on the same internal network as the Next.js server (private network, VPC, Docker internal network, etc.). Do **not** expose the port publicly — bind to `127.0.0.1`, not `0.0.0.0`.
- Consider running behind Gunicorn with Uvicorn workers for better concurrency:
  ```bash
  gunicorn main:app -k uvicorn.workers.UvicornWorker -w 2
  ```
- CPU inference may become a bottleneck under high traffic — consider GPU inference or a lighter model pack if this becomes an issue.
- Whichever port you bind, it must match the `FACE_API_URL` configured in the dashboard's `.env`.
