"""
Face Recognition Service (stateless)
-------------------------------------
Menggunakan DeepFace (model ArcFace) untuk membuat embedding wajah.

Arsitektur "Opsi A" (disepakati dengan tim Next.js):
- Servis ini TIDAK menyimpan data apa pun (tidak ada db.json, tidak ada
  folder uploads permanen). Prisma di sisi Next.js adalah satu-satunya
  sumber kebenaran untuk embedding wajah karyawan.
- /embed  : terima 1 foto -> kembalikan 1 embedding vector. Next.js yang
            menyimpannya ke tabel FaceEmbedding.
- /verify : terima 1 foto absensi + daftar reference_vectors (diambil dari
            Prisma oleh Next.js) -> kembalikan match/no_match + distance.
            Verifikasi selalu 1:1 terhadap SATU user yang sedang login,
            bukan 1:N ke seluruh karyawan.
- Setiap request wajib menyertakan header `X-Internal-Secret` yang sama
  persis dengan FACE_API_SECRET di .env servis ini. Servis ini idealnya
  tidak exposed ke internet (internal network only) — header ini cuma
  lapis kedua jaga-jaga.

Cara jalankan:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Catatan:
- DeepFace akan otomatis download model ArcFace saat pertama kali dipakai
  (butuh koneksi internet, bisa makan waktu beberapa menit di run pertama).
"""

import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace

# --- Load .env kalau python-dotenv tersedia (opsional, tidak wajib di prod
# kalau env var sudah di-set lewat sistem/orchestrator) ---
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Muat semua model di awal, bukan saat request pertama datang.

    DeepFace memuat model secara lazy, jadi tanpa ini absensi pertama setelah
    servis dinyalakan menanggung ~18-40 detik pemuatan model (detector, Fasnet,
    ArcFace) — request berikutnya baru ~1-2 detik. Diukur di mesin dev: dengan
    warmup ini request pertama turun ke ~2 detik.

    Kalau uvicorn dijalankan dengan --reload, tiap perubahan file me-restart
    worker dan biaya ini dibayar ulang — jangan pakai --reload saat mengukur
    waktu absensi atau di produksi.
    """
    DeepFace.build_model(model_name=DETECTOR_BACKEND, task="face_detector")
    DeepFace.build_model(model_name=MODEL_NAME, task="facial_recognition")
    if ANTI_SPOOFING_ENABLED:
        DeepFace.build_model(model_name="Fasnet", task="spoofing")
    yield


app = FastAPI(title="Face Recognition Service", lifespan=lifespan)

# Sesuaikan origin saat deploy. Ini cuma dipanggil server-to-server dari
# Next.js (server action), bukan langsung dari browser, tapi CORS dijaga
# untuk kondisi dev di mana Next.js server jalan lewat proxy/browser tools.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACE_API_SECRET = os.environ.get("FACE_API_SECRET")
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.68"))
DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR_BACKEND", "retinaface")
MODEL_NAME = "ArcFace"
# Liveness check (tolak foto dari layar/cetak). Bisa dimatikan sementara lewat
# .env kalau perlu debug, tapi disarankan tetap ON di /verify (absensi).
ANTI_SPOOFING_ENABLED = os.environ.get("FACE_ANTI_SPOOFING", "true").lower() == "true"

# DeepFace/TensorFlow memakai satu instance model yang di-cache global, dan
# predict-nya bukan CPU-bound yang aman dipanggil dari banyak thread sekaligus.
# Endpoint di bawah sengaja `def` (bukan `async def`) supaya FastAPI menjalankannya
# di threadpool dan event loop tidak ikut beku; lock ini yang menjaga supaya
# pekerjaan model tetap satu per satu.
_model_lock = threading.Lock()

if not FACE_API_SECRET:
    raise RuntimeError(
        "FACE_API_SECRET belum di-set. Isi di .env servis ini (harus sama "
        "persis dengan FACE_API_SECRET di .env Next.js)."
    )


def verify_secret(x_internal_secret: Optional[str] = Header(default=None)) -> None:
    """Dependency: tolak request kalau header X-Internal-Secret tidak cocok."""
    if x_internal_secret != FACE_API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_embedding_from_upload(photo: UploadFile) -> List[float]:
    """Simpan foto ke file sementara, ekstrak embedding, lalu selalu bersihkan
    file sementara tsb (servis ini stateless — tidak ada foto yang disimpan
    permanen)."""
    suffix = os.path.splitext(photo.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(photo.file.read())
        tmp_path = tmp.name

    try:
        with _model_lock:
            if ANTI_SPOOFING_ENABLED:
                # Liveness check dulu (Fasnet) SEBELUM ekstrak embedding — tolak
                # kalau ini foto dari foto/layar (printed photo / replay attack),
                # bukan wajah asli di depan kamera.
                faces = DeepFace.extract_faces(
                    img_path=tmp_path,
                    detector_backend=DETECTOR_BACKEND,
                    anti_spoofing=True,
                    enforce_detection=True,
                )
                if not faces or not faces[0].get("is_real", False):
                    raise HTTPException(
                        status_code=400,
                        detail="Wajah terdeteksi tidak asli (kemungkinan foto dari layar/cetak). Gunakan kamera langsung.",
                    )

            result = DeepFace.represent(
                img_path=tmp_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )
        # DeepFace.represent bisa mengembalikan beberapa wajah; ambil yang
        # pertama (foto enrollment/absensi diasumsikan 1 wajah per foto).
        return result[0]["embedding"]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Wajah tidak terdeteksi pada foto. Coba foto lain dengan pencahayaan lebih baik.",
        )
    finally:
        os.remove(tmp_path)


def cosine_distance(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 1.0
    return float(1 - (np.dot(a_arr, b_arr) / denom))


class EmbeddingPayload(BaseModel):
    vector: List[float]
    model: str


class EmbedResponse(BaseModel):
    status: str = "ok"
    embedding: EmbeddingPayload


class VerifyResponse(BaseModel):
    status: str
    distance: Optional[float] = None


@app.post("/embed", response_model=EmbedResponse)
def embed(
    photo: UploadFile = File(...),
    _: None = Depends(verify_secret),
):
    """Ekstrak embedding dari satu foto wajah. Dipakai saat enrollment —
    Next.js yang menyimpan hasilnya ke Prisma (FaceEmbedding), servis ini
    tidak menyimpan apa pun."""
    vector = get_embedding_from_upload(photo)

    return EmbedResponse(
        status="ok",
        embedding=EmbeddingPayload(vector=vector, model=MODEL_NAME),
    )


@app.post("/verify", response_model=VerifyResponse)
def verify(
    photo: UploadFile = File(...),
    reference_vectors: str = Form(...),
    _: None = Depends(verify_secret),
):
    """Verifikasi 1:1 — cocokkan foto absensi dengan embedding milik SATU
    karyawan (yang sedang login). `reference_vectors` dikirim Next.js sebagai
    JSON string berisi array of number[] (embedding-embedding karyawan itu,
    diambil dari Prisma)."""
    try:
        vectors: List[List[float]] = json.loads(reference_vectors)
        if not isinstance(vectors, list) or not vectors:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="reference_vectors tidak valid")

    embedding = get_embedding_from_upload(photo)

    best_distance = min(cosine_distance(embedding, ref) for ref in vectors)

    if best_distance <= MATCH_THRESHOLD:
        return VerifyResponse(status="match", distance=round(best_distance, 4))

    return VerifyResponse(status="no_match", distance=round(best_distance, 4))
