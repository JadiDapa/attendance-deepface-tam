"""
Face Recognition Service (stateless)
-------------------------------------
Menggunakan InsightFace (ArcFace, buffalo_l pack) via ONNX Runtime untuk
membuat embedding wajah.

CATATAN BACKEND: implementasi awal servis ini memakai DeepFace/TensorFlow,
tapi TensorFlow versi prebuilt (PyPI) mensyaratkan CPU dengan AVX. VPS ini
punya CPU virtual QEMU tanpa AVX sama sekali, jadi TensorFlow crash
(SIGILL) begitu di-import. Servis ini dipindah ke onnxruntime + insightface
karena ONNX Runtime tidak mensyaratkan AVX. Konsekuensinya:
- Model embedding beda (insightface w600k_r50, bukan deepface ArcFace) -
  embedding lama (kalau ada) TIDAK kompatibel dan harus di-enroll ulang.
- Anti-spoofing (liveness) SEMENTARA NONAKTIF - Fasnet/MiniFASNet asli
  butuh PyTorch, yang juga berisiko kena masalah AVX yang sama, dan belum
  ada pengganti ONNX yang tervalidasi. Lihat ANTI_SPOOFING_ENABLED di bawah.
- Kalau nanti deploy ke host dengan AVX, pertimbangkan balik ke
  deepface/TensorFlow untuk anti-spoofing bawaan + akurasi yang sedikit
  lebih matang.

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
- insightface akan otomatis download model pack "buffalo_l" saat pertama
  kali dipakai (butuh koneksi internet, ~280MB, sekali saja lalu dicache
  di ~/.insightface/models/).
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from pydantic import BaseModel

# --- Load .env kalau python-dotenv tersedia (opsional, tidak wajib di prod
# kalau env var sudah di-set lewat sistem/orchestrator) ---
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("face-api")
logging.basicConfig(level=logging.INFO)

MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)

# Deteksi + embedding dalam satu model pack (SCRFD untuk deteksi wajah,
# w600k_r50/ArcFace untuk embedding 512-d).
face_app: Optional[FaceAnalysis] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Muat model insightface di awal, bukan saat request pertama datang,
    supaya request absensi pertama tidak menanggung biaya load model."""
    global face_app
    face_app = FaceAnalysis(name=MODEL_PACK, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
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
# NOTE: threshold ini dikalibrasi untuk deepface/ArcFace, belum tentu pas
# untuk embedding insightface (model beda). Perlu diuji ulang dengan data
# nyata sebelum dipakai serius.

ANTI_SPOOFING_ENABLED = os.environ.get("FACE_ANTI_SPOOFING", "true").lower() == "true"
if ANTI_SPOOFING_ENABLED:
    logger.warning(
        "FACE_ANTI_SPOOFING=true di .env, tapi backend ONNX/insightface di "
        "servis ini BELUM punya liveness check. Anti-spoofing dianggap OFF "
        "untuk sesi ini — jangan andalkan servis ini untuk menolak foto "
        "hasil layar/cetak sampai ini diimplementasikan."
    )

# insightface/onnxruntime session belum tentu aman dipanggil dari banyak
# thread sekaligus untuk satu instance FaceAnalysis; lock ini menjaga
# supaya pekerjaan model tetap satu per satu. Endpoint di bawah sengaja
# `def` (bukan `async def`) supaya FastAPI menjalankannya di threadpool
# dan event loop tidak ikut beku menunggu lock ini.
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
    """Decode foto langsung dari memori (servis ini stateless — tidak ada
    foto yang ditulis ke disk atau disimpan permanen), lalu ekstrak
    embedding wajah paling yakin (det_score tertinggi) kalau ada lebih
    dari satu wajah terdeteksi."""
    data = photo.file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    with _model_lock:
        faces = face_app.get(img)

    if not faces:
        raise HTTPException(
            status_code=400,
            detail="Wajah tidak terdeteksi pada foto. Coba foto lain dengan pencahayaan lebih baik.",
        )

    best_face = max(faces, key=lambda f: f.det_score)
    return best_face.embedding.tolist()


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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PACK, "anti_spoofing": False}


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
        embedding=EmbeddingPayload(vector=vector, model=MODEL_PACK),
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
