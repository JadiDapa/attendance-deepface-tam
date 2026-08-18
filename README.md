# Face Recognition Service

Backend stateless (FastAPI + DeepFace/ArcFace) yang dipanggil server Next.js
lewat `lib/face-recognition.ts`. Tidak menyimpan data apa pun — Next.js/Prisma
yang jadi satu-satunya sumber data (embedding, karyawan, dll).

**PENTING:** service ini tidak boleh exposed ke internet. Jalankan di
jaringan internal (localhost saat dev, private network/VPC saat produksi) —
hanya server Next.js yang boleh bisa menjangkaunya.

## Setup

```bash
cd face-api-backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# isi FACE_API_SECRET dengan string acak, SAMAKAN dengan
# FACE_API_SECRET di .env Next.js kamu
```

## Jalankan

`.env` otomatis dibaca oleh `python-dotenv` — tidak perlu export manual di
shell manapun (Windows/Mac/Linux sama saja):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Kalau server gagal start dengan pesan `FACE_API_SECRET belum di-set`, cek:

- Apakah file `.env` (bukan cuma `.env.example`) benar-benar ada di folder ini
- Apakah `FACE_API_SECRET` di dalamnya sudah diisi (bukan string kosong `""`)

Model ArcFace akan otomatis di-download DeepFace saat request pertama kali
masuk (butuh koneksi internet sekali di awal, lalu dicache lokal).

## Test Manual (curl)

```bash
# Health check (tanpa auth)
curl http://localhost:8000/health

# Ekstrak embedding (samakan FACE_API_SECRET dengan punya kamu)
curl -X POST http://localhost:8000/embed \
  -H "X-Internal-Secret: isi-sesuai-env" \
  -F "photo=@/path/ke/foto.jpg"

# Verifikasi — reference_vectors contoh dari respons /embed di atas,
# array of arrays (walau cuma 1 referensi, tetap dibungkus array)
curl -X POST http://localhost:8000/verify \
  -H "X-Internal-Secret: isi-sesuai-env" \
  -F "photo=@/path/ke/foto-lain.jpg" \
  -F 'reference_vectors=[[0.01, -0.02, ...]]'
```

## Menyamakan dengan Next.js

Pastikan nilai berikut **identik** di `.env` Next.js dan `.env` backend ini:

| Next.js (`.env`)                                 | Backend (`.env`)                                  |
| ------------------------------------------------ | ------------------------------------------------- |
| `FACE_API_SECRET`                                | `FACE_API_SECRET`                                 |
| `FACE_MATCH_THRESHOLD` (opsional, informational) | `FACE_MATCH_THRESHOLD` (yang benar-benar dipakai) |

`FACE_API_URL` di Next.js diarahkan ke alamat service ini, mis.
`http://localhost:8000` saat dev.

## Produksi

- Deploy di server/container terpisah, di internal network yang sama dengan
  server Next.js (private network/VPC, Docker internal network, dsb) —
  jangan expose port 8000 ke publik.
- Pertimbangkan menjalankan lewat Gunicorn + Uvicorn workers untuk
  concurrency lebih baik: `gunicorn main:app -k uvicorn.workers.UvicornWorker -w 2`
- Kalau nanti trafiknya tinggi, model ArcFace di CPU bisa jadi bottleneck —
  pertimbangkan GPU inference atau model yang lebih ringan (mis. MobileFaceNet).
