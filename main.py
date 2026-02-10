import os
import glob
import time
from typing import Dict, List, Optional

import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(APP_DIR, "faces")

# DB: emp_id -> list of encodings
DB: Dict[str, List[np.ndarray]] = {}

app = FastAPI(title="AI Attendance Demo (Upload Image)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "message": "AI Attendance API is running", "faces_dir": FACES_DIR}

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def _load_faces() -> Dict[str, List[np.ndarray]]:
    if not os.path.exists(FACES_DIR):
        raise HTTPException(status_code=400, detail=f"Không thấy thư mục faces/ tại {FACES_DIR}")

    db: Dict[str, List[np.ndarray]] = {}
    # faces/nv001/1.jpg ...
    emp_dirs = [p for p in glob.glob(os.path.join(FACES_DIR, "*")) if os.path.isdir(p)]

    for emp_path in emp_dirs:
        emp_id = os.path.basename(emp_path)
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            imgs.extend(glob.glob(os.path.join(emp_path, ext)))

        encs: List[np.ndarray] = []
        for img_path in imgs:
            try:
                img = face_recognition.load_image_file(img_path)
                # lấy encoding face đầu tiên trong ảnh
                faces = face_recognition.face_encodings(img)
                if len(faces) == 0:
                    continue
                encs.append(faces[0])
            except Exception:
                continue

        if encs:
            db[emp_id] = encs

    return db

@app.post("/reload")
def reload_db():
    global DB
    DB = _load_faces()
    return {
        "ok": True,
        "employees": len(DB),
        "samples": {k: len(v) for k, v in DB.items()}
    }

@app.post("/recognize_image")
async def recognize_image(
    file: UploadFile = File(...),
    threshold: float = 0.42,  # cosine similarity threshold (tùy chỉnh)
):
    if not DB:
        raise HTTPException(status_code=400, detail="DB rỗng. Hãy gọi POST /reload trước.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")

    try:
        # đọc ảnh từ bytes
        img = face_recognition.load_image_file(np.frombuffer(content, dtype=np.uint8))
    except Exception:
        # fallback: load_image_file cần path, nên cách chắc ăn:
        # -> decode bằng face_recognition không ổn định theo bản, dùng numpy+cv2 thì cần opencv
        raise HTTPException(status_code=400, detail="Không đọc được ảnh. Hãy upload jpg/png hợp lệ.")

    # tìm face locations + encodings
    encs = face_recognition.face_encodings(img)
    if len(encs) == 0:
        return {"ok": False, "message": "Không phát hiện khuôn mặt trong ảnh."}

    query = encs[0]

    best_emp: Optional[str] = None
    best_sim: float = -1.0

    for emp_id, samples in DB.items():
        for s in samples:
            sim = _cosine(query, s)
            if sim > best_sim:
                best_sim = sim
                best_emp = emp_id

    matched = best_emp is not None and best_sim >= threshold
    return {
        "ok": True,
        "matched": matched,
        "emp_id": best_emp if matched else None,
        "best_candidate": best_emp,
        "similarity": round(best_sim, 4),
        "threshold": threshold,
        "ts": int(time.time())
    }
