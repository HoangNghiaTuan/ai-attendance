import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# InsightFace (nhẹ hơn dlib + face_recognition khi deploy)
from insightface.app import FaceAnalysis

FACES_DIR = Path("faces")  # faces/nv001/1.jpg ...
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.38"))  # càng thấp càng dễ nhận nhầm; test 0.35~0.45
APP_NAME = "ai-attendance"

app = FastAPI(title=APP_NAME)

# Init model
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# In-memory DB: {emp_id: [embedding...]}
DB: Dict[str, List[np.ndarray]] = {}


class RecognizeResult(BaseModel):
    ok: bool
    employee_id: Optional[str] = None
    similarity: Optional[float] = None
    message: str


def _imread_bytes(data: bytes) -> np.ndarray:
    img = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không đọc được ảnh (file hỏng hoặc không phải ảnh).")
    return bgr


def _get_embedding(bgr: np.ndarray) -> np.ndarray:
    faces = face_app.get(bgr)
    if not faces:
        raise ValueError("Không phát hiện khuôn mặt trong ảnh.")
    # Lấy face lớn nhất
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    emb = faces[0].embedding.astype(np.float32)
    # normalize
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def load_faces_from_repo():
    DB.clear()
    if not FACES_DIR.exists():
        print("faces/ not found")
        return

    for emp_dir in FACES_DIR.iterdir():
        if not emp_dir.is_dir():
            continue
        emp_id = emp_dir.name
        embs: List[np.ndarray] = []
        for img_path in emp_dir.glob("*.jpg"):
            try:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                emb = _get_embedding(bgr)
                embs.append(emb)
            except Exception:
                continue
        if embs:
            DB[emp_id] = embs

    print(f"Loaded {sum(len(v) for v in DB.values())} face samples for {len(DB)} employees")


@app.on_event("startup")
def _startup():
    load_faces_from_repo()


@app.get("/health")
def health():
    return {"ok": True, "employees": len(DB), "samples": sum(len(v) for v in DB.values())}


@app.post("/reload")
def reload_db():
    load_faces_from_repo()
    return {"ok": True, "employees": len(DB), "samples": sum(len(v) for v in DB.values())}


@app.post("/recognize", response_model=RecognizeResult)
async def recognize(file: UploadFile = File(...)):
    if not DB:
        raise HTTPException(status_code=400, detail="DB rỗng. Hãy upload ảnh vào faces/ hoặc gọi /reload")

    data = await file.read()
    try:
        bgr = _imread_bytes(data)
        emb = _get_embedding(bgr)
    except Exception as e:
        return RecognizeResult(ok=False, message=str(e))

    best_emp = None
    best_sim = -1.0

    for emp_id, samples in DB.items():
        for s in samples:
            sim = _cosine(emb, s)
            if sim > best_sim:
                best_sim = sim
                best_emp = emp_id

    if best_emp is None or best_sim < SIM_THRESHOLD:
        return RecognizeResult(ok=True, employee_id=None, similarity=best_sim, message="unknown")

    return RecognizeResult(ok=True, employee_id=best_emp, similarity=best_sim, message="matched")
