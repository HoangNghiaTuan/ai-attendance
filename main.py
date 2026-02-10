import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from insightface.app import FaceAnalysis

FACES_DIR = Path("faces")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.38"))

app = FastAPI(title="ai-attendance-local")

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

DB: Dict[str, List[np.ndarray]] = {}

class RecognizeResult(BaseModel):
    ok: bool
    employee_id: Optional[str] = None
    similarity: Optional[float] = None
    message: str

def _get_embedding(bgr: np.ndarray) -> np.ndarray:
    faces = face_app.get(bgr)
    if not faces:
        raise ValueError("Không phát hiện khuôn mặt trong ảnh.")
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    emb = faces[0].embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def load_faces():
    DB.clear()
    if not FACES_DIR.exists():
        return
    for emp_dir in FACES_DIR.iterdir():
        if not emp_dir.is_dir():
            continue
        emp_id = emp_dir.name
        embs: List[np.ndarray] = []
        for img_path in emp_dir.glob("*.jpg"):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            try:
                embs.append(_get_embedding(bgr))
            except Exception:
                pass
        if embs:
            DB[emp_id] = embs

@app.on_event("startup")
def _startup():
    load_faces()

@app.get("/health")
def health():
    return {"ok": True, "employees": len(DB), "samples": sum(len(v) for v in DB.values())}

@app.post("/reload")
def reload_db():
    load_faces()
    return {"ok": True, "employees": len(DB), "samples": sum(len(v) for v in DB.values())}

@app.post("/recognize", response_model=RecognizeResult)
async def recognize(file: UploadFile = File(...)):
    if not DB:
        raise HTTPException(status_code=400, detail="DB rỗng. Upload ảnh vào faces/ rồi gọi /reload")
    data = await file.read()
    img = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if bgr is None:
        return RecognizeResult(ok=False, message="Không đọc được ảnh.")
    try:
        emb = _get_embedding(bgr)
    except Exception as e:
        return RecognizeResult(ok=False, message=str(e))

    best_emp, best_sim = None, -1.0
    for emp_id, samples in DB.items():
        for s in samples:
            sim = _cosine(emb, s)
            if sim > best_sim:
                best_sim = sim
                best_emp = emp_id

    if best_emp is None or best_sim < SIM_THRESHOLD:
        return RecognizeResult(ok=True, employee_id=None, similarity=best_sim, message="unknown")
    return RecognizeResult(ok=True, employee_id=best_emp, similarity=best_sim, message="matched")

@app.get("/snap-and-recognize", response_model=RecognizeResult)
def snap_and_recognize(
    rtsp: str = Query(..., description="RTSP URL"),
):
    if not DB:
        raise HTTPException(status_code=400, detail="DB rỗng. Upload ảnh vào faces/ rồi gọi /reload")

    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        return RecognizeResult(ok=False, message="Không mở được RTSP. Kiểm tra URL/user/pass/mạng.")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return RecognizeResult(ok=False, message="Không đọc được frame từ camera.")

    try:
        emb = _get_embedding(frame)
    except Exception as e:
        return RecognizeResult(ok=False, message=str(e))

    best_emp, best_sim = None, -1.0
    for emp_id, samples in DB.items():
        for s in samples:
            sim = _cosine(emb, s)
            if sim > best_sim:
                best_sim = sim
                best_emp = emp_id

    if best_emp is None or best_sim < SIM_THRESHOLD:
        return RecognizeResult(ok=True, employee_id=None, similarity=best_sim, message="unknown")
    return RecognizeResult(ok=True, employee_id=best_emp, similarity=best_sim, message="matched")
