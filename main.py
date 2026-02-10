from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

app = FastAPI(title="AI Attendance Backend")

# In-memory DB (test). Production sẽ dùng Postgres.
EMP_DB: Dict[str, List[float]] = {}

class EnrollReq(BaseModel):
    employee_id: str
    embedding: List[float]

class AttendanceReq(BaseModel):
    timestamp: str
    location: str
    employee_id: str
    score: float
    camera_ip: str

@app.get("/health")
def health():
    return {"ok": True, "employees": len(EMP_DB)}

@app.post("/enroll")
def enroll(req: EnrollReq):
    emb = np.array(req.embedding, dtype=np.float32)
    if emb.ndim != 1 or emb.shape[0] < 128:
        raise HTTPException(400, "Embedding không hợp lệ (kỳ vọng vector 512).")
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise HTTPException(400, "Embedding norm=0.")
    emb = (emb / norm).tolist()
    EMP_DB[req.employee_id] = emb
    return {"ok": True, "employee_id": req.employee_id}

@app.get("/db")
def get_db():
    return {"db": EMP_DB}

@app.post("/attendance")
def attendance(req: AttendanceReq):
    # Test: nhận log, bước sau sẽ ghi Google Sheet
    return {"ok": True, "received": req.model_dump()}
