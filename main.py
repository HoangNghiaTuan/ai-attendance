from fastapi import FastAPI, UploadFile, File
import face_recognition
import numpy as np
import cv2

app = FastAPI()

# Lưu face encoding tạm (test)
KNOWN_FACE = None
KNOWN_NAME = None

@app.get("/")
def root():
    return {"status": "AI Attendance Backend Running"}

@app.post("/register")
async def register_face(name: str, file: UploadFile = File(...)):
    global KNOWN_FACE, KNOWN_NAME

    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    encodings = face_recognition.face_encodings(img)
    if not encodings:
        return {"error": "No face detected"}

    KNOWN_FACE = encodings[0]
    KNOWN_NAME = name

    return {"message": f"Registered {name}"}

@app.post("/check")
async def check_face(file: UploadFile = File(...)):
    if KNOWN_FACE is None:
        return {"error": "No registered face"}

    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    encodings = face_recognition.face_encodings(img)
    if not encodings:
        return {"result": "No face"}

    match = face_recognition.compare_faces([KNOWN_FACE], encodings[0])[0]

    return {
        "match": match,
        "name": KNOWN_NAME if match else "Unknown"
    }
