from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid
import shutil

from crop_cccd import crop_cccd

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def home():
    return {
        "message": "API crop CCCD dang chay"
    }

@app.post("/crop")
def crop_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
    output_path = os.path.join(OUTPUT_DIR, file_id + "_crop.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ok = crop_cccd(input_path, output_path)

    if not ok:
        return {
            "status": "ERROR",
            "message": "Khong crop duoc CCCD. Anh kiem tra anh co ro 4 goc khong."
        }

    return FileResponse(
        output_path,
        media_type="image/jpeg",
        filename="cccd_crop.jpg"
    )