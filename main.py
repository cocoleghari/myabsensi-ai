from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
from PIL import Image, ImageOps
import tempfile, os, shutil, io
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

CONFIDENCE_THRESHOLD = 0.6  # ← ubah di sini jika perlu sesuaikan batas minimum

def preprocess_image(raw_bytes: bytes, save_path: str):
    """Normalisasi orientasi dan simpan sebagai JPEG standar."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)  # fix rotasi EXIF
    img.save(save_path, "JPEG", quality=95)

@app.post("/verify")
async def verify_face(
    foto_absen: UploadFile = File(...),
    foto_referensi: UploadFile = File(...)
):
    tmp_dir = tempfile.mkdtemp()
    try:
        path_absen = os.path.join(tmp_dir, "absen.jpg")
        path_referensi = os.path.join(tmp_dir, "referensi.jpg")

        preprocess_image(await foto_absen.read(), path_absen)
        preprocess_image(await foto_referensi.read(), path_referensi)

        result = DeepFace.verify(
            img1_path=path_absen,
            img2_path=path_referensi,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
        )

        distance = float(result["distance"])
        threshold = float(result["threshold"])

        # Confidence untuk display
        max_distance = 0.8
        confidence = max(0.0, min(1.0, 1 - (distance / max_distance)))

        # Verified berdasarkan confidence >= 60%
        verified = confidence >= CONFIDENCE_THRESHOLD

        logging.info(f"distance={distance:.4f}, threshold={threshold:.4f}, verified={verified}, confidence={confidence:.4f}")

        return {
            "verified": verified,
            "distance": distance,
            "confidence": round(confidence, 4),
            "threshold": threshold,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir)