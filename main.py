from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import tempfile, os, shutil

app = FastAPI()

@app.post("/verify")
async def verify_face(
    foto_absen: UploadFile = File(...),
    foto_referensi: UploadFile = File(...)
):
    tmp_dir = tempfile.mkdtemp()
    try:
        path_absen = os.path.join(tmp_dir, "absen.jpg")
        path_referensi = os.path.join(tmp_dir, "referensi.jpg")

        with open(path_absen, "wb") as f:
            f.write(await foto_absen.read())
        with open(path_referensi, "wb") as f:
            f.write(await foto_referensi.read())

        result = DeepFace.verify(
            img1_path=path_absen,
            img2_path=path_referensi,
            model_name="Facenet512",   # paling akurat
            enforce_detection=True
        )

        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "confidence": 1 - result["distance"],  # 0-1, makin tinggi makin mirip
            "threshold": result["threshold"]
        }

    except ValueError as e:
        # Wajah tidak terdeteksi di salah satu foto
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir)