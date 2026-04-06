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
            model_name="Facenet512",
            enforce_detection=True
        )

        # ✅ Konversi numpy types ke Python native
        return {
            "verified": bool(result["verified"]),        # numpy.bool_ → bool
            "distance": float(result["distance"]),       # numpy.float_ → float
            "confidence": float(1 - result["distance"]), # numpy.float_ → float
            "threshold": float(result["threshold"])      # numpy.float_ → float
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir)