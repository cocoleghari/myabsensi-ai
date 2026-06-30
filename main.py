from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
from PIL import Image, ImageOps
import tempfile, os, shutil, io
import logging
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("face_recognition")

app = FastAPI()

# ══════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════

# ── Keputusan verified ──────────────────────────────────────────────
# PENTING: keputusan verified/tidak berbasis threshold ASLI dari DeepFace
# (result["threshold"], untuk Facenet512 = 0.30), bukan confidence buatan
# sendiri. Threshold ini sudah dikalibrasi DeepFace untuk model & metric
# yang dipakai (cosine distance Facenet512).
#
# PERINGATAN KERAS — JANGAN NAIKKAN MARGIN INI TANPA VALIDASI KEAMANAN:
# Pengujian lapangan menemukan distance untuk DUA ORANG BERBEDA bisa
# serendah 0.25-0.43 — overlap dengan distance kondisi sulit (gelap/
# ekspresi beda) milik orang yang SAMA. Ini adalah keterbatasan statistik
# bawaan model face-embedding 2D, BUKAN sesuatu yang bisa dihilangkan
# hanya dengan menaikkan margin. Margin besar = false-accept lebih sering.
# Default margin kecil ini DISENGAJA demi keamanan: lebih baik user disuruh
# foto ulang (false-reject) daripada orang lain berhasil absen menggantikan
# orang lain (false-accept).
VERIFIED_MARGIN = 0.05

# Confidence (0-100%) untuk DISPLAY ke user saja, tidak dipakai untuk
# keputusan verified. Diskalakan relatif ke threshold asli DeepFace.
CONFIDENCE_SCALE_FACTOR = 2.0

# ── Detector backend ──────────────────────────────────────────────
# retinaface jauh lebih akurat untuk selfie miring / low-light dibanding
# opencv, tapi lebih berat & butuh package tambahan (retina-face). Kalau
# gagal load/crash, otomatis fallback ke opencv supaya service tidak down.
PREFERRED_DETECTOR = "retinaface"
FALLBACK_DETECTOR = "opencv"

# ── Resize standar ──────────────────────────────────────────────
# Semua foto di-resize ke lebar maksimum ini SEBELUM diproses (mempertahankan
# aspect ratio). Tujuannya: konsistensi input ke detector & model (foto dari
# kamera HP modern bisa 3000-4000px, jauh lebih besar dari yang dibutuhkan
# model, dan memperlambat proses tanpa menambah akurasi).
STANDARD_MAX_WIDTH = 800

# ── CLAHE (kontras adaptif untuk kondisi gelap) ──────────────────────
# Hanya diterapkan kalau brightness rata-rata foto di bawah threshold ini,
# supaya foto yang sudah cukup terang tidak diubah secara tidak perlu.
DARKNESS_THRESHOLD = 100
CLAHE_CLIP_LIMIT = 2.5
CLAHE_GRID_SIZE = (8, 8)

# ── Denoise ──────────────────────────────────────────────
# Diterapkan HANYA kalau foto terdeteksi gelap (noise kamera HP paling
# terasa di kondisi minim cahaya). Memakai fastNlMeansDenoisingColored,
# dengan parameter ringan agar tidak menghaluskan detail wajah berlebihan
# (over-denoise bisa menghilangkan fitur yang justru dibutuhkan model).
DENOISE_H = 7          # kekuatan filter untuk komponen luminance
DENOISE_H_COLOR = 7    # kekuatan filter untuk komponen warna
DENOISE_TEMPLATE_SIZE = 7
DENOISE_SEARCH_SIZE = 21


# ══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def resize_standard(img_bgr: np.ndarray) -> np.ndarray:
    """Resize ke lebar maksimum standar, menjaga aspect ratio. Tidak upscale."""
    h, w = img_bgr.shape[:2]
    if w <= STANDARD_MAX_WIDTH:
        return img_bgr
    scale = STANDARD_MAX_WIDTH / w
    new_w, new_h = STANDARD_MAX_WIDTH, int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe_enhancement(img_bgr: np.ndarray) -> np.ndarray:
    """
    Terapkan CLAHE pada channel Luminance (L dari LAB color space) supaya
    kontras membaik tanpa merusak warna asli kulit/wajah.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Denoise ringan, dipakai hanya untuk foto kondisi gelap (noise kamera)."""
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None,
        DENOISE_H, DENOISE_H_COLOR,
        DENOISE_TEMPLATE_SIZE, DENOISE_SEARCH_SIZE,
    )


def preprocess_image(raw_bytes: bytes, save_path: str) -> dict:
    """
    Pipeline preprocessing standar untuk semua foto sebelum dikirim ke
    face detector & model recognition:
      1. Fix rotasi EXIF
      2. Resize ke lebar standar (konsistensi input, performa)
      3. Deteksi kondisi gelap -> kalau gelap: CLAHE + denoise ringan
      4. Simpan sebagai JPEG standar

    Return dict info preprocessing untuk logging/debugging.
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)  # fix rotasi EXIF
    original_size = img.size  # (width, height)

    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize standar SEBELUM analisa brightness, supaya konsisten
    img_bgr = resize_standard(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    is_dark = mean_brightness < DARKNESS_THRESHOLD
    denoised = False
    if is_dark:
        img_bgr = apply_clahe_enhancement(img_bgr)
        img_bgr = apply_denoise(img_bgr)
        denoised = True

    img_rgb_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    final_img = Image.fromarray(img_rgb_final)
    final_img.save(save_path, "JPEG", quality=95)

    return {
        "enhanced": is_dark,
        "denoised": denoised,
        "mean_brightness": round(mean_brightness, 1),
        "original_size": original_size,
        "final_size": final_img.size,
    }


# ══════════════════════════════════════════════════════════════════════
# DETEKSI WAJAH (untuk validasi jumlah wajah & bounding box)
# ══════════════════════════════════════════════════════════════════════

def detect_faces_with_boxes(image_path: str, detector_backend: str) -> list:
    """
    Deteksi semua wajah pada gambar dan kembalikan bounding box-nya.
    Dipakai untuk:
      1. Validasi jumlah wajah (0 = error, >1 = pilih wajah utama + warning)
      2. Data untuk visualisasi bounding box

    Return: list of dict {x, y, w, h, confidence}
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector_backend,
            enforce_detection=False,  # jangan raise di sini, kita validasi manual
            align=False,
        )
    except Exception as e:
        logger.warning(f"[DETECT_FACES] gagal ekstrak wajah: {e}")
        return []

    boxes = []
    for face in faces:
        region = face.get("facial_area", {})
        conf = face.get("confidence", None)
        if region:
            boxes.append({
                "x": region.get("x", 0),
                "y": region.get("y", 0),
                "w": region.get("w", 0),
                "h": region.get("h", 0),
                "confidence": round(float(conf), 4) if conf is not None else None,
            })
    return boxes


def select_main_face(boxes: list):
    """
    Kalau ada lebih dari satu wajah terdeteksi, pilih wajah UTAMA dengan
    heuristik: area bounding box terbesar (asumsinya wajah yang paling
    dekat ke kamera / paling dominan dalam frame adalah wajah yang absen,
    bukan orang lain yang lewat di background).
    """
    if not boxes:
        return None
    return max(boxes, key=lambda b: b["w"] * b["h"])


def draw_bounding_boxes(image_path: str, boxes: list, main_box, output_path: str):
    """
    Gambar bounding box di atas foto untuk visualisasi/debugging.
    Wajah utama (yang dipakai untuk verifikasi) digambar hijau, wajah
    lain (terdeteksi tapi diabaikan) digambar oranye dengan label.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        is_main = main_box is not None and box is main_box
        color = (0, 200, 0) if is_main else (0, 165, 255)  # BGR: hijau / oranye
        label = "WAJAH UTAMA" if is_main else "diabaikan"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    cv2.imwrite(output_path, img)


# ══════════════════════════════════════════════════════════════════════
# VERIFIKASI WAJAH
# ══════════════════════════════════════════════════════════════════════

def run_verify(path_absen: str, path_referensi: str, detector_backend: str):
    """Jalankan DeepFace.verify. Bisa raise ValueError jika wajah tidak terdeteksi."""
    return DeepFace.verify(
        img1_path=path_absen,
        img2_path=path_referensi,
        model_name="Facenet512",
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
    )


def verify_with_fallback(path_absen: str, path_referensi: str):
    """Coba PREFERRED_DETECTOR dulu, fallback ke FALLBACK_DETECTOR kalau crash teknis."""
    try:
        result = run_verify(path_absen, path_referensi, PREFERRED_DETECTOR)
        return result, PREFERRED_DETECTOR
    except ValueError:
        # Wajah memang tidak terdeteksi -> jangan ditelan, lempar lagi
        raise
    except Exception as backend_err:
        logger.warning(
            f"Detector '{PREFERRED_DETECTOR}' gagal dipakai ({backend_err}), "
            f"fallback ke '{FALLBACK_DETECTOR}'"
        )
        result = run_verify(path_absen, path_referensi, FALLBACK_DETECTOR)
        return result, FALLBACK_DETECTOR


@app.post("/verify")
async def verify_face(
    foto_absen: UploadFile = File(...),
    foto_referensi: UploadFile = File(...),
):
    tmp_dir = tempfile.mkdtemp()
    detector_used = PREFERRED_DETECTOR

    try:
        path_absen = os.path.join(tmp_dir, "absen.jpg")
        path_referensi = os.path.join(tmp_dir, "referensi.jpg")

        absen_info = preprocess_image(await foto_absen.read(), path_absen)
        ref_info = preprocess_image(await foto_referensi.read(), path_referensi)

        logger.info(
            f"[PREPROCESS] foto_absen={absen_info} | foto_referensi={ref_info}"
        )

        # ── Validasi jumlah wajah pada foto kehadiran ──
        # Foto referensi tidak divalidasi multi-wajah karena itu foto yang
        # disetel admin/HR (terkontrol), sementara foto kehadiran diambil
        # bebas oleh user dan lebih rawan ada orang lain di background.
        faces_absen = detect_faces_with_boxes(path_absen, PREFERRED_DETECTOR)
        multi_face_warning = None

        if len(faces_absen) == 0:
            logger.warning("[MULTI_FACE] Tidak ada wajah terdeteksi pada foto kehadiran")
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "face_not_detected",
                    "message": "Wajah tidak terdeteksi pada foto kehadiran. Pastikan wajah terlihat jelas dan coba foto ulang.",
                },
            )
        elif len(faces_absen) > 1:
            multi_face_warning = (
                f"Terdeteksi {len(faces_absen)} wajah pada foto. Sistem memilih wajah "
                f"yang paling dominan/dekat kamera untuk diverifikasi."
            )
            logger.warning(f"[MULTI_FACE] {len(faces_absen)} wajah terdeteksi, memilih wajah utama")

        # ── Verifikasi utama (retinaface dengan fallback opencv) ──
        result, detector_used = verify_with_fallback(path_absen, path_referensi)

        distance = float(result["distance"])
        threshold = float(result["threshold"])

        effective_threshold = threshold + VERIFIED_MARGIN
        verified = distance <= effective_threshold

        raw_confidence = 1 - (distance / (threshold * CONFIDENCE_SCALE_FACTOR))
        confidence = max(0.0, min(1.0, raw_confidence))

        logger.info(
            f"[VERIFY] detector={detector_used} "
            f"distance={distance:.4f} model_threshold={threshold:.4f} "
            f"effective_threshold={effective_threshold:.4f} "
            f"raw_confidence={raw_confidence:.4f} clipped_confidence={confidence:.4f} "
            f"verified={verified} jumlah_wajah_terdeteksi={len(faces_absen)}"
        )

        response = {
            "verified": verified,
            "distance": round(distance, 4),
            "confidence": round(confidence, 4),
            "raw_confidence": round(raw_confidence, 4),
            "threshold": round(threshold, 4),
            "effective_threshold": round(effective_threshold, 4),
            "detector_used": detector_used,
            "jumlah_wajah_terdeteksi": len(faces_absen),
        }
        if multi_face_warning:
            response["warning"] = multi_face_warning

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"[FACE_NOT_DETECTED] detector={detector_used} error={e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "face_not_detected",
                "message": "Wajah tidak terdeteksi dengan jelas pada salah satu foto. Pastikan pencahayaan cukup dan wajah terlihat penuh, lalu coba foto ulang.",
                "detector_used": detector_used,
            },
        )
    except Exception as e:
        logger.error(f"[SYSTEM_ERROR] detector={detector_used} error={e}")
        raise HTTPException(
            status_code=500,
            detail={"error_type": "system_error", "message": str(e)},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/verify/visualize")
async def verify_face_visualize(
    foto_absen: UploadFile = File(...),
    foto_referensi: UploadFile = File(...),
):
    """
    Endpoint debugging: jalankan deteksi wajah pada foto kehadiran, gambar
    bounding box (hijau = wajah utama yang dipakai, oranye = diabaikan),
    dan kembalikan gambar hasil + hasil verifikasi sebagai JSON.

    Gambar bounding box dikembalikan sebagai base64 JPEG di field
    'foto_absen_dengan_bbox' supaya bisa langsung ditampilkan/disimpan.
    """
    import base64

    tmp_dir = tempfile.mkdtemp()
    try:
        path_absen = os.path.join(tmp_dir, "absen.jpg")
        path_referensi = os.path.join(tmp_dir, "referensi.jpg")
        path_bbox_output = os.path.join(tmp_dir, "absen_bbox.jpg")

        absen_info = preprocess_image(await foto_absen.read(), path_absen)
        ref_info = preprocess_image(await foto_referensi.read(), path_referensi)

        faces_absen = detect_faces_with_boxes(path_absen, PREFERRED_DETECTOR)
        main_face = select_main_face(faces_absen)

        draw_bounding_boxes(path_absen, faces_absen, main_face, path_bbox_output)

        with open(path_bbox_output, "rb") as f:
            bbox_image_b64 = base64.b64encode(f.read()).decode("utf-8")

        verify_result = None
        if len(faces_absen) >= 1:
            try:
                result, detector_used = verify_with_fallback(path_absen, path_referensi)
                distance = float(result["distance"])
                threshold = float(result["threshold"])
                effective_threshold = threshold + VERIFIED_MARGIN
                verify_result = {
                    "distance": round(distance, 4),
                    "threshold": round(threshold, 4),
                    "effective_threshold": round(effective_threshold, 4),
                    "verified": distance <= effective_threshold,
                    "detector_used": detector_used,
                }
            except ValueError as e:
                verify_result = {"error": "face_not_detected", "message": str(e)}

        return {
            "jumlah_wajah_terdeteksi": len(faces_absen),
            "semua_bounding_box": faces_absen,
            "wajah_utama_dipilih": main_face,
            "preprocessing_absen": absen_info,
            "preprocessing_referensi": ref_info,
            "hasil_verifikasi": verify_result,
            "foto_absen_dengan_bbox": f"data:image/jpeg;base64,{bbox_image_b64}",
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/verify/debug")
async def verify_face_debug(
    foto_absen: UploadFile = File(...),
    foto_referensi: UploadFile = File(...),
):
    """
    Endpoint debugging: jalankan verifikasi dengan BEBERAPA detector backend
    sekaligus (opencv, retinaface, mtcnn) dan kembalikan semua hasilnya untuk
    dibandingkan. JANGAN dipakai di endpoint produksi utama (lambat).
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        path_absen = os.path.join(tmp_dir, "absen.jpg")
        path_referensi = os.path.join(tmp_dir, "referensi.jpg")

        absen_info = preprocess_image(await foto_absen.read(), path_absen)
        ref_info = preprocess_image(await foto_referensi.read(), path_referensi)

        backends_to_test = ["opencv", "retinaface", "mtcnn"]
        results = {
            "_preprocessing_info": {
                "foto_absen": absen_info,
                "foto_referensi": ref_info,
            }
        }

        for backend in backends_to_test:
            try:
                result = run_verify(path_absen, path_referensi, backend)
                distance = float(result["distance"])
                threshold = float(result["threshold"])
                effective_threshold = threshold + VERIFIED_MARGIN
                raw_confidence = 1 - (distance / (threshold * CONFIDENCE_SCALE_FACTOR))
                confidence = max(0.0, min(1.0, raw_confidence))

                results[backend] = {
                    "status": "ok",
                    "distance": round(distance, 4),
                    "model_threshold": round(threshold, 4),
                    "effective_threshold": round(effective_threshold, 4),
                    "raw_confidence": round(raw_confidence, 4),
                    "confidence": round(confidence, 4),
                    "verified": distance <= effective_threshold,
                }
            except ValueError as e:
                results[backend] = {"status": "face_not_detected", "message": str(e)}
            except Exception as e:
                results[backend] = {"status": "error", "message": str(e)}

        logger.info(f"[DEBUG_COMPARE] results={results}")
        return {"comparison": results}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/health")
async def health_check():
    return {"status": "ok"}