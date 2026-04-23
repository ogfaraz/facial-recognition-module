import os
import time
from collections import deque

import cv2
import numpy as np

TARGET_SAMPLES = int(os.getenv("FR_REGISTER_SAMPLES", "45"))
MAX_CAPTURE_SECONDS = int(os.getenv("FR_REGISTER_TIMEOUT", "35"))
CAPTURE_EVERY_N_FRAMES = int(os.getenv("FR_REGISTER_INTERVAL", "3"))
MIN_FACE_SIZE = int(os.getenv("FR_REGISTER_MIN_FACE", "90"))
MIN_SHARPNESS = float(os.getenv("FR_REGISTER_MIN_SHARPNESS", "80"))
MIN_BRIGHTNESS = float(os.getenv("FR_REGISTER_MIN_BRIGHTNESS", "45"))
MAX_BRIGHTNESS = float(os.getenv("FR_REGISTER_MAX_BRIGHTNESS", "210"))
MAX_HIST_SIMILARITY = float(os.getenv("FR_REGISTER_MAX_HIST_SIM", "0.985"))
REGISTER_HIST_MEMORY = int(os.getenv("FR_REGISTER_HIST_MEMORY", "24"))
REGISTER_FACE_MARGIN = int(os.getenv("FR_REGISTER_FACE_MARGIN", "20"))
REGISTER_ENABLE_PROFILE_DETECT = os.getenv("FR_REGISTER_ENABLE_PROFILE_DETECT", "1").lower() in {
    "1",
    "true",
    "yes",
}
REGISTER_FALLBACK_DETECTOR_BACKEND = os.getenv("FR_REGISTER_FALLBACK_DETECTOR_BACKEND", "retinaface")
REGISTER_FALLBACK_EVERY_N_FRAMES = int(os.getenv("FR_REGISTER_FALLBACK_EVERY_N", "4"))


def _load_deepface():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from deepface import DeepFace

    return DeepFace


def _largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def _dedupe_faces(faces, iou_threshold=0.35):
    if not faces:
        return []

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    deduped = []
    for face in faces:
        if all(_box_iou(face, existing) < iou_threshold for existing in deduped):
            deduped.append(face)
    return deduped


def _detect_faces_multi_pose(gray, frontal_detector, profile_detector):
    faces = []
    frontal = frontal_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    for (x, y, w, h) in frontal:
        faces.append((int(x), int(y), int(w), int(h)))

    if REGISTER_ENABLE_PROFILE_DETECT and not profile_detector.empty():
        profiles_left = profile_detector.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        for (x, y, w, h) in profiles_left:
            faces.append((int(x), int(y), int(w), int(h)))

        flipped = cv2.flip(gray, 1)
        profiles_right = profile_detector.detectMultiScale(
            flipped,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        width = gray.shape[1]
        for (x, y, w, h) in profiles_right:
            rx = width - x - w
            faces.append((int(rx), int(y), int(w), int(h)))

    return _dedupe_faces(faces)


def _detect_faces_fallback(DeepFace, frame_bgr):
    try:
        objs = DeepFace.extract_faces(
            img_path=frame_bgr,
            detector_backend=REGISTER_FALLBACK_DETECTOR_BACKEND,
            enforce_detection=False,
            align=False,
        )
        faces = []
        for obj in objs:
            area = obj.get("facial_area") or {}
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                faces.append((x, y, w, h))
        return _dedupe_faces(faces)
    except Exception:
        return []


def _face_quality(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    good = (
        sharpness >= MIN_SHARPNESS
        and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    )
    return sharpness, brightness, good


def _face_hist(face_bgr):
    hist = cv2.calcHist([face_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def register_user(name):
    user_dir = os.path.join("database", name)
    os.makedirs(user_dir, exist_ok=True)

    # Start clean so stale/poor images do not keep hurting identity quality.
    for item in os.listdir(user_dir):
        path = os.path.join(user_dir, item)
        if os.path.isfile(path) and item.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(path)

    frontal_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    DeepFace = None

    print("Auto registration started.")
    print("Slowly rotate your head left/right, change expression, and slightly change distance.")
    print("No key press needed. Press Q only to cancel.")

    start = time.time()
    frame_count = 0
    captured = 0
    recent_hists = deque(maxlen=max(1, REGISTER_HIST_MEMORY))

    while captured < TARGET_SAMPLES and (time.time() - start) < MAX_CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_faces_multi_pose(gray, frontal_detector, profile_detector)
        if len(faces) == 0 and frame_count % REGISTER_FALLBACK_EVERY_N_FRAMES == 0:
            if DeepFace is None:
                DeepFace = _load_deepface()
            faces = _detect_faces_fallback(DeepFace, frame)

        best_face = _largest_face(faces)
        status_color = (0, 180, 255)
        status_text = "Looking for face..."

        if best_face is not None:
            x, y, w, h = best_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

            margin = REGISTER_FACE_MARGIN
            h_img, w_img, _ = frame.shape
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(w_img, x + w + margin)
            y1 = min(h_img, y + h + margin)
            face_crop = frame[y0:y1, x0:x1]

            sharpness, brightness, good_quality = _face_quality(face_crop)
            status_text = f"Sharp:{sharpness:.0f} Bright:{brightness:.0f}"

            should_capture = (
                frame_count % CAPTURE_EVERY_N_FRAMES == 0 and good_quality
            )
            if should_capture:
                cur_hist = _face_hist(face_crop)
                if len(recent_hists) == 0:
                    distinct_enough = True
                else:
                    similarities = [
                        cv2.compareHist(h.astype("float32"), cur_hist.astype("float32"), cv2.HISTCMP_CORREL)
                        for h in recent_hists
                    ]
                    distinct_enough = max(similarities) <= MAX_HIST_SIMILARITY

                if distinct_enough:
                    out_path = os.path.join(user_dir, f"img_{captured + 1:03d}.jpg")
                    cv2.imwrite(out_path, face_crop)
                    captured += 1
                    recent_hists.append(cur_hist)
                    status_color = (0, 220, 0)
                    status_text = f"Captured {captured}/{TARGET_SAMPLES}"
                else:
                    status_text = "Need more variation (pose/expression/distance)"

        elapsed = int(time.time() - start)
        cv2.putText(
            frame,
            f"Samples: {captured}/{TARGET_SAMPLES} | Time: {elapsed}s/{MAX_CAPTURE_SECONDS}s",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            status_text,
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2,
        )

        cv2.imshow("Auto Registration", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured >= max(10, TARGET_SAMPLES // 3):
        print(f"Registration complete for {name}: {captured} samples saved.")
    else:
        print(f"Registration incomplete for {name}: only {captured} samples saved. Try again with better lighting.")
