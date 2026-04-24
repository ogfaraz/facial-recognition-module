# File: recognize.py
import hashlib
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

from database_manager import log_attendance

MODEL_NAME = os.getenv("FR_MODEL_NAME", "SFace") 
DETECTOR_BACKEND = os.getenv("FR_DETECTOR_BACKEND", "opencv")
REPRESENT_BACKEND = os.getenv("FR_REPRESENT_BACKEND", "skip")
COSINE_THRESHOLD = float(os.getenv("FR_COSINE_THRESHOLD", "0.50"))
SAMPLE_THRESHOLD = float(os.getenv("FR_SAMPLE_THRESHOLD", "0.50"))
MARGIN_THRESHOLD = float(os.getenv("FR_MARGIN_THRESHOLD", "0.012")) 
RECOGNIZE_EVERY_N_FRAMES = int(os.getenv("FR_RECOGNIZE_EVERY_N", "6"))
DETECT_EVERY_N_FRAMES = int(os.getenv("FR_DETECT_EVERY_N", "1"))
ATTENDANCE_COOLDOWN_SECS = int(os.getenv("FR_ATTENDANCE_COOLDOWN", "30"))
PROCESS_WIDTH = int(os.getenv("FR_PROCESS_WIDTH", "480"))  # wider = better far-face detection
MAX_FACES_PER_FRAME = int(os.getenv("FR_MAX_FACES", "4"))
MAX_ACTIVE_TRACKS = int(os.getenv("FR_MAX_ACTIVE_TRACKS", "8"))
MAX_RECOGNITIONS_PER_CYCLE = int(os.getenv("FR_MAX_RECOGNITIONS_PER_CYCLE", "1"))
TRACK_MATCH_IOU = float(os.getenv("FR_TRACK_MATCH_IOU", "0.2"))
TRACK_MAX_CENTER_DIST = float(os.getenv("FR_TRACK_MAX_CENTER_DIST", "1.25"))
CONFIRM_STREAK = int(os.getenv("FR_CONFIRM_STREAK", "2"))
VOTE_WINDOW = int(os.getenv("FR_VOTE_WINDOW", "10"))      # wider window = more evidence required
VOTE_MIN_COUNT = int(os.getenv("FR_VOTE_MIN_COUNT", "5")) # need clear majority (5/10) to confirm
# Consecutive Unknown results needed to immediately revoke a confirmed identity.
# 3 straight Unknowns = different person, don't wait.
REVOKE_UNKNOWN_COUNT = int(os.getenv("FR_REVOKE_UNKNOWN_COUNT", "3"))

# INCREASED PERSISTENCE FOR DRIVERS: Box will survive for 15 frames (~0.5 seconds) 
# even if face is temporarily lost while turning neck.
BOX_PERSIST_FRAMES = int(os.getenv("FR_BOX_PERSIST_FRAMES", "15"))
BOX_SMOOTHING_ALPHA = float(os.getenv("FR_BOX_SMOOTHING_ALPHA", "0.55"))
MIN_QUERY_FACE = int(os.getenv("FR_QUERY_MIN_FACE", "25"))       # lowered: catch far/small faces
MIN_QUERY_SHARPNESS = float(os.getenv("FR_QUERY_MIN_SHARPNESS", "15"))  # lowered: allow slight blur at distance
MIN_QUERY_BRIGHTNESS = float(os.getenv("FR_QUERY_MIN_BRIGHTNESS", "40"))
MAX_QUERY_BRIGHTNESS = float(os.getenv("FR_QUERY_MAX_BRIGHTNESS", "210"))
ENABLE_PROFILE_DETECT = os.getenv("FR_ENABLE_PROFILE_DETECT", "1").lower() in {"1", "true", "yes"}

FALLBACK_DETECTOR_BACKEND = os.getenv("FR_FALLBACK_DETECTOR_BACKEND", "opencv")
FALLBACK_DETECT_EVERY_N_FRAMES = int(os.getenv("FR_FALLBACK_DETECT_EVERY_N", "0"))
# Minimum detection cycles a track must survive before its box is shown.
# Eliminates split-second phantom boxes from one-frame Haar false positives.
MIN_DETECT_AGE = int(os.getenv("FR_MIN_DETECT_AGE", "2"))
CACHE_VERSION = 7  # bumped: augmented embeddings + TTA recognition

persistent_id_map = {}
next_persistent_id = 1

def get_persistent_id(name):
    global next_persistent_id
    if name not in persistent_id_map:
        persistent_id_map[name] = next_persistent_id
        next_persistent_id += 1
    return persistent_id_map[name]

def _load_deepface():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from deepface import DeepFace
    return DeepFace

def _apply_clahe(gray):
    """
    CLAHE before Haar detection improves detection rate under cabin / dashcam
    lighting (shadows, glare, IR night-vision) with negligible Pi CPU cost.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _l2_normalize(vec):
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0 else vec / norm

def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    return 0.0 if union_area <= 0 else inter_area / float(union_area)

def _box_center(box):
    return float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0)

def _dedupe_faces(faces, iou_threshold=0.35):
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    deduped = []
    for face in faces:
        if all(_box_iou(face, ex) < iou_threshold for ex in deduped): deduped.append(face)
    return deduped

def _detect_faces_multi_pose(gray, frontal_detector, profile_detector, alt_detectors=None):
    """
    CLAHE-enhanced multi-pose detection.
    Runs frontal + two alt frontal cascades (fills 30-65° gap) +
    left-profile + right-profile (mirrored) cascades.
    """
    enhanced = _apply_clahe(gray)
    faces = []

    # Primary frontal: scaleFactor=1.1 + smaller minSize catches far faces.
    # minNeighbors=4 (up from 3) offsets the false-positive risk of smaller minSize.
    for (x, y, w, h) in frontal_detector.detectMultiScale(
            enhanced, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)):
        faces.append((int(x), int(y), int(w), int(h)))

    if alt_detectors:
        for det in alt_detectors:
            if not det.empty():
                # Fine-scale alt cascades: fills 30-65° angle gap AND catches small faces
                for (x, y, w, h) in det.detectMultiScale(
                        enhanced, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)):
                    faces.append((int(x), int(y), int(w), int(h)))

    if ENABLE_PROFILE_DETECT and not profile_detector.empty():
        # scaleFactor=1.05 catches small/far profiles; minNeighbors=2 vs 1 reduces noise
        for (x, y, w, h) in profile_detector.detectMultiScale(
                enhanced, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)):
            faces.append((int(x), int(y), int(w), int(h)))
        flipped = cv2.flip(enhanced, 1)
        for (x, y, w, h) in profile_detector.detectMultiScale(
                flipped, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)):
            faces.append((int(gray.shape[1] - x - w), int(y), int(w), int(h)))
    return _dedupe_faces(faces)

def _smooth_box(prev_box, new_box, alpha):
    if prev_box is None: return tuple(int(v) for v in new_box)
    return tuple(int(alpha * n + (1.0 - alpha) * p) for p, n in zip(prev_box, new_box))

def _face_quality(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    good = (sharpness >= MIN_QUERY_SHARPNESS and MIN_QUERY_BRIGHTNESS <= brightness <= MAX_QUERY_BRIGHTNESS)
    return sharpness, brightness, good


def _prepare_face_for_embedding(face_bgr):
    """
    Normalize a face crop to 112×112 (SFace native resolution) with
    CLAHE equalization on the luminance channel.

    Must be applied identically during BOTH registration (saving) and
    recognition (querying) so embeddings are always comparable:
    - Removes background context → model sees only facial features
    - Scale/distance invariant regardless of how close or far the face is
    - Lighting-robust via LAB-space equalization
    """
    face_112 = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(face_112, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

def _augment_for_embedding(face_bgr):
    """
    Produce 6 augmented variants of a 112×112 face for cache enrichment.
    Covers: lighting extremes, slight blur (far-face), and head-tilt pairs.
    Called at cache-build time for every original image; augmented images
    saved by register.py are used as-is and not double-augmented.
    Returns a flat list of augmented images (no labels needed here).
    """
    h, w = face_bgr.shape[:2]
    augs = []
    # Horizontal flip
    augs.append(cv2.flip(face_bgr, 1))
    # Bright / dark (simulate overexposure and shadow)
    augs.append(np.clip(face_bgr.astype(np.float32) * 1.30, 0, 255).astype(np.uint8))
    augs.append(np.clip(face_bgr.astype(np.float32) * 0.65, 0, 255).astype(np.uint8))
    # Gaussian blur — simulates far or out-of-focus face
    augs.append(cv2.GaussianBlur(face_bgr, (5, 5), 1.5))
    # Small in-plane rotations — covers head-tilt variation
    for angle in (10, -10):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        augs.append(cv2.warpAffine(face_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101))
    return augs


def _get_robust_embedding(face_norm, DeepFace):
    """
    Test-Time Augmentation: compute embeddings for the original face plus
    two brightness variants, then average + re-normalize.
    The averaged embedding is more stable across lighting and sensor noise,
    improving match confidence for far or partially lit faces.
    """
    variants = [
        face_norm,
        np.clip(face_norm.astype(np.float32) * 1.25, 0, 255).astype(np.uint8),
        np.clip(face_norm.astype(np.float32) * 0.75, 0, 255).astype(np.uint8),
    ]
    embeddings = []
    for v in variants:
        try:
            reps = DeepFace.represent(v, model_name=MODEL_NAME,
                                      detector_backend=REPRESENT_BACKEND,
                                      enforce_detection=False)
            if reps:
                embeddings.append(_l2_normalize(
                    np.asarray(reps[0]["embedding"], dtype=np.float32)))
        except Exception:
            pass
    if not embeddings:
        return None
    if len(embeddings) == 1:
        return embeddings[0].astype(np.float32)
    return _l2_normalize(
        np.mean(np.vstack(embeddings), axis=0)).astype(np.float32)


def _dataset_fingerprint(image_paths):
    hasher = hashlib.sha256()
    for user, path in image_paths:
        stat = os.stat(path)
        hasher.update(f"{user}|{os.path.relpath(path, 'database')}|{stat.st_size}|{int(stat.st_mtime)}\n".encode("utf-8"))
    return hasher.hexdigest()

def _build_known_embeddings(database_dir):
    image_paths = [(u, os.path.join(database_dir, u, i)) for u in sorted(os.listdir(database_dir)) if os.path.isdir(os.path.join(database_dir, u)) for i in sorted(os.listdir(os.path.join(database_dir, u)))]
    if not image_paths: return np.empty((0, 0), dtype=np.float32), []
    
    fingerprint = _dataset_fingerprint(image_paths)
    cache_file = os.path.join(database_dir, ".embeddings_cache.npz")
    
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            if str(data["fingerprint"].item()) == fingerprint and str(data["model_name"].item()) == MODEL_NAME and int(data.get("cache_version", 1)) == CACHE_VERSION:
                return data["embeddings"].astype(np.float32), data["names"].tolist()
        except Exception: pass

    DeepFace = _load_deepface()
    known_embs, known_names = [], []
    print(f"Building Pi-optimized embedding cache using {MODEL_NAME}...")

    for user, img_path in image_paths:
        try:
            # Load the saved image (may be original or an aug variant saved by register.py)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            is_original = "_aug_" not in os.path.basename(img_path)

            # For originals: embed original + 6 augmented variants.
            # For already-augmented files (saved by register.py): embed as-is.
            variants = ([img_bgr] + _augment_for_embedding(img_bgr)) if is_original else [img_bgr]

            for variant in variants:
                reps = DeepFace.represent(variant, model_name=MODEL_NAME,
                                          detector_backend=REPRESENT_BACKEND,
                                          enforce_detection=False)
                if reps:
                    known_embs.append(
                        _l2_normalize(np.asarray(reps[0]["embedding"],
                                                 dtype=np.float32)).astype(np.float32))
                    known_names.append(user)
        except Exception:
            continue

    known_matrix = np.vstack(known_embs) if known_embs else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(cache_file, cache_version=CACHE_VERSION, fingerprint=fingerprint, model_name=MODEL_NAME, embeddings=known_matrix, names=np.asarray(known_names, dtype=object))
    return known_matrix, known_names

def _identify_face(cur_emb, known_embs, known_names):
    """
    Per-person nearest-neighbour identification.

    For each registered person, take their MINIMUM cosine distance to the
    query embedding (i.e. the closest sample they have, regardless of pose).
    Then pick the person with the overall best match and require:
      1. Their best distance is below SAMPLE_THRESHOLD.
      2. The gap to the next-best person is above MARGIN_THRESHOLD.

    This is the correct approach for a diverse-pose database: if a driver
    is registered at 45-degree profile AND the query is also a 45-degree
    profile, that registered sample will have a low distance even when all
    frontal samples for the same person have high distances.  The old top-3
    voting averaged those high-distance frontal samples in, causing Unknown.
    """
    dists = 1.0 - np.dot(known_embs, cur_emb)

    # Per-person best (minimum) cosine distance
    person_best: dict = {}
    for name, dist in zip(known_names, dists):
        d = float(dist)
        if name not in person_best or d < person_best[name]:
            person_best[name] = d

    if not person_best:
        return {"accepted": False, "name": "Unknown", "sample_dist": 1.0, "margin": 0.0}

    sorted_persons = sorted(person_best.items(), key=lambda x: x[1])
    best_name, best_dist = sorted_persons[0]
    next_best_dist = sorted_persons[1][1] if len(sorted_persons) > 1 else 1.0
    margin = next_best_dist - best_dist

    accepted = best_dist <= SAMPLE_THRESHOLD and margin >= MARGIN_THRESHOLD
    return {"accepted": accepted, "name": best_name, "sample_dist": best_dist, "margin": margin}

def _new_track(track_id, box):
    return {
        "id": track_id,
        "box": tuple(int(v) for v in box),
        "ttl": BOX_PERSIST_FRAMES,
        "age_detections": 0,          # times matched to a real detection; suppresses phantom boxes
        "vote_history": deque(maxlen=VOTE_WINDOW),
        "stable_name": None,          # None = Verifying, "Unknown" = confirmed stranger
        "consecutive_unknown": 0,     # back-to-back Unknown results
        "verification_hint": "",
        "last_recog_frame": -10**9
    }

def _update_track_vote(track, label):
    """
    Two-stage identity state machine:

    GRANT  — a named person wins the vote window majority (>= VOTE_MIN_COUNT).
             stable_name is set immediately.

    REVOKE — REVOKE_UNKNOWN_COUNT consecutive Unknown results arrive.
             Identity is wiped immediately (back to None / Verifying).
             Vote history is also cleared so re-confirmation requires fresh evidence.
             This means a different person walking into frame loses the old label
             within ~0.5-1 second instead of dragging it for many seconds.

    The old stable_hold countdown is removed — it caused identity to linger
    long after a different face was clearly in frame.
    """
    track["vote_history"].append(label)
    counts = Counter(track["vote_history"])
    top_label, top_count = counts.most_common(1)[0]

    if label == "Unknown":
        track["consecutive_unknown"] += 1
    else:
        track["consecutive_unknown"] = 0

    # ── Consecutive-Unknown handler ────────────────────────────────────────
    # REVOKE: only fires when a NAMED identity was already granted.
    #   → wipes identity so a new face can earn its own label.
    # CONFIRM-UNKNOWN: fires when the track is still Verifying (stable_name=None)
    #   → marks them Unknown immediately instead of looping forever.
    # If already "Unknown", consecutive hits have no extra effect.
    if track["consecutive_unknown"] >= REVOKE_UNKNOWN_COUNT:
        if track["stable_name"] is not None and track["stable_name"] != "Unknown":
            # A recognised person has stopped matching → revoke and let re-verify
            track["stable_name"] = None
            track["vote_history"].clear()
            track["consecutive_unknown"] = 0
        elif track["stable_name"] is None:
            # Unverified face keeps coming back Unknown → confirm as Unknown now
            track["stable_name"] = "Unknown"
        # If already "Unknown" — nothing to change
        return

    # Grant identity when vote majority reached
    if top_label != "Unknown" and top_count >= VOTE_MIN_COUNT:
        track["stable_name"] = top_label
    elif top_label == "Unknown" and top_count >= VOTE_MIN_COUNT:
        track["stable_name"] = "Unknown"

def _match_faces_to_tracks(faces, tracks):
    if not faces: return {}, [], list(tracks.keys())
    if not tracks: return {}, list(faces), []
    
    pairs = []
    for f_idx, face in enumerate(faces):
        fcx, fcy = _box_center(face)
        for tid, trk in tracks.items():
            tbox = trk["box"]
            iou = _box_iou(face, tbox)
            dist = np.hypot(fcx - _box_center(tbox)[0], fcy - _box_center(tbox)[1])
            max_ref = max(1.0, TRACK_MAX_CENTER_DIST * max(tbox[2], tbox[3]))
            if iou >= TRACK_MATCH_IOU or dist <= max_ref:
                pairs.append((iou + 0.25 * max(0.0, 1.0 - dist / max_ref), f_idx, tid))
    
    assignments, used_faces, used_tracks = {}, set(), set()
    for _, f_idx, tid in sorted(pairs, key=lambda p: p[0], reverse=True):
        if f_idx not in used_faces and tid not in used_tracks:
            assignments[tid] = faces[f_idx]
            used_faces.add(f_idx); used_tracks.add(tid)
            
    return assignments, [f for i, f in enumerate(faces) if i not in used_faces], [t for t in tracks if t not in used_tracks]

def _select_tracks_for_recognition(tracks, frame_count):
    candidates = []
    for track in tracks.values():
        is_known = track["stable_name"] is not None and track["stable_name"] != "Unknown"
        required_wait = RECOGNIZE_EVERY_N_FRAMES * 5 if is_known else RECOGNIZE_EVERY_N_FRAMES
        frames_since = frame_count - track["last_recog_frame"]
        
        if frames_since < required_wait: continue
            
        candidates.append((float(max(1, track["box"][2] * track["box"][3])) + (1000.0 * frames_since), track))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [track for _, track in candidates[:max(1, MAX_RECOGNITIONS_PER_CYCLE)]]

def run_recognition():
    known_embs, known_names = _build_known_embeddings("database")
    if len(known_names) == 0: return print("No registered users found. Register first.")
    if isinstance(known_names, np.ndarray): known_names = known_names.tolist()

    frontal_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    alt_detectors = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    ]
    DeepFace = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count, next_track_id, tracks, last_logged = 0, 1, {}, {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)  # mirror to match registered (mirrored) samples

        scale = PROCESS_WIDTH / float(frame.shape[1]) if frame.shape[1] > PROCESS_WIDTH else 1.0
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        resize_ratio = 1.0 / scale if scale < 1.0 else 1.0

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces_multi_pose(gray, frontal_detector, profile_detector, alt_detectors)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max(1, MAX_FACES_PER_FRAME)]
            assignments, unmatched_faces, unmatched_tracks = _match_faces_to_tracks(faces, tracks)

            for tid, new_box in assignments.items():
                tracks[tid]["box"] = _smooth_box(tracks[tid]["box"], new_box, BOX_SMOOTHING_ALPHA)
                tracks[tid]["ttl"] = BOX_PERSIST_FRAMES
                tracks[tid]["age_detections"] += 1  # confirmed by a real detection this cycle

            for tid in unmatched_tracks: tracks[tid]["ttl"] -= 1
            for new_box in unmatched_faces:
                if len(tracks) < max(1, MAX_ACTIVE_TRACKS):
                    tracks[next_track_id] = _new_track(next_track_id, new_box)
                    next_track_id += 1
            
            tracks = {tid: t for tid, t in tracks.items() if t["ttl"] > 0}

        if tracks:
            for track in _select_tracks_for_recognition(tracks, frame_count):
                if DeepFace is None: DeepFace = _load_deepface()
                track["last_recog_frame"] = frame_count
                x, y, fw, fh = track["box"]
                
                if fw < MIN_QUERY_FACE or fh < MIN_QUERY_FACE: 
                    track["verification_hint"] = "Face too small"
                    continue

                face_crop = small_frame[max(0, y):min(small_frame.shape[0], y + fh), max(0, x):min(small_frame.shape[1], x + fw)]
                if face_crop.size == 0: continue

                sharpness, brightness, is_good = _face_quality(face_crop)
                if not is_good: 
                    track["verification_hint"] = f"Low quality s={sharpness:.0f} b={brightness:.0f}"
                    continue

                # Normalize to 112x112 with CLAHE — identical preprocessing to registration.
                # This is what makes close and far faces produce the same embedding space.
                face_norm = _prepare_face_for_embedding(face_crop)

                try:
                    # TTA: average embeddings across brightness variants for robustness
                    cur_e = _get_robust_embedding(face_norm, DeepFace)
                    if cur_e is not None:
                        match = _identify_face(cur_e, known_embs, known_names)
                        
                        if match["accepted"]:
                            _update_track_vote(track, match["name"])
                            track["verification_hint"] = f"{match['name']} d={match['sample_dist']:.2f}"
                        else:
                            _update_track_vote(track, "Unknown")
                            track["verification_hint"] = f"Unknown d={match['sample_dist']:.2f} m={match['margin']:.2f}"
                except Exception: 
                    track["verification_hint"] = "Embedding failed"

        recognized_now = 0
        for track_id, track in tracks.items():
            # Suppress phantom boxes: don't render until the track has been confirmed
            # by at least MIN_DETECT_AGE real detection cycles. One-frame Haar false
            # positives die out before reaching this threshold.
            if track["age_detections"] < MIN_DETECT_AGE and track["stable_name"] is None:
                continue

            x, y, fw, fh = track["box"]
            x1 = max(0, min(frame.shape[1] - 1, int(x * resize_ratio)))
            y1 = max(0, min(frame.shape[0] - 1, int(y * resize_ratio)))
            x2 = max(0, min(frame.shape[1] - 1, int((x + fw) * resize_ratio)))
            y2 = max(0, min(frame.shape[0] - 1, int((y + fh) * resize_ratio)))

            if track["stable_name"] is not None and track["stable_name"] != "Unknown":
                recognized_now += 1
                color = (0, 220, 0)
                pid = get_persistent_id(track["stable_name"])
                label = f"{track['stable_name']} (ID:{pid})"
                
                now = time.time()
                if now - last_logged.get(track["stable_name"], 0) >= ATTENDANCE_COOLDOWN_SECS:
                    log_attendance(track["stable_name"])
                    last_logged[track["stable_name"]] = now
            elif track["stable_name"] == "Unknown":
                color = (0, 0, 255) 
                label = f"Unknown (Trk:{track_id})"
            else:
                color = (0, 180, 255)
                label = f"Verifying (Trk:{track_id})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(25, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            
            if track["verification_hint"]:
                cv2.putText(frame, track["verification_hint"], (x1, min(frame.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)

        if not tracks:
            cv2.putText(frame, "No face detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)

        cv2.putText(frame, f"Driver Tracking Mode | Tracks:{len(tracks)} Known:{recognized_now}", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imshow("Fast Attendance", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()