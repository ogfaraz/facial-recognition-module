import hashlib
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

from database_manager import log_attendance

MODEL_NAME = os.getenv("FR_MODEL_NAME", "Facenet512")
DETECTOR_BACKEND = os.getenv("FR_DETECTOR_BACKEND", "opencv")
REPRESENT_BACKEND = os.getenv("FR_REPRESENT_BACKEND", "skip")
COSINE_THRESHOLD = float(os.getenv("FR_COSINE_THRESHOLD", "0.33"))
SAMPLE_THRESHOLD = float(os.getenv("FR_SAMPLE_THRESHOLD", "0.34"))
MARGIN_THRESHOLD = float(os.getenv("FR_MARGIN_THRESHOLD", "0.03"))
RECOGNIZE_EVERY_N_FRAMES = int(os.getenv("FR_RECOGNIZE_EVERY_N", "6"))
DETECT_EVERY_N_FRAMES = int(os.getenv("FR_DETECT_EVERY_N", "1"))
ATTENDANCE_COOLDOWN_SECS = int(os.getenv("FR_ATTENDANCE_COOLDOWN", "30"))
PROCESS_WIDTH = int(os.getenv("FR_PROCESS_WIDTH", "320"))
MAX_FACES_PER_FRAME = int(os.getenv("FR_MAX_FACES", "4"))
MAX_ACTIVE_TRACKS = int(os.getenv("FR_MAX_ACTIVE_TRACKS", "8"))
MAX_RECOGNITIONS_PER_CYCLE = int(os.getenv("FR_MAX_RECOGNITIONS_PER_CYCLE", "2"))
TRACK_MATCH_IOU = float(os.getenv("FR_TRACK_MATCH_IOU", "0.2"))
TRACK_MAX_CENTER_DIST = float(os.getenv("FR_TRACK_MAX_CENTER_DIST", "1.25"))
CONFIRM_STREAK = int(os.getenv("FR_CONFIRM_STREAK", "2"))
VOTE_WINDOW = int(os.getenv("FR_VOTE_WINDOW", "6"))
VOTE_MIN_COUNT = int(os.getenv("FR_VOTE_MIN_COUNT", str(CONFIRM_STREAK)))
BOX_PERSIST_FRAMES = int(os.getenv("FR_BOX_PERSIST_FRAMES", "8"))
BOX_SMOOTHING_ALPHA = float(os.getenv("FR_BOX_SMOOTHING_ALPHA", "0.55"))
MIN_QUERY_FACE = int(os.getenv("FR_QUERY_MIN_FACE", "60"))
MIN_QUERY_SHARPNESS = float(os.getenv("FR_QUERY_MIN_SHARPNESS", "45"))
MIN_QUERY_BRIGHTNESS = float(os.getenv("FR_QUERY_MIN_BRIGHTNESS", "45"))
MAX_QUERY_BRIGHTNESS = float(os.getenv("FR_QUERY_MAX_BRIGHTNESS", "210"))
ENABLE_PROFILE_DETECT = os.getenv("FR_ENABLE_PROFILE_DETECT", "1").lower() in {
    "1",
    "true",
    "yes",
}
FALLBACK_DETECTOR_BACKEND = os.getenv("FR_FALLBACK_DETECTOR_BACKEND", "retinaface")
FALLBACK_DETECT_EVERY_N_FRAMES = int(os.getenv("FR_FALLBACK_DETECT_EVERY_N", "4"))
CACHE_VERSION = 3


def _load_deepface():
    # Reduce TensorFlow startup noise for cleaner production logs.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from deepface import DeepFace

    return DeepFace


def _l2_normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec
    return vec / norm


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


def _box_center(box):
    x, y, w, h = box
    return float(x + w / 2.0), float(y + h / 2.0)


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
        minNeighbors=6,
        minSize=(70, 70),
    )
    for (x, y, w, h) in frontal:
        faces.append((int(x), int(y), int(w), int(h)))

    if ENABLE_PROFILE_DETECT and not profile_detector.empty():
        profiles_left = profile_detector.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(70, 70),
        )
        for (x, y, w, h) in profiles_left:
            faces.append((int(x), int(y), int(w), int(h)))

        flipped = cv2.flip(gray, 1)
        profiles_right = profile_detector.detectMultiScale(
            flipped,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(70, 70),
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
            detector_backend=FALLBACK_DETECTOR_BACKEND,
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
            if w >= 60 and h >= 60:
                faces.append((x, y, w, h))
        return _dedupe_faces(faces)
    except Exception:
        return []


def _smooth_box(prev_box, new_box, alpha):
    if prev_box is None:
        return tuple(int(v) for v in new_box)

    px, py, pw, ph = prev_box
    nx, ny, nw, nh = new_box
    sx = int(alpha * nx + (1.0 - alpha) * px)
    sy = int(alpha * ny + (1.0 - alpha) * py)
    sw = int(alpha * nw + (1.0 - alpha) * pw)
    sh = int(alpha * nh + (1.0 - alpha) * ph)
    return sx, sy, sw, sh


def _face_quality(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    good = (
        sharpness >= MIN_QUERY_SHARPNESS
        and MIN_QUERY_BRIGHTNESS <= brightness <= MAX_QUERY_BRIGHTNESS
    )
    return sharpness, brightness, good


def _iter_database_images(database_dir):
    if not os.path.isdir(database_dir):
        return []

    image_paths = []
    for user in sorted(os.listdir(database_dir)):
        user_dir = os.path.join(database_dir, user)
        if not os.path.isdir(user_dir):
            continue
        for img in sorted(os.listdir(user_dir)):
            path = os.path.join(user_dir, img)
            if os.path.isfile(path):
                image_paths.append((user, path))
    return image_paths


def _dataset_fingerprint(image_paths):
    hasher = hashlib.sha256()
    for user, path in image_paths:
        stat = os.stat(path)
        rel_path = os.path.relpath(path, "database")
        hasher.update(f"{user}|{rel_path}|{stat.st_size}|{int(stat.st_mtime)}\n".encode("utf-8"))
    return hasher.hexdigest()


def _load_embeddings_from_cache(cache_file, expected_fingerprint):
    if not os.path.exists(cache_file):
        return None, None

    try:
        data = np.load(cache_file, allow_pickle=True)
        fingerprint = str(data["fingerprint"].item())
        model = str(data["model_name"].item())
        cache_version = int(data.get("cache_version", np.array(1)).item())
        if (
            fingerprint != expected_fingerprint
            or model != MODEL_NAME
            or cache_version != CACHE_VERSION
        ):
            return None, None
        known_embs = data["embeddings"].astype(np.float32)
        known_names = data["names"].tolist()
        if known_embs.size == 0 or len(known_names) == 0:
            return None, None
        return known_embs, known_names
    except Exception:
        return None, None


def _save_embeddings_cache(cache_file, fingerprint, known_embs, known_names):
    np.savez_compressed(
        cache_file,
        cache_version=CACHE_VERSION,
        fingerprint=fingerprint,
        model_name=MODEL_NAME,
        embeddings=np.asarray(known_embs, dtype=np.float32),
        names=np.asarray(known_names, dtype=object),
    )


def _build_known_embeddings(database_dir):
    image_paths = _iter_database_images(database_dir)
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    fingerprint = _dataset_fingerprint(image_paths)
    cache_file = os.path.join(database_dir, ".embeddings_cache.npz")
    cached_embs, cached_names = _load_embeddings_from_cache(cache_file, fingerprint)
    if cached_embs is not None:
        print(f"Loaded {len(cached_names)} embeddings from cache")
        return cached_embs, cached_names

    DeepFace = _load_deepface()
    known_embs, known_names = [], []
    print("Building embedding cache...")

    for user, img_path in image_paths:
        try:
            reps = DeepFace.represent(
                img_path,
                model_name=MODEL_NAME,
                detector_backend=REPRESENT_BACKEND,
                enforce_detection=False,
            )
            if not reps:
                continue
            emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
            emb = _l2_normalize(emb).astype(np.float32)
            known_embs.append(emb)
            known_names.append(user)
        except Exception:
            continue

    if not known_embs:
        return np.empty((0, 0), dtype=np.float32), []

    known_matrix = np.vstack(known_embs)
    _save_embeddings_cache(cache_file, fingerprint, known_matrix, known_names)
    print(f"Cached {len(known_names)} embeddings")
    return known_matrix, known_names


def _build_user_profiles(known_embs, known_names):
    profiles = {}
    for emb, name in zip(known_embs, known_names):
        profiles.setdefault(name, []).append(emb)

    out = {}
    for name, embs in profiles.items():
        mat = np.vstack(embs).astype(np.float32)
        centroid = _l2_normalize(np.mean(mat, axis=0)).astype(np.float32)
        out[name] = {
            "embeddings": mat,
            "centroid": centroid,
        }
    return out


def _identify_face(cur_emb, user_profiles, user_names, centroids):
    centroid_dists = 1.0 - np.dot(centroids, cur_emb)

    best_idx = int(np.argmin(centroid_dists))
    best_name = user_names[best_idx]
    best_centroid_dist = float(centroid_dists[best_idx])

    if len(user_names) > 1:
        sorted_d = np.sort(centroid_dists)
        margin = float(sorted_d[1] - sorted_d[0])
    else:
        margin = 1.0

    best_user_embs = user_profiles[best_name]["embeddings"]
    sample_dists = np.sort(1.0 - np.dot(best_user_embs, cur_emb))
    top_k = min(3, len(sample_dists))
    best_sample_dist = float(np.mean(sample_dists[:top_k]))

    accepted = (
        best_centroid_dist <= COSINE_THRESHOLD
        and best_sample_dist <= SAMPLE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
    )

    return {
        "accepted": accepted,
        "name": best_name,
        "centroid_dist": best_centroid_dist,
        "sample_dist": best_sample_dist,
        "margin": margin,
    }


def _new_track(track_id, box):
    return {
        "id": track_id,
        "box": tuple(int(v) for v in box),
        "ttl": BOX_PERSIST_FRAMES,
        "vote_history": deque(maxlen=VOTE_WINDOW),
        "stable_name": None,
        "stable_hold": 0,
        "verification_hint": "",
        "last_recog_frame": -10**9,
    }


def _update_track_vote(track, label):
    track["vote_history"].append(label)
    counts = Counter(track["vote_history"])
    top_label, top_count = counts.most_common(1)[0]
    if top_label != "Unknown" and top_count >= VOTE_MIN_COUNT:
        track["stable_name"] = top_label
        track["stable_hold"] = BOX_PERSIST_FRAMES
    elif top_label == "Unknown" and top_count >= VOTE_MIN_COUNT:
        if track["stable_hold"] > 0:
            track["stable_hold"] -= 1
        else:
            track["stable_name"] = None
    elif track["stable_hold"] > 0:
        track["stable_hold"] -= 1


def _match_faces_to_tracks(faces, tracks):
    if not faces:
        return {}, [], list(tracks.keys())
    if not tracks:
        return {}, list(faces), []

    track_ids = list(tracks.keys())
    pairs = []
    for face_idx, face in enumerate(faces):
        fcx, fcy = _box_center(face)
        for track_id in track_ids:
            tbox = tracks[track_id]["box"]
            tcx, tcy = _box_center(tbox)
            iou = _box_iou(face, tbox)
            center_dist = float(np.hypot(fcx - tcx, fcy - tcy))
            max_ref = max(1.0, TRACK_MAX_CENTER_DIST * float(max(tbox[2], tbox[3])))
            close_enough = center_dist <= max_ref
            if iou >= TRACK_MATCH_IOU or close_enough:
                center_score = max(0.0, 1.0 - (center_dist / max_ref))
                score = iou + (0.25 * center_score)
                pairs.append((score, face_idx, track_id))

    pairs.sort(key=lambda p: p[0], reverse=True)
    assignments = {}
    used_faces = set()
    used_tracks = set()
    for _, face_idx, track_id in pairs:
        if face_idx in used_faces or track_id in used_tracks:
            continue
        assignments[track_id] = faces[face_idx]
        used_faces.add(face_idx)
        used_tracks.add(track_id)

    unmatched_faces = [faces[i] for i in range(len(faces)) if i not in used_faces]
    unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
    return assignments, unmatched_faces, unmatched_tracks


def _select_tracks_for_recognition(tracks, frame_count):
    candidates = []
    for track in tracks.values():
        if frame_count - track["last_recog_frame"] < RECOGNIZE_EVERY_N_FRAMES:
            continue
        x, y, w, h = track["box"]
        area = float(max(1, w * h))
        staleness = float(frame_count - track["last_recog_frame"])
        priority = area + (1000.0 * staleness)
        candidates.append((priority, track))

    candidates.sort(key=lambda t: t[0], reverse=True)
    limit = max(1, MAX_RECOGNITIONS_PER_CYCLE)
    return [track for _, track in candidates[:limit]]


def run_recognition():
    database_dir = "database"
    known_embs, known_names = _build_known_embeddings(database_dir)
    if len(known_names) == 0:
        print("No registered users found. Register at least one user first.")
        return

    user_profiles = _build_user_profiles(known_embs, known_names)
    user_names = list(user_profiles.keys())
    centroids = np.vstack([user_profiles[n]["centroid"] for n in user_names])

    frontal_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
    DeepFace = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    next_track_id = 1
    last_logged = {}
    tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if w > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / float(w)
            small_frame = cv2.resize(
                frame,
                (PROCESS_WIDTH, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            resize_ratio = w / float(PROCESS_WIDTH)
        else:
            small_frame = frame
            resize_ratio = 1.0

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces_multi_pose(gray, frontal_detector, profile_detector)
            if len(faces) == 0 and frame_count % FALLBACK_DETECT_EVERY_N_FRAMES == 0:
                if DeepFace is None:
                    DeepFace = _load_deepface()
                faces = _detect_faces_fallback(DeepFace, small_frame)

            if faces:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                faces = faces[: max(1, MAX_FACES_PER_FRAME)]

            assignments, unmatched_faces, unmatched_tracks = _match_faces_to_tracks(faces, tracks)

            for track_id, new_box in assignments.items():
                track = tracks[track_id]
                track["box"] = _smooth_box(track["box"], new_box, BOX_SMOOTHING_ALPHA)
                track["ttl"] = BOX_PERSIST_FRAMES

            for track_id in unmatched_tracks:
                track = tracks.get(track_id)
                if track is not None:
                    track["ttl"] -= 1

            for new_box in unmatched_faces:
                if len(tracks) >= max(1, MAX_ACTIVE_TRACKS):
                    break
                tracks[next_track_id] = _new_track(next_track_id, new_box)
                next_track_id += 1

            expired_ids = [tid for tid, track in tracks.items() if track["ttl"] <= 0]
            for tid in expired_ids:
                tracks.pop(tid, None)

        if tracks and frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            candidates = _select_tracks_for_recognition(tracks, frame_count)
            if candidates and DeepFace is None:
                DeepFace = _load_deepface()

            for track in candidates:
                x, y, fw, fh = track["box"]
                track["last_recog_frame"] = frame_count

                if fw < MIN_QUERY_FACE or fh < MIN_QUERY_FACE:
                    track["verification_hint"] = "Face too small"
                    _update_track_vote(track, "Unknown")
                    continue

                x = max(0, x)
                y = max(0, y)
                x2 = min(small_frame.shape[1], x + fw)
                y2 = min(small_frame.shape[0], y + fh)
                face_crop = small_frame[y:y2, x:x2]

                if face_crop.size == 0:
                    _update_track_vote(track, "Unknown")
                    continue

                sharpness, brightness, is_good_quality = _face_quality(face_crop)
                if not is_good_quality:
                    track["verification_hint"] = f"Low quality s={sharpness:.0f} b={brightness:.0f}"
                    _update_track_vote(track, "Unknown")
                    continue

                try:
                    reps = DeepFace.represent(
                        face_crop,
                        model_name=MODEL_NAME,
                        detector_backend=REPRESENT_BACKEND,
                        enforce_detection=False,
                    )
                    if not reps:
                        _update_track_vote(track, "Unknown")
                        continue

                    cur_e = np.asarray(reps[0]["embedding"], dtype=np.float32)
                    cur_e = _l2_normalize(cur_e).astype(np.float32)
                    match = _identify_face(cur_e, user_profiles, user_names, centroids)
                    if match["accepted"]:
                        _update_track_vote(track, match["name"])
                        track["verification_hint"] = f"{match['name']} d={match['sample_dist']:.3f}"
                    else:
                        _update_track_vote(track, "Unknown")
                        track["verification_hint"] = (
                            f"Unknown d={match['sample_dist']:.3f} m={match['margin']:.3f}"
                        )
                except Exception:
                    track["verification_hint"] = "Embedding failed"
                    _update_track_vote(track, "Unknown")

        recognized_now = 0
        for track_id in sorted(tracks.keys()):
            track = tracks[track_id]
            x, y, fw, fh = track["box"]
            x1 = int(x * resize_ratio)
            y1 = int(y * resize_ratio)
            x2 = int((x + fw) * resize_ratio)
            y2 = int((y + fh) * resize_ratio)

            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y2 = max(0, min(frame.shape[0] - 1, y2))

            if track["stable_name"] is not None:
                recognized_now += 1
                color = (0, 220, 0)
                label = f"{track['stable_name']} (T{track_id})"
                now = time.time()
                if now - last_logged.get(track["stable_name"], 0) >= ATTENDANCE_COOLDOWN_SECS:
                    log_attendance(track["stable_name"])
                    last_logged[track["stable_name"]] = now
            else:
                color = (0, 180, 255)
                label = f"T{track_id} Verifying"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )

            if track["verification_hint"]:
                cv2.putText(
                    frame,
                    track["verification_hint"],
                    (x1, min(frame.shape[0] - 8, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 0),
                    1,
                )

        if not tracks:
            cv2.putText(
                frame,
                "No face detected",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
            )

        cv2.putText(
            frame,
            f"tracks:{len(tracks)} known:{recognized_now} detect:{DETECT_EVERY_N_FRAMES} recog:{RECOGNIZE_EVERY_N_FRAMES}",
            (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        cv2.imshow("Fast Attendance", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()