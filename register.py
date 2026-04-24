# File: register.py
import os
import time
from collections import deque

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants  (all overridable via environment variables)
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SAMPLES         = int(os.getenv("FR_REGISTER_SAMPLES",        "100"))
MAX_CAPTURE_SECONDS    = int(os.getenv("FR_REGISTER_TIMEOUT",        "300"))   # was 180 — allows all 11 phases to breathe
CAPTURE_EVERY_N_FRAMES = int(os.getenv("FR_REGISTER_INTERVAL",       "5"))    # backup frame gate
MIN_CAPTURE_INTERVAL   = float(os.getenv("FR_REGISTER_MIN_INTERVAL",  "0.30"))  # was 0.22 — slightly slower capture pace
PHASE_HOLD_SECS        = float(os.getenv("FR_REGISTER_PHASE_HOLD",   "4.0"))   # was 3.0 — more time to get into position
MIN_PHASE_SECS         = float(os.getenv("FR_REGISTER_MIN_PHASE",    "12.0"))  # was 8.0 — each phase gets real coverage
# If no face is detected for this many seconds during a phase, auto-skip to the next.
# Increased to 20s so extreme profile / far-away phases don't skip prematurely.
PHASE_SKIP_NO_DETECT_SECS = float(os.getenv("FR_REGISTER_PHASE_SKIP", "20.0"))
MIN_FACE_SIZE          = int(os.getenv("FR_REGISTER_MIN_FACE",       "80"))
MIN_SHARPNESS          = float(os.getenv("FR_REGISTER_MIN_SHARPNESS","60"))
MIN_BRIGHTNESS         = float(os.getenv("FR_REGISTER_MIN_BRIGHTNESS","40"))
MAX_BRIGHTNESS         = float(os.getenv("FR_REGISTER_MAX_BRIGHTNESS","215"))
MAX_HIST_SIMILARITY    = float(os.getenv("FR_REGISTER_MAX_HIST_SIM", "0.985"))
REGISTER_HIST_MEMORY   = int(os.getenv("FR_REGISTER_HIST_MEMORY",   "24"))
REGISTER_FACE_MARGIN   = int(os.getenv("FR_REGISTER_FACE_MARGIN",   "20"))
REGISTER_ENABLE_PROFILE_DETECT = (
    os.getenv("FR_REGISTER_ENABLE_PROFILE_DETECT", "1").lower() in {"1", "true", "yes"}
)

# Always 0 – prevents accidental RetinaFace fallback on Pi hardware
REGISTER_FALLBACK_EVERY_N_FRAMES = 0

# ──────────────────────────────────────────────────────────────────────────────
# Guided registration phases  (10 driver-specific poses)
#
# Each entry: (display_label, fraction_of_total_samples, sharpness_override)
#   sharpness_override = None  -> use global MIN_SHARPNESS
#   sharpness_override = float -> relaxed value for profile / tilt phases
#                                 where slight blur is expected and acceptable
# ──────────────────────────────────────────────────────────────────────────────
REGISTRATION_PHASES = [
    # label                                              fraction  sharpness_override
    ("Look straight at camera  [NEUTRAL]",               0.09,     None ),
    ("Slight LEFT turn - keep BOTH eyes visible",        0.07,     None ),
    ("Slight RIGHT turn - keep BOTH eyes visible",       0.07,     None ),
    ("Hard LEFT - checking LEFT side mirror",            0.10,     42.0 ),
    ("Hard RIGHT - checking RIGHT side mirror",          0.10,     42.0 ),
    ("Look DOWN - GPS / phone / dashboard",              0.08,     48.0 ),
    ("Tilt UP - sun visor / overhead console",           0.07,     None ),
    ("Natural DRIVING pose - relax, eyes ahead",         0.08,     None ),
    ("Change EXPRESSION freely (smile / talk / serious)",0.08,     None ),
    ("Move VERY CLOSE - within arm's reach of camera",  0.13,     None ),  # close-up samples
    ("Move VERY FAR back from camera",                   0.13,     None ),  # far-away samples
]

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_clahe(gray):
    """
    Adaptive histogram equalisation.
    Significantly improves Haar cascade detection in cabin/dashcam lighting
    (harsh shadows, sun glare, night IR) with negligible Pi CPU cost.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    return 0.0 if union_area <= 0 else inter_area / float(union_area)


def _dedupe_faces(faces, iou_threshold=0.35):
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    deduped = []
    for face in faces:
        if all(_box_iou(face, ex) < iou_threshold for ex in deduped):
            deduped.append(face)
    return deduped


def _detect_faces_multi_pose(gray, frontal_detector, profile_detector, alt_detectors=None):
    """
    CLAHE-enhanced multi-pose Haar detection.
    Runs:
      - Default frontal cascade
      - Two alternate frontal cascades (catches 30-60 degree turns the default misses)
      - Profile cascade + mirrored (for 70-90 degree side views)
    minNeighbors intentionally low; the histogram diversity gate prevents duplicates.
    """
    enhanced = _apply_clahe(gray)
    faces = []

    for (x, y, w, h) in frontal_detector.detectMultiScale(
            enhanced, scaleFactor=1.2, minNeighbors=3,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)):
        faces.append((int(x), int(y), int(w), int(h)))

    # Alt frontal cascades fill the 30-65 degree gap
    if alt_detectors:
        for det in alt_detectors:
            if not det.empty():
                for (x, y, w, h) in det.detectMultiScale(
                        enhanced, scaleFactor=1.1, minNeighbors=2,
                        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)):
                    faces.append((int(x), int(y), int(w), int(h)))

    if REGISTER_ENABLE_PROFILE_DETECT and not profile_detector.empty():
        # minNeighbors=1: maximum sensitivity for extreme profiles
        for (x, y, w, h) in profile_detector.detectMultiScale(
                enhanced, scaleFactor=1.1, minNeighbors=1,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)):
            faces.append((int(x), int(y), int(w), int(h)))

        flipped = cv2.flip(enhanced, 1)
        for (x, y, w, h) in profile_detector.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=1,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)):
            faces.append((int(gray.shape[1] - x - w), int(y), int(w), int(h)))

    return _dedupe_faces(faces)


def _face_quality(face_bgr, sharpness_threshold=None):
    """
    Returns (sharpness, brightness, is_good).
    sharpness_threshold overrides MIN_SHARPNESS – used by profile phases
    where slight motion-blur is expected and should not block capture.
    """
    if sharpness_threshold is None:
        sharpness_threshold = MIN_SHARPNESS
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    good = sharpness >= sharpness_threshold and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    return sharpness, brightness, good


def _estimate_pose_label(face_gray, eye_detector):
    """
    Lightweight pose indicator via eye counting.
    2 eyes -> Frontal   |   1 eye -> Side   |   0 eyes -> Extreme
    Used only for user feedback; does not block capture.
    """
    eyes = eye_detector.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
    n = len(eyes)
    if n >= 2:
        return "Frontal"
    if n == 1:
        return "Side"
    return "Extreme"


def _face_hist(face_bgr):
    hist = cv2.calcHist([face_bgr], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _prepare_face_for_embedding(face_bgr):
    """
    Normalize a face crop to 112×112 (SFace native resolution) with
    CLAHE equalization on the luminance channel.

    This is the single most important preprocessing step for accuracy:
    - Strips all background context → SFace learns ONLY facial features,
      not the camera background, wall colour, or seat texture behind the driver.
    - Scale/distance invariant: a close face and a far face produce identical
      output dimensions, so embeddings are comparable regardless of distance.
    - Lighting-robust: LAB-space equalization reduces sensitivity to
      cabin shadows, sun glare, and dashcam IR differences.
    """
    face_112 = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(face_112, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _augment_sample(face_112):
    """
    Generate 6 augmented variants of a 112×112 normalized face.
    Saved alongside each original so the embedding database covers:
      - Lighting extremes (bright/overexposed, dark/shadowed)
      - Far / out-of-focus faces (blur)
      - In-plane head-tilt (±10°)
      - Mirror pose (horizontal flip)
      - Low-quality / noisy sensor (Gaussian noise)
    Returns list of (filename_suffix, image_array) tuples.
    """
    h, w = face_112.shape[:2]
    variants = []

    # 1. Horizontal flip — mirror pose variation
    variants.append(("aug_flip", cv2.flip(face_112, 1)))

    # 2. Bright — overexposure / direct light
    variants.append(("aug_bright",
        np.clip(face_112.astype(np.float32) * 1.35, 0, 255).astype(np.uint8)))

    # 3. Dark — cabin shadow / low-light
    variants.append(("aug_dark",
        np.clip(face_112.astype(np.float32) * 0.60, 0, 255).astype(np.uint8)))

    # 4. Blur — simulates far distance or out-of-focus
    variants.append(("aug_blur", cv2.GaussianBlur(face_112, (5, 5), 1.5)))

    # 5. Rotation +10° — head tilt right
    M_pos = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 10, 1.0)
    variants.append(("aug_rot_p10",
        cv2.warpAffine(face_112, M_pos, (w, h), borderMode=cv2.BORDER_REFLECT_101)))

    # 6. Rotation −10° — head tilt left
    M_neg = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -10, 1.0)
    variants.append(("aug_rot_n10",
        cv2.warpAffine(face_112, M_neg, (w, h), borderMode=cv2.BORDER_REFLECT_101)))

    return variants


# ──────────────────────────────────────────────────────────────────────────────
# Main registration entry point
# ──────────────────────────────────────────────────────────────────────────────
def register_user(name):
    user_dir = os.path.join("database", name)
    os.makedirs(user_dir, exist_ok=True)

    # Clear old images so re-registration starts clean
    for item in os.listdir(user_dir):
        path = os.path.join(user_dir, item)
        if os.path.isfile(path) and item.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(path)

    frontal_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml")
    eye_detector     = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    # Alt frontal cascades fill the 30-65° gap between frontal and profile
    alt_detectors = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    ]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 52)
    print(f"  Guided Driver Registration  ->  {name}")
    print(f"  Target : {TARGET_SAMPLES} diverse samples  |  "
          f"Timeout: {MAX_CAPTURE_SECONDS}s")
    print("  Follow the pose instructions on screen.")
    print("  Press Q to cancel at any time.")
    print("=" * 52 + "\n")

    # Build per-phase cumulative sample targets
    cumulative_targets = []
    running = 0
    for i, (_, pct, _) in enumerate(REGISTRATION_PHASES):
        if i < len(REGISTRATION_PHASES) - 1:
            running += int(TARGET_SAMPLES * pct)
            cumulative_targets.append(running)
        else:
            cumulative_targets.append(TARGET_SAMPLES)

    n_phases            = len(REGISTRATION_PHASES)
    start               = time.time()
    frame_count         = 0
    captured            = 0
    recent_hists        = deque(maxlen=max(1, REGISTER_HIST_MEMORY))
    phase_idx           = 0
    phase_pause_until   = start + PHASE_HOLD_SECS
    phase_capture_start = None
    last_capture_time   = 0.0
    last_face_time      = start   # tracks when we last saw a face (for auto-skip)
    pose_label          = ""

    while captured < TARGET_SAMPLES and (time.time() - start) < MAX_CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frame        = cv2.flip(frame, 1)   # mirror so left/right match natural movement
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        faces = _detect_faces_multi_pose(gray, frontal_detector, profile_detector, alt_detectors)

        is_paused = current_time < phase_pause_until

        # Track when active capture starts for this phase (after countdown ends)
        if not is_paused and phase_capture_start is None:
            phase_capture_start = current_time

        # ── Advance phase only when BOTH conditions are met ────────────────
        # 1. Enough samples collected for this phase.
        # 2. At least MIN_PHASE_SECS of active capture time elapsed.
        #    This prevents rapid phase cycling when the user is far away
        #    and the face still gets detected on every frame.
        phase_time_done = (
            phase_capture_start is not None
            and (current_time - phase_capture_start) >= MIN_PHASE_SECS
        )
        if (phase_idx < n_phases - 1
                and captured >= cumulative_targets[phase_idx]
                and phase_time_done):
            phase_idx           += 1
            phase_pause_until    = current_time + PHASE_HOLD_SECS
            phase_capture_start  = None
            recent_hists.clear()
            pose_label = ""
            is_paused  = True  # immediately enter countdown for new phase

        phase_label, _, phase_sharpness = REGISTRATION_PHASES[phase_idx]

        best_face    = _largest_face(faces)
        status_color = (0, 180, 255)
        status_text  = "Looking for face..."

        if best_face is not None:
            last_face_time = current_time  # reset no-detection timer
            x, y, w, h = best_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

            # Tight crop — no margin means no background encoded into the embedding
            H_img, W_img = frame.shape[:2]
            x0 = max(0, x);  y0 = max(0, y)
            x1 = min(W_img, x + w);  y1 = min(H_img, y + h)
            face_crop = frame[y0:y1, x0:x1]

            # Pose estimation (eye cascade – lightweight, Pi-safe)
            face_gray_crop = gray[y0:y1, x0:x1]
            pose_label = _estimate_pose_label(face_gray_crop, eye_detector)

            sharpness, brightness, good_quality = _face_quality(face_crop, phase_sharpness)
            status_text = f"Sharp:{sharpness:.0f}  Bright:{brightness:.0f}"

            time_ok = (current_time - last_capture_time) >= MIN_CAPTURE_INTERVAL
            should_capture = (
                not is_paused
                and frame_count % CAPTURE_EVERY_N_FRAMES == 0
                and time_ok
                and good_quality
            )

            if should_capture:
                # Diversity check on the normalized face (what actually gets saved)
                face_norm = _prepare_face_for_embedding(face_crop)
                cur_hist = _face_hist(face_norm)
                distinct_enough = True
                if recent_hists:
                    sims = [
                        cv2.compareHist(
                            h.astype("float32"), cur_hist.astype("float32"),
                            cv2.HISTCMP_CORREL)
                        for h in recent_hists
                    ]
                    distinct_enough = max(sims) <= MAX_HIST_SIMILARITY

                if distinct_enough:
                    out_path = os.path.join(user_dir, f"img_{captured + 1:03d}.jpg")
                    # Save the 112×112 normalized face — no background, scale-invariant
                    cv2.imwrite(out_path, face_norm)

                    # Save augmented variants alongside the original.
                    # Covers: low-light, overexposure, blur (far face), head-tilt, mirror.
                    aug_base = os.path.join(user_dir, f"img_{captured + 1:03d}")
                    for suffix, aug_img in _augment_sample(face_norm):
                        cv2.imwrite(f"{aug_base}_{suffix}.jpg", aug_img)

                    captured          += 1
                    last_capture_time  = current_time
                    recent_hists.append(cur_hist)
                    status_color  = (0, 220, 0)
                    aug_total = captured * 6   # 6 augments per original
                    status_text   = f"Captured {captured}/{TARGET_SAMPLES}  (+{aug_total} augmented)"
                else:
                    status_color = (0, 160, 255)
                    status_text  = "Move slightly - need more variation"

            elif not good_quality and not is_paused:
                status_color = (0, 80, 255)
                min_sharp = phase_sharpness if phase_sharpness is not None else MIN_SHARPNESS
                status_text = (f"Low quality  sharp:{sharpness:.0f}/{min_sharp:.0f}"
                               f"  bright:{brightness:.0f}")
        else:
            pose_label = ""
            # Auto-skip: if no face detected for too long during an active phase,
            # move on rather than burning the entire session timeout.
            if (not is_paused
                    and phase_idx < n_phases - 1
                    and (current_time - last_face_time) > PHASE_SKIP_NO_DETECT_SECS):
                phase_idx           += 1
                phase_pause_until    = current_time + PHASE_HOLD_SECS
                phase_capture_start  = None
                last_face_time       = current_time  # reset timer for new phase
                recent_hists.clear()
                is_paused            = True
                phase_label, _, phase_sharpness = REGISTRATION_PHASES[phase_idx]

        # ── UI rendering ──────────────────────────────────────────────────
        H, W = frame.shape[:2]
        elapsed = int(current_time - start)

        # Top header bar
        cv2.rectangle(frame, (0, 0), (W, 46), (20, 20, 20), -1)
        cv2.putText(frame, f"Registering: {name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 220, 70), 2)
        cv2.putText(frame,
                    f"Samples: {captured}/{TARGET_SAMPLES}  |  "
                    f"Time: {elapsed}s / {MAX_CAPTURE_SECONDS}s",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1)

        # Overall progress bar
        bar_y = 46
        bar_fill = int(W * captured / max(1, TARGET_SAMPLES))
        cv2.rectangle(frame, (0, bar_y), (W, bar_y + 7), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, bar_y), (bar_fill, bar_y + 7), (0, 200, 80), -1)

        # Status / quality line
        cv2.putText(frame, status_text,
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

        # Pose indicator (top-right)
        if pose_label and best_face is not None:
            pose_color = {"Frontal": (0, 220, 0),
                          "Side":    (0, 180, 255),
                          "Extreme": (0, 80, 255)}.get(pose_label, (150, 150, 150))
            cv2.putText(frame, f"Pose: {pose_label}",
                        (W - 175, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, pose_color, 2)

        # Per-phase mini progress
        phase_start = cumulative_targets[phase_idx - 1] if phase_idx > 0 else 0
        phase_done  = max(0, captured - phase_start)
        phase_total = max(1, cumulative_targets[phase_idx] - phase_start)
        cv2.putText(frame,
                    f"Phase {phase_idx+1}/{n_phases}: {phase_done}/{phase_total} samples",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

        # Bottom instruction banner (semi-transparent black strip)
        banner_h = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, H - banner_h), (W, H), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

        instr_color = (0, 255, 120) if not is_paused else (0, 140, 255)
        if is_paused:
            countdown = max(0, int(phase_pause_until - current_time) + 1)
            instr_prefix = f"Get into position - starting in {countdown}..."
        else:
            instr_prefix = f"Task {phase_idx+1}/{n_phases}:"
        cv2.putText(frame,
                    f"{instr_prefix} {phase_label}",
                    (10, H - banner_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, instr_color, 2)

        # Phase checklist  (bottom strip, right-aligned)
        checklist_y = H - banner_h + 54
        for i in range(n_phases):
            if i < phase_idx or (i == phase_idx and captured >= cumulative_targets[i]):
                marker, col = "[V]", (0, 220, 0)
            elif i == phase_idx:
                marker, col = "[>]", (0, 200, 255)
            else:
                marker, col = "[ ]", (90, 90, 90)
            cv2.putText(frame, f"{marker}{i+1}",
                        (8 + i * 58, checklist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        cv2.imshow("Driver Registration", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured >= max(10, TARGET_SAMPLES // 3):
        total_files = captured + captured * 6  # originals + 6 augmented per original
        print(f"\n[OK] Registration complete for '{name}':")
        print(f"     {captured} original samples + {captured * 6} augmented = {total_files} total images saved.")
        print("     Run recognition to test. Re-register if recognition is still poor.")
    else:
        print(f"\n[!!] Registration incomplete for '{name}': only {captured} samples.")
        print("     Improve lighting and follow each pose instruction carefully.")