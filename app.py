import cv2
import mediapipe as mp
import math
import numpy as np
import time

# ── MediaPipe setup ──────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker    = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ── Constants ────────────────────────────────────────────────────────────────
BLUE_AURA   = (255, 180,  60)   # BGR  idle aura
GOLD_AURA   = ( 30, 200, 255)   # BGR  attack aura
KI_COLOR    = (255, 230, 100)   # BGR  ki blast core
SPIRIT_COLOR= (180, 255, 255)   # BGR  spirit bomb

SMOOTH      = 0.45              # landmark smoothing factor
KI_COOLDOWN = 0.5               # seconds between ki blasts
KI_SPEED    = 22                # pixels per frame
KI_RADIUS   = 18                # pixels
SPIRIT_MAX  = 130               # max spirit bomb radius
SPIRIT_GROW = 0.4               # radius growth per frame


# ════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ════════════════════════════════════════════════════════════════════════════
def angle_3d(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def lm_px(lm, w, h):
    """Landmark → pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def looped_frame(cap):
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame if ret else None


# ════════════════════════════════════════════════════════════════════════════
# Drawing / effect helpers
# ════════════════════════════════════════════════════════════════════════════
def draw_glow_circle(canvas, cx, cy, radius, color, layers=6, base_alpha=0.55):
    """Additive glow: multiple translucent circles, largest → most transparent."""
    overlay = canvas.copy()
    for i in range(layers, 0, -1):
        r   = int(radius * i / layers * 1.8)
        alpha = base_alpha * (1 - i / (layers + 1))
        cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        overlay = canvas.copy()
    # bright core
    cv2.circle(canvas, (cx, cy), max(4, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)


def draw_aura(canvas, landmarks, w, h, color, intensity=1.0):
    """
    Draws a body-hugging aura using torso + limb landmarks.
    Uses addWeighted layered ellipses around key joints.
    """
    joints = [11, 12, 23, 24, 13, 14, 25, 26]   # shoulders, hips, elbows, knees
    overlay = np.zeros_like(canvas)

    for idx in joints:
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            continue
        px, py = lm_px(lm, w, h)
        r = int(h * 0.09 * intensity)
        for layer in range(5, 0, -1):
            alpha = 0.06 * layer * intensity
            rad   = r * layer // 3
            cv2.circle(overlay, (px, py), rad, color, -1, cv2.LINE_AA)

    # torso ellipse
    ls = landmarks[11]; rs = landmarks[12]
    lh = landmarks[23]; rh = landmarks[24]
    if all(l.visibility > 0.4 for l in [ls, rs, lh, rh]):
        cx = int(((ls.x + rs.x) / 2) * w)
        cy = int(((ls.y + rh.y) / 2) * h)
        ew = int(abs(ls.x - rs.x) * w * 1.2)
        eh = int(abs(ls.y - rh.y) * h * 1.3)
        for layer in range(5, 0, -1):
            alpha = 0.05 * layer * intensity
            cv2.ellipse(overlay, (cx, cy),
                        (ew * layer // 3, eh * layer // 3),
                        0, 0, 360, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.55, canvas, 1.0, 0, canvas)


def overlay_video_frame(canvas, vframe, cx, cy, target_w, target_h):
    """Blend a video frame (additive) centred at (cx, cy)."""
    if vframe is None:
        return
    H, W = canvas.shape[:2]
    vf = cv2.resize(vframe, (target_w, target_h))
    x1 = max(0, cx - target_w // 2);  y1 = max(0, cy - target_h // 2)
    x2 = min(W, cx + target_w // 2);  y2 = min(H, cy + target_h // 2)
    ex1 = target_w // 2 - (cx - x1);  ey1 = target_h // 2 - (cy - y1)
    ex2 = target_w // 2 + (x2 - cx);  ey2 = target_h // 2 + (y2 - cy)
    if x2 > x1 and y2 > y1:
        roi        = canvas[y1:y2, x1:x2]
        effect_roi = vf[ey1:ey2, ex1:ex2]
        canvas[y1:y2, x1:x2] = cv2.add(roi, effect_roi)


# ════════════════════════════════════════════════════════════════════════════
# Ki blast dataclass
# ════════════════════════════════════════════════════════════════════════════
class KiBlast:
    def __init__(self, x, y, dx, dy):
        self.x  = float(x)
        self.y  = float(y)
        self.dx = dx   # normalised direction * speed
        self.dy = dy
        self.alive = True
        self.age   = 0

    def update(self, W, H):
        self.x  += self.dx
        self.y  += self.dy
        self.age += 1
        if self.x < -50 or self.x > W + 50 or self.y < -50 or self.y > H + 50:
            self.alive = False

    def draw(self, canvas):
        cx, cy = int(self.x), int(self.y)
        draw_glow_circle(canvas, cx, cy, KI_RADIUS, KI_COLOR, layers=5, base_alpha=0.5)


# ════════════════════════════════════════════════════════════════════════════
# Spirit bomb state
# ════════════════════════════════════════════════════════════════════════════
class SpiritBomb:
    def __init__(self, cx, cy):
        self.cx     = float(cx)
        self.cy     = float(cy)
        self.radius = 20.0
        self.state  = "GROWING"   # GROWING → DROPPING → DONE
        self.vy     = 0.0
        self.flash  = 0           # flash frames remaining

    def update(self, H):
        if self.state == "GROWING":
            self.radius = min(self.radius + SPIRIT_GROW, SPIRIT_MAX)
        elif self.state == "DROPPING":
            self.vy  += 2.5        # gravity
            self.cy  += self.vy
            if self.cy > H + self.radius:
                self.state = "DONE"
                self.flash = 8

    def release(self):
        if self.state == "GROWING":
            self.state = "DROPPING"
            self.vy    = 4.0

    def draw(self, canvas):
        if self.state == "DONE":
            return
        cx, cy = int(self.cx), int(self.cy)
        draw_glow_circle(canvas, cx, cy, int(self.radius), SPIRIT_COLOR, layers=7, base_alpha=0.5)
        # rotating energy ring
        t   = time.time()
        for i in range(8):
            angle  = t * 120 + i * 45          # degrees, rotates over time
            rad_a  = math.radians(angle)
            rx     = int(cx + self.radius * 0.85 * math.cos(rad_a))
            ry     = int(cy + self.radius * 0.85 * math.sin(rad_a))
            cv2.circle(canvas, (rx, ry), 4, (255, 255, 255), -1, cv2.LINE_AA)

    @property
    def done(self):
        return self.state == "DONE" and self.flash == 0


# ════════════════════════════════════════════════════════════════════════════
# Gesture detection
# ════════════════════════════════════════════════════════════════════════════
def detect_gesture(lms, prev_state, w, h):
    """
    Returns (state, dominant_wrist_px, forearm_dir_px)
    state ∈ {IDLE, CHARGING, KI_BLAST, KAMEHAMEHA, SPIRIT_RELEASE}
    """
    ls  = lms[11]; rs  = lms[12]
    le  = lms[13]; re  = lms[14]
    lw  = lms[15]; rw  = lms[16]

    vis_l = lw.visibility > 0.4
    vis_r = rw.visibility > 0.4
    if not (vis_l or vis_r):
        return "IDLE", (-1, -1), (0, 0)

    wrist_dist = math.hypot(lw.x - rw.x, lw.y - rw.y)

    # ── KAMEHAMEHA: wrists very close + both arms extended forward ──────────
    kame_thresh = 0.45 if prev_state == "KAMEHAMEHA" else 0.30
    if wrist_dist < kame_thresh:
        la = angle_3d(ls, le, lw)
        ra = angle_3d(rs, re, rw)
        ang_thresh = 120 if prev_state == "KAMEHAMEHA" else 135
        if max(la, ra) > ang_thresh:
            wx = int(((lw.x + rw.x) / 2) * w)
            wy = int(((lw.y + rw.y) / 2) * h)
            return "KAMEHAMEHA", (wx, wy), (1, 0)

    # ── CHARGING (spirit bomb): wrists close + hands raised above shoulders ─
    charge_thresh = 0.40 if prev_state == "CHARGING" else 0.28
    if wrist_dist < charge_thresh:
        avg_wrist_y = (lw.y + rw.y) / 2
        avg_shoulder_y = (ls.y + rs.y) / 2
        if avg_wrist_y < avg_shoulder_y - 0.05:      # hands above shoulders
            wx = int(((lw.x + rw.x) / 2) * w)
            wy = int(((lw.y + rw.y) / 2) * h)
            return "CHARGING", (wx, wy), (0, -1)

    # ── SPIRIT_RELEASE: was charging, wrists now far apart + arms dropped ───
    if prev_state == "CHARGING" and wrist_dist > 0.45:
        return "SPIRIT_RELEASE", (-1, -1), (0, 0)

    # ── KI_BLAST: one arm extended (elbow angle > 155°), other relaxed ──────
    la = angle_3d(ls, le, lw) if vis_l else 0
    ra = angle_3d(rs, re, rw) if vis_r else 0
    KI_THRESH = 155

    dominant, elbow, wrist = None, None, None
    if la > KI_THRESH and ra < 110:
        dominant, elbow, wrist = "left",  le, lw
    elif ra > KI_THRESH and la < 110:
        dominant, elbow, wrist = "right", re, rw

    if dominant:
        wx, wy = lm_px(wrist, w, h)
        ex, ey = lm_px(elbow, w, h)
        dx, dy = wx - ex, wy - ey
        length = math.hypot(dx, dy) or 1
        dir_x  = dx / length * KI_SPEED
        dir_y  = dy / length * KI_SPEED
        return "KI_BLAST", (wx, wy), (dir_x, dir_y)

    return "IDLE", (-1, -1), (0, 0)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    cap       = cv2.VideoCapture(0)
    cap_kame  = cv2.VideoCapture('assets/kamehameha.mp4')
    cap_energy= cv2.VideoCapture('assets/energy.mp4')

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )

    landmarker  = PoseLandmarker.create_from_options(options)
    prev_state  = "IDLE"
    frame_idx   = 0

    # smoothed hand position
    smooth_hx, smooth_hy = -1, -1

    # ki blasts alive on screen
    ki_blasts: list[KiBlast] = []
    last_ki_time = 0.0

    # spirit bomb (only one at a time)
    spirit: SpiritBomb | None = None
    spirit_flash = 0            # global screen flash frames

    # idle frame buffer (avoids flickering on lost detection)
    idle_buf = 0
    IDLE_BUF_MAX = 6

    with landmarker as pose:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                break

            image = cv2.flip(image, 1)
            H, W  = image.shape[:2]

            # slight darken for contrast
            image = cv2.convertScaleAbs(image, alpha=0.6, beta=0)

            rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_idx += 1
            result = pose.detect_for_video(mp_img, frame_idx * 33)

            # ── Gesture detection ────────────────────────────────────────
            current_state = "IDLE"
            hand_px       = (-1, -1)
            dir_vec       = (0, 0)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                current_state, hand_px, dir_vec = detect_gesture(
                    lms, prev_state, W, H)

                # idle buffer: don't snap to IDLE immediately
                if current_state == "IDLE" and prev_state != "IDLE":
                    idle_buf += 1
                    if idle_buf < IDLE_BUF_MAX:
                        current_state = prev_state
                        hand_px       = (smooth_hx, smooth_hy)
                else:
                    idle_buf = 0
            else:
                idle_buf += 1
                if idle_buf < IDLE_BUF_MAX and prev_state != "IDLE":
                    current_state = prev_state
                    hand_px       = (smooth_hx, smooth_hy)

            # ── Smooth hand position ─────────────────────────────────────
            hx, hy = hand_px
            if hx != -1:
                if smooth_hx == -1:
                    smooth_hx, smooth_hy = hx, hy
                else:
                    smooth_hx = int((1 - SMOOTH) * smooth_hx + SMOOTH * hx)
                    smooth_hy = int((1 - SMOOTH) * smooth_hy + SMOOTH * hy)
            elif current_state == "IDLE":
                smooth_hx, smooth_hy = -1, -1

            # ── State transitions ────────────────────────────────────────
            if current_state == "KAMEHAMEHA" and prev_state != "KAMEHAMEHA":
                cap_kame.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if current_state == "CHARGING" and prev_state != "CHARGING":
                # start a new spirit bomb above head
                sbx = smooth_hx if smooth_hx != -1 else W // 2
                sby = smooth_hy if smooth_hy != -1 else H // 4
                spirit = SpiritBomb(sbx, sby)

            if current_state == "SPIRIT_RELEASE" and spirit and spirit.state == "GROWING":
                spirit.release()
                spirit_flash = 10

            # ── Aura color ───────────────────────────────────────────────
            attacking = current_state in ("KI_BLAST", "KAMEHAMEHA", "SPIRIT_RELEASE")
            aura_color = GOLD_AURA if attacking else BLUE_AURA
            aura_intensity = 1.3 if attacking else 0.85

            # ── Draw aura (always on) ────────────────────────────────────
            if result.pose_landmarks:
                draw_aura(image, result.pose_landmarks[0], W, H,
                          aura_color, intensity=aura_intensity)

            # ── Kamehameha beam ──────────────────────────────────────────
            if current_state == "KAMEHAMEHA" and smooth_hx != -1:
                vf = looped_frame(cap_kame)
                if vf is not None:
                    oh, ow = vf.shape[:2]
                    bh = int(H * 0.72)
                    bw = int(bh * ow / oh)
                    overlay_video_frame(image, vf,
                                        smooth_hx + bw // 4, smooth_hy,
                                        bw, bh)

            # ── Charging (energy effect around spirit bomb position) ──────
            if current_state == "CHARGING" and spirit:
                vf = looped_frame(cap_energy)
                if vf is not None:
                    size = int(spirit.radius * 2.5)
                    overlay_video_frame(image, vf,
                                        int(spirit.cx), int(spirit.cy),
                                        size, size)

            # ── Spirit bomb update & draw ────────────────────────────────
            if spirit:
                if current_state == "CHARGING" and spirit.state == "GROWING":
                    spirit.cx = float(smooth_hx) if smooth_hx != -1 else spirit.cx
                    spirit.cy = float(smooth_hy) if smooth_hy != -1 else spirit.cy
                spirit.update(H)
                spirit.draw(image)
                if spirit.done:
                    spirit_flash = max(spirit_flash, spirit.flash)
                    spirit = None

            # ── Spirit bomb screen flash ─────────────────────────────────
            if spirit_flash > 0:
                alpha = spirit_flash / 10
                white = np.full_like(image, 255)
                cv2.addWeighted(white, alpha * 0.6, image, 1 - alpha * 0.6, 0, image)
                spirit_flash -= 1

            # ── Ki blasts ────────────────────────────────────────────────
            now = time.time()
            if (current_state == "KI_BLAST"
                    and smooth_hx != -1
                    and now - last_ki_time > KI_COOLDOWN):
                dx, dy = dir_vec
                ki_blasts.append(KiBlast(smooth_hx, smooth_hy, dx, dy))
                last_ki_time = now

            for kb in ki_blasts:
                kb.update(W, H)
                kb.draw(image)
            ki_blasts = [kb for kb in ki_blasts if kb.alive]

            # ── HUD: state label ─────────────────────────────────────────
            label_color = {
                "IDLE":           (180, 180, 180),
                "CHARGING":       (100, 255, 255),
                "KI_BLAST":       ( 60, 220, 255),
                "KAMEHAMEHA":     ( 30, 200, 255),
                "SPIRIT_RELEASE": (255, 255, 255),
            }.get(current_state, (255, 255, 255))

            cv2.putText(image, current_state, (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2, cv2.LINE_AA)

            prev_state = current_state

            cv2.imshow('Dragon Ball Z', image)
            if cv2.waitKey(5) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cap_kame.release()
    cap_energy.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
