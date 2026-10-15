"""
================================================================================
DRIVEGUARD AI - ADVANCED SCIENTIFIC DRIVER DROWSINESS & DISTRACTION MONITORING
================================================================================

Based on State-of-the-Art research from Google Scholar, IEEE, and NHTSA:
1. PERCLOS (Percentage of Eye Closure over Time - NHTSA / Wierwille standard):
   The gold-standard metric for slow eyelid closure and drowsiness assessment.
2. Multi-Modal Fatigue Fusion Index (0 - 100%):
   Fuses PERCLOS, average blink duration, yawn frequency, head nodding / droop,
   and acute closure duration to eliminate false positives from single thresholds.
3. 3D Head Pose Estimation & Head Nodding Detection:
   Solves Perspective-n-Point (SolvePnP) to track Pitch, Yaw, and Roll Euler angles.
   Detects cervical muscle tone loss (head droop / micro-sleep nod).
4. Sub-Pupillary Iris Tracking & Gaze Classification:
   Tracks iris center relative to eye corners to measure off-road gaze deviation.
5. Personalized Startup Baseline Calibration:
   5-second calibration window computing robust median baselines for individual
   driver geometry, ambient lighting, and facial anatomy.
6. 4-Tier Risk Hierarchy & Intelligent Alerts:
   NORMAL -> ADVISORY (Mild Fatigue) -> WARNING -> CRITICAL (Micro-sleep/Buzzer).
7. Non-blocking Threaded Audio Alarms & Diagnostic CSV Telemetry Logging.
================================================================================
"""

import argparse
import csv
from collections import deque
import os
import sys
from threading import Lock, Thread
import time

# Auto-detect virtual environment if executed with bare system Python
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from playsound3 import playsound
except ImportError:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_dir, "venv", "bin", "python")
    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"-> [DriveGuard AI]: Switching from '{sys.executable}' to project venv: {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("\n" + "=" * 65)
        print("[ERROR]: Missing required libraries (cv2, mediapipe, numpy, playsound3)!")
        print("Please activate the virtual environment:")
        print("    source venv/bin/activate")
        print("    python drowsiness_yawn.py")
        print("\nOr install them via:")
        print("    pip install -r requirements.txt")
        print("=" * 65 + "\n")
        sys.exit(1)

from feature_logger import FeatureLogger, SUPPORTED_ACTIVITIES


# ==============================================================================
# CONFIGURATION CONSTANTS & SCIENTIFIC THRESHOLDS
# ==============================================================================

# Baseline calibration window in seconds
CALIBRATION_SECONDS = 5.0

# Eye Aspect Ratio (EAR) adaptation parameters
EAR_THRESHOLD_FACTOR = 0.72   # Eye is closed when EAR < 72% of baseline
MIN_EAR_THRESHOLD = 0.12      # Absolute safety lower bound
MAX_EAR_THRESHOLD = 0.28      # Absolute safety upper bound

# Eye closure duration triggers (seconds)
MIN_BLINK_DURATION = 0.08     # Below this is noise/glitch
MAX_BLINK_DURATION = 0.40     # Standard normal blink upper bound
PROLONGED_BLINK_THRESHOLD = 0.45  # Blinks > 0.45s indicate fatigue onset
EYE_CLOSED_ALERT_SECONDS = 1.50   # Immediate critical micro-sleep alarm

# PERCLOS parameters (NHTSA / IEEE standard: 60-second observation window)
PERCLOS_WINDOW_SECONDS = 60.0
PERCLOS_MILD_THRESHOLD = 0.12    # 12% closure indicates mild fatigue
PERCLOS_HIGH_THRESHOLD = 0.20    # 20% closure indicates high fatigue
PERCLOS_CRITICAL_THRESHOLD = 0.30  # 30% closure is severe danger

# Mouth Aspect Ratio (MAR) adaptation parameters
MAR_THRESHOLD_OFFSET = 0.24   # Yawn triggered when MAR exceeds baseline + 0.24
MIN_MAR_THRESHOLD = 0.30      # Lower clamp bound
MAX_MAR_THRESHOLD = 0.65      # Upper clamp bound
MIN_YAWN_DURATION = 0.80      # Sustained mouth opening duration for yawn count
YAWN_ALERT_SECONDS = 1.20     # Sustained open mouth triggering yawn warning

# Head Pose & Gaze thresholds (degrees and normalized ratios)
HEAD_YAW_THRESHOLD = 25.0     # Degrees left/right off-road
HEAD_PITCH_UP_THRESHOLD = 20.0
HEAD_PITCH_DOWN_THRESHOLD = 18.0  # Downward pitch threshold for nodding/distraction
HEAD_ROLL_THRESHOLD = 25.0
HEAD_NOD_DURATION_MIN = 0.50  # Head droop held for > 0.5s is recorded as a nod

GAZE_HORIZONTAL_DEVIATION = 0.14
GAZE_VERTICAL_DEVIATION = 0.16
LOOK_AWAY_ALERT_SECONDS = 2.0  # Sustained off-road distraction threshold

# Composite Fatigue Index weights (sum = 1.0)
WEIGHT_PERCLOS = 0.35
WEIGHT_BLINK_DURATION = 0.20
WEIGHT_YAWN = 0.20
WEIGHT_HEAD_NOD = 0.15
WEIGHT_INSTANT_CLOSURE = 0.10

# Risk Level Classification Thresholds (0 - 100)
FATIGUE_ADVISORY = 35.0
FATIGUE_WARNING = 60.0
FATIGUE_CRITICAL = 80.0

DISTRACTION_WARNING = 50.0
DISTRACTION_CRITICAL = 80.0


# ==============================================================================
# MEDIAPIPE FACEMESH LANDMARK INDICES
# ==============================================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP_INNER = 82
MOUTH_BOTTOM_INNER = 87

# 3D Head Pose Anthropometric Landmarks
NOSE_TIP = 1
CHIN = 152
SCREEN_LEFT_EYE = 33      # Landmark 33: Screen-left eye corner
SCREEN_RIGHT_EYE = 263    # Landmark 263: Screen-right eye corner
SCREEN_LEFT_MOUTH = 61    # Landmark 61: Screen-left mouth corner
SCREEN_RIGHT_MOUTH = 291  # Landmark 291: Screen-right mouth corner


# ==============================================================================
# ALARM & AUDIO SUBSYSTEM
# ==============================================================================

alarm_lock = Lock()
alarm_playing = False


def _play_alarm_worker(audio_path: str):
    """Worker thread target to execute audio playback without blocking video."""
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            return
        alarm_playing = True

    try:
        if os.path.exists(audio_path):
            playsound(audio_path)
    except Exception as error:
        print(f"[DriveGuard Alert Error]: {error}")
    finally:
        with alarm_lock:
            alarm_playing = False


def trigger_alarm(audio_path: str):
    """Spawns an asynchronous audio playback thread if not already playing."""
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            return

    thread = Thread(target=_play_alarm_worker, args=(audio_path,), daemon=True)
    thread.start()


# ==============================================================================
# MATHEMATICAL & GEOMETRIC UTILITIES
# ==============================================================================

def calculate_distance(p1, p2) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def landmark_to_point(landmark, width: int, height: int):
    """Converts normalized MediaPipe landmark coordinates to pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Restricts value within [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def angle_difference(current_angle: float, baseline_angle: float) -> float:
    """Computes the shortest angular difference handling wrap-around at +/-180 deg."""
    return (current_angle - baseline_angle + 180.0) % 360.0 - 180.0


# ==============================================================================
# FEATURE EXTRACTION: EAR, MAR, GAZE & HEAD POSE
# ==============================================================================

def calculate_ear(landmarks, eye_indices, width: int, height: int):
    """
    Computes Eye Aspect Ratio (EAR) based on Soukupová and Čech (2016).
    EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
    """
    points = [landmark_to_point(landmarks[idx], width, height) for idx in eye_indices]
    vert_1 = calculate_distance(points[1], points[5])
    vert_2 = calculate_distance(points[2], points[4])
    horiz = calculate_distance(points[0], points[3])

    if horiz <= 0:
        return 0.0, points

    ear = (vert_1 + vert_2) / (2.0 * horiz)
    return float(ear), points


def calculate_mar(landmarks, width: int, height: int):
    """
    Computes Mouth Aspect Ratio (MAR) to evaluate oral aperture for yawning.
    """
    top1 = landmark_to_point(landmarks[MOUTH_TOP], width, height)
    bot1 = landmark_to_point(landmarks[MOUTH_BOTTOM], width, height)
    top2 = landmark_to_point(landmarks[MOUTH_TOP_INNER], width, height)
    bot2 = landmark_to_point(landmarks[MOUTH_BOTTOM_INNER], width, height)
    left = landmark_to_point(landmarks[MOUTH_LEFT], width, height)
    right = landmark_to_point(landmarks[MOUTH_RIGHT], width, height)

    vert_1 = calculate_distance(top1, bot1)
    vert_2 = calculate_distance(top2, bot2)
    horiz = calculate_distance(left, right)

    points = [left, top2, top1, right, bot2, bot1]
    if horiz <= 0:
        return 0.0, points

    mar = (vert_1 + vert_2) / (2.0 * horiz)
    return float(mar), points


def calculate_iris_center(landmarks, iris_indices, width: int, height: int):
    """Averages coordinates of iris contour landmarks to obtain the iris center."""
    points = [landmark_to_point(landmarks[idx], width, height) for idx in iris_indices]
    cx = int(sum(p[0] for p in points) / len(points))
    cy = int(sum(p[1] for p in points) / len(points))
    return (cx, cy), points


def calculate_eye_gaze_ratio(landmarks, iris_indices, outer_idx, inner_idx, top_idx, bot_idx, width: int, height: int):
    """
    Computes relative horizontal and vertical gaze ratios [0.0 - 1.0]
    of the iris within the eye boundary box.
    """
    iris_center, iris_points = calculate_iris_center(landmarks, iris_indices, width, height)
    outer = landmark_to_point(landmarks[outer_idx], width, height)
    inner = landmark_to_point(landmarks[inner_idx], width, height)
    top = landmark_to_point(landmarks[top_idx], width, height)
    bot = landmark_to_point(landmarks[bot_idx], width, height)

    eye_left_x = min(outer[0], inner[0])
    eye_right_x = max(outer[0], inner[0])
    eye_top_y = min(top[1], bot[1])
    eye_bottom_y = max(top[1], bot[1])

    eye_w = eye_right_x - eye_left_x
    eye_h = eye_bottom_y - eye_top_y

    h_ratio = 0.5 if eye_w <= 0 else (iris_center[0] - eye_left_x) / eye_w
    v_ratio = 0.5 if eye_h <= 0 else (iris_center[1] - eye_top_y) / eye_h

    h_ratio = float(clamp(h_ratio, 0.0, 1.0))
    v_ratio = float(clamp(v_ratio, 0.0, 1.0))
    return h_ratio, v_ratio, iris_center, iris_points


def calculate_raw_gaze(landmarks, width: int, height: int):
    """Combines left and right eye gaze measurements."""
    l_h, l_v, l_center, l_pts = calculate_eye_gaze_ratio(
        landmarks, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, width, height
    )
    r_h, r_v, r_center, r_pts = calculate_eye_gaze_ratio(
        landmarks, RIGHT_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, width, height
    )

    return {
        "horizontal_ratio": float((l_h + r_h) / 2.0),
        "vertical_ratio": float((l_v + r_v) / 2.0),
        "left_center": l_center,
        "right_center": r_center,
        "left_points": l_pts,
        "right_points": r_pts,
    }


def classify_gaze(gaze_data: dict, baseline_x: float, baseline_y: float):
    """Classifies gaze direction relative to calibrated forward baseline."""
    delta_x = gaze_data["horizontal_ratio"] - baseline_x
    delta_y = gaze_data["vertical_ratio"] - baseline_y

    if delta_y > GAZE_VERTICAL_DEVIATION:
        direction = "DOWN"
    elif delta_y < -GAZE_VERTICAL_DEVIATION:
        direction = "UP"
    elif delta_x > GAZE_HORIZONTAL_DEVIATION:
        direction = "RIGHT"
    elif delta_x < -GAZE_HORIZONTAL_DEVIATION:
        direction = "LEFT"
    else:
        direction = "CENTER"

    return direction, delta_x, delta_y


def estimate_raw_head_pose(landmarks, width: int, height: int):
    """
    Computes 3D head pose orientation using OpenCV SolvePnP (Iterative Levenberg-Marquardt).
    Maps 6 facial landmarks to an anthropometric 3D face model in OpenCV camera coordinates:
    +X: Screen Right | +Y: Screen Down | +Z: Forward into camera
    """
    image_points = np.array([
        landmark_to_point(landmarks[NOSE_TIP], width, height),
        landmark_to_point(landmarks[CHIN], width, height),
        landmark_to_point(landmarks[SCREEN_LEFT_EYE], width, height),
        landmark_to_point(landmarks[SCREEN_RIGHT_EYE], width, height),
        landmark_to_point(landmarks[SCREEN_LEFT_MOUTH], width, height),
        landmark_to_point(landmarks[SCREEN_RIGHT_MOUTH], width, height),
    ], dtype=np.float64)

    # 3D generic facial model points in millimeters
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (center)
        (0.0, 330.0, -65.0),      # Chin (below nose: +Y)
        (-225.0, -170.0, -135.0), # Screen left eye (left of nose: -X, above nose: -Y)
        (225.0, -170.0, -135.0),  # Screen right eye (right of nose: +X, above nose: -Y)
        (-150.0, 150.0, -125.0),  # Screen left mouth (left of nose: -X, below nose: +Y)
        (150.0, 150.0, -125.0),   # Screen right mouth (right of nose: +X, below nose: +Y)
    ], dtype=np.float64)

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    # Convention: Looking down is (+), looking up is (-), looking right is (+), looking left is (-)
    pitch = -float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "nose_point": (int(image_points[0][0]), int(image_points[0][1])),
    }


def classify_head_pose(raw_head_pose: dict, base_pitch: float, base_yaw: float, base_roll: float):
    """Calibrated deviation angles and primary head direction."""
    c_pitch = angle_difference(raw_head_pose["pitch"], base_pitch)
    c_yaw = angle_difference(raw_head_pose["yaw"], base_yaw)
    c_roll = angle_difference(raw_head_pose["roll"], base_roll)

    if c_yaw > HEAD_YAW_THRESHOLD:
        direction = "RIGHT"
    elif c_yaw < -HEAD_YAW_THRESHOLD:
        direction = "LEFT"
    elif c_pitch > HEAD_PITCH_DOWN_THRESHOLD:
        direction = "DOWN"
    elif c_pitch < -HEAD_PITCH_UP_THRESHOLD:
        direction = "UP"
    else:
        direction = "FORWARD"

    return {"pitch": c_pitch, "yaw": c_yaw, "roll": c_roll, "direction": direction}


# ==============================================================================
# SCIENTIFIC METRIC ENGINES
# ==============================================================================

class PERCLOSEngine:
    """
    Implements NHTSA / Wierwille PERCLOS (Percentage of Eye Closure).
    Maintains a rolling time-window buffer of eye states:
    PERCLOS = (Duration Eyes Closed in Window / Total Window Duration) * 100%
    """

    def __init__(self, window_seconds: float = PERCLOS_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        # Deque of (timestamp, is_closed) tuples
        self.history = deque()

    def update(self, current_time: float, is_closed: bool) -> float:
        self.history.append((current_time, is_closed))

        # Evict samples older than observation window
        cutoff = current_time - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if len(self.history) < 2:
            return 0.0

        total_duration = self.history[-1][0] - self.history[0][0]
        if total_duration <= 0.1:
            return 0.0

        # Calculate time-weighted closure
        closed_duration = 0.0
        for i in range(len(self.history) - 1):
            t_curr, closed_state = self.history[i]
            t_next, _ = self.history[i + 1]
            dt = t_next - t_curr
            if closed_state:
                closed_duration += dt

        perclos = closed_duration / total_duration
        return float(clamp(perclos, 0.0, 1.0))


class HeadNodDetector:
    """
    Detects downward head drooping/nodding associated with cervical muscle tone loss
    during micro-sleep episodes.
    """

    def __init__(self, pitch_threshold: float = HEAD_PITCH_DOWN_THRESHOLD):
        self.pitch_threshold = pitch_threshold
        self.nod_start_time = None
        self.total_nods = 0
        self.nodding_active = False

    def update(self, current_time: float, pitch_deviation: float, eyes_closed: bool) -> bool:
        """
        Returns True when a confirmed head nod event is identified.
        A nod is detected if head drops downward (> threshold) for >= 0.5s,
        or downward pitch co-occurs with closed eyes.
        """
        is_drooping = (pitch_deviation > self.pitch_threshold)

        if is_drooping:
            if self.nod_start_time is None:
                self.nod_start_time = current_time

            duration = current_time - self.nod_start_time
            # Immediate nod trigger if eyes closed while head droops, or sustained droop
            if (eyes_closed and duration >= 0.35) or (duration >= HEAD_NOD_DURATION_MIN):
                if not self.nodding_active:
                    self.nodding_active = True
                    self.total_nods += 1
                    return True
        else:
            self.nod_start_time = None
            self.nodding_active = False

        return False


class FatigueScoreEngine:
    """
    Multi-Modal Fatigue Fusion Model (0 - 100%).
    Fuses PERCLOS, average blink duration, yawn frequency, head nodding, and acute closure.
    """

    @staticmethod
    def calculate(
        perclos: float,
        avg_blink_duration: float,
        recent_yawns_in_window: int,
        head_nod_count: int,
        current_eye_closed_duration: float,
    ) -> float:
        # 1. PERCLOS Component (0 - 100): 0% at 0.0, 100% at 0.30+
        perclos_score = clamp(perclos / PERCLOS_CRITICAL_THRESHOLD, 0.0, 1.0) * 100.0

        # 2. Blink Duration Component (0 - 100): normal is 0.20s; fatigued is > 0.40s
        if avg_blink_duration <= MAX_BLINK_DURATION:
            blink_score = clamp((avg_blink_duration - MIN_BLINK_DURATION) / (MAX_BLINK_DURATION - MIN_BLINK_DURATION), 0.0, 1.0) * 40.0
        else:
            blink_score = 40.0 + clamp((avg_blink_duration - MAX_BLINK_DURATION) / 0.30, 0.0, 1.0) * 60.0

        # 3. Yawn Frequency Component (0 - 100): 1 yawn = 35%, 2 yawns = 70%, 3+ = 100%
        yawn_score = clamp(recent_yawns_in_window * 35.0, 0.0, 100.0)

        # 4. Head Nod Component (0 - 100): each nod is a strong micro-sleep signal
        nod_score = clamp(head_nod_count * 50.0, 0.0, 100.0)

        # 5. Acute Instantaneous Closure Component (0 - 100)
        instant_score = clamp(current_eye_closed_duration / EYE_CLOSED_ALERT_SECONDS, 0.0, 1.0) * 100.0

        composite = (
            WEIGHT_PERCLOS * perclos_score
            + WEIGHT_BLINK_DURATION * blink_score
            + WEIGHT_YAWN * yawn_score
            + WEIGHT_HEAD_NOD * nod_score
            + WEIGHT_INSTANT_CLOSURE * instant_score
        )
        return float(clamp(composite, 0.0, 100.0))


class DistractionScoreEngine:
    """
    Tracks off-road visual attention (0 - 100%).
    Combines instantaneous look-away duration and rolling off-road percentage over 15 seconds.
    """

    def __init__(self, window_seconds: float = 15.0):
        self.window_seconds = window_seconds
        self.history = deque()

    def update(self, current_time: float, is_looking_away: bool, look_away_duration: float) -> float:
        self.history.append((current_time, is_looking_away))
        cutoff = current_time - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        ratio = 0.0
        if len(self.history) >= 2:
            total_dt = self.history[-1][0] - self.history[0][0]
            if total_dt > 0.1:
                away_dt = sum(
                    (self.history[i + 1][0] - self.history[i][0])
                    for i in range(len(self.history) - 1)
                    if self.history[i][1]
                )
                ratio = away_dt / total_dt

        # Combine rolling ratio (60%) + sustained duration (40%)
        duration_component = clamp(look_away_duration / LOOK_AWAY_ALERT_SECONDS, 0.0, 1.0)
        score = (0.50 * ratio + 0.50 * duration_component) * 100.0
        return float(clamp(score, 0.0, 100.0))


# ==============================================================================
# DRAWING ROUTINES & HEADS-UP DISPLAY (HUD)
# ==============================================================================

def draw_eye_contour(frame, points, color=(0, 255, 0)):
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)


def draw_mouth_contour(frame, points, color=(0, 255, 255)):
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)


def draw_iris_marker(frame, center, color=(255, 0, 255)):
    cv2.circle(frame, center, 3, color, -1)


def draw_hud(
    frame,
    fatigue_score: float,
    distraction_score: float,
    perclos: float,
    ear: float,
    ear_t: float,
    mar: float,
    mar_t: float,
    blink_count: int,
    blink_rate: float,
    avg_blink_dur: float,
    yawn_count: int,
    head_nods: int,
    head_dir: str,
    pitch: float,
    yaw: float,
    gaze_dir: str,
    fps: float,
    active_alerts: list,
    logger_status: dict = None,
):
    """
    Renders a state-of-the-art telemetry Heads-Up Display with status panels,
    progress gauges, live metrics, and data recording status.
    """
    height, width = frame.shape[:2]

    # Semi-transparent top dashboard banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 85), (20, 20, 24), -1)
    # Side panel background for detailed metrics
    cv2.rectangle(overlay, (width - 240, 90), (width - 5, height - 10), (15, 15, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # 1. Primary Status & Risk Level Banner
    if fatigue_score >= FATIGUE_CRITICAL or any("DROWSINESS" in a or "MICROSLEEP" in a for a in active_alerts):
        status_text = "STATUS: CRITICAL DROWSINESS"
        status_color = (0, 0, 255)
    elif distraction_score >= DISTRACTION_CRITICAL or any("DISTRACTION" in a for a in active_alerts):
        status_text = "STATUS: DISTRACTION DANGER"
        status_color = (0, 0, 255)
    elif fatigue_score >= FATIGUE_WARNING or distraction_score >= DISTRACTION_WARNING:
        status_text = "STATUS: WARNING (ELEVATED RISK)"
        status_color = (0, 140, 255)
    elif fatigue_score >= FATIGUE_ADVISORY:
        status_text = "STATUS: ADVISORY (MILD FATIGUE)"
        status_color = (0, 220, 255)
    else:
        status_text = "STATUS: ATTENTIVE / NORMAL"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.68, status_color, 2)

    # Active Alarm Banner Flash
    if active_alerts:
        alert_str = " | ".join(active_alerts)
        cv2.putText(frame, f"! {alert_str} !", (15, 55), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "DriveGuard AI - Multi-Modal Scientific DMS", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # 2. Top-Center Gauges: Fatigue Score & Distraction Score
    # Fatigue Bar
    fatigue_bar_x = 420
    fatigue_bar_w = 120
    fatigue_fill = int((fatigue_score / 100.0) * fatigue_bar_w)
    f_col = (0, 255, 0) if fatigue_score < FATIGUE_ADVISORY else ((0, 165, 255) if fatigue_score < FATIGUE_CRITICAL else (0, 0, 255))
    cv2.putText(frame, f"Fatigue: {fatigue_score:.0f}%", (fatigue_bar_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(frame, (fatigue_bar_x, 28), (fatigue_bar_x + fatigue_bar_w, 40), (60, 60, 60), 1)
    cv2.rectangle(frame, (fatigue_bar_x, 28), (fatigue_bar_x + fatigue_fill, 40), f_col, -1)

    # Distraction Bar
    dist_bar_x = 420
    dist_bar_w = 120
    dist_fill = int((distraction_score / 100.0) * dist_bar_w)
    d_col = (0, 255, 0) if distraction_score < DISTRACTION_WARNING else (0, 0, 255)
    cv2.putText(frame, f"Distraction: {distraction_score:.0f}%", (dist_bar_x, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(frame, (dist_bar_x, 62), (dist_bar_x + dist_bar_w, 74), (60, 60, 60), 1)
    cv2.rectangle(frame, (dist_bar_x, 62), (dist_bar_x + dist_fill, 74), d_col, -1)

    # FPS counter in top right
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 85, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # 3. Right Telemetry Sidebar
    y_start = 112
    y_step = 22
    sidebar_items = [
        ("PERCLOS (60s):", f"{perclos * 100.0:.1f}%", (0, 255, 255) if perclos > 0.15 else (220, 220, 220)),
        ("EAR / Adapt T:", f"{ear:.3f} / {ear_t:.3f}", (255, 255, 0)),
        ("MAR / Adapt T:", f"{mar:.3f} / {mar_t:.3f}", (255, 255, 0)),
        ("Total Blinks:", f"{blink_count}", (255, 255, 255)),
        ("Blink Rate:", f"{blink_rate:.1f} /min", (255, 255, 255)),
        ("Avg Blink Dur:", f"{avg_blink_dur:.2f}s", (0, 165, 255) if avg_blink_dur > 0.35 else (255, 255, 255)),
        ("Yawns Detected:", f"{yawn_count}", (0, 255, 255) if yawn_count > 0 else (255, 255, 255)),
        ("Head Nods:", f"{head_nods}", (0, 0, 255) if head_nods > 0 else (255, 255, 255)),
        ("Head Orientation:", f"{head_dir}", (0, 255, 255)),
        ("Pitch / Yaw:", f"{pitch:+.1f} / {yaw:+.1f}", (200, 200, 200)),
        ("Gaze Direction:", f"{gaze_dir}", (255, 0, 255)),
    ]

    for label, val, col in sidebar_items:
        cv2.putText(frame, label, (width - 230, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        cv2.putText(frame, val, (width - 105, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        y_start += y_step

    # 4. Bottom-Left Recording & Data Collection Panel (GRU Training Mode)
    if logger_status:
        rec_active = logger_status.get("is_recording", False)
        rec_panel_w = 405
        rec_panel_h = 76
        rec_panel_y = height - rec_panel_h - 10

        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (10, rec_panel_y), (10 + rec_panel_w, height - 10), (18, 18, 22), -1)
        cv2.addWeighted(panel_overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (10, rec_panel_y), (10 + rec_panel_w, height - 10), (0, 0, 255) if rec_active else (70, 70, 70), 1)

        if rec_active:
            cv2.circle(frame, (25, rec_panel_y + 16), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC [ACTIVE - 10Hz]", (37, rec_panel_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 0, 255), 1)
            cv2.putText(frame, f"Samples: {logger_status['sample_count']} ({logger_status['duration_s']:.1f}s)", (195, rec_panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (25, rec_panel_y + 16), 5, (120, 120, 120), -1)
            cv2.putText(frame, "REC: STANDBY [Press 'r' to Record]", (37, rec_panel_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (160, 160, 160), 1)

        act_label = logger_status.get("activity_label", "normal_alert")
        p_id = logger_status.get("participant_id", "P001")
        cv2.putText(frame, f"Act: {act_label} [Keys 1-8]", (20, rec_panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        cv2.putText(frame, f"ID: {p_id}", (260, rec_panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        fpath = logger_status.get("filepath")
        if fpath:
            fname = os.path.basename(fpath)
            cv2.putText(frame, f"File: {fname}", (20, rec_panel_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        else:
            cv2.putText(frame, "Controls: [r] Rec | [1-8] Activity | [p] Participant | [q] Quit", (20, rec_panel_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)


# ==============================================================================
# MAIN APPLICATION ENGINE
# ==============================================================================

def run_driveguard(
    webcam_idx: int = 0,
    video_path: str = None,
    alarm_path: str = "Alert.wav",
    log_csv: str = None,
    calibration_duration: float = CALIBRATION_SECONDS,
    alarm_distraction: bool = True,
    no_gui: bool = False,
    record: bool = False,
    participant_id: str = "P001",
    session_id: str = None,
    activity_label: str = "normal_alert",
    sampling_rate: float = 10.0,
    data_dir: str = "data/raw",
    fatigue_label: str = "unknown",
    distraction_label: str = "unknown",
):
    """Initializes and runs the DriveGuard AI driver monitoring system."""
    if not os.path.exists(alarm_path):
        print(f"[ERROR]: Alarm audio file not found at: {os.path.abspath(alarm_path)}")
        raise SystemExit(1)

    print("\n========================================================")
    print("      DRIVEGUARD AI - SCIENTIFIC DMS INITIALIZING      ")
    print("========================================================")
    print(f"-> Audio Alarm Resource: {os.path.abspath(alarm_path)}")

    # Behavioural Feature Logger (GRU Temporal Training)
    feature_logger = FeatureLogger(
        data_dir=data_dir,
        participant_id=participant_id,
        session_id=session_id,
        sampling_rate=sampling_rate,
        activity_label=activity_label,
        fatigue_label=fatigue_label,
        distraction_label=distraction_label,
    )
    if record:
        feature_logger.start_recording()

    # CSV Logging Setup
    csv_file = None
    csv_writer = None
    if log_csv:
        csv_file = open(log_csv, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "fps", "ear", "adaptive_ear_t", "mar", "adaptive_mar_t",
            "perclos_pct", "fatigue_score", "distraction_score", "head_pitch", "head_yaw",
            "head_direction", "gaze_direction", "blink_count", "blink_rate",
            "avg_blink_duration", "yawn_count", "head_nods", "alerts"
        ])
        print(f"-> Telemetry logging active: {os.path.abspath(log_csv)}")

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize Video Source
    video_source = video_path if video_path else webcam_idx
    print(f"-> Opening video capture: {video_source}")
    camera = cv2.VideoCapture(video_source)
    if not camera.isOpened():
        print(f"[ERROR]: Could not open video capture device '{video_source}'.")
        face_mesh.close()
        if csv_file:
            csv_file.close()
        raise SystemExit(1)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Scientific Engines
    perclos_engine = PERCLOSEngine(window_seconds=PERCLOS_WINDOW_SECONDS)
    nod_detector = HeadNodDetector(pitch_threshold=HEAD_PITCH_DOWN_THRESHOLD)
    distraction_engine = DistractionScoreEngine(window_seconds=15.0)

    # Calibration Buffers
    calibration_complete = False
    calibration_start_time = None
    calib_ear, calib_mar = [], []
    calib_pitch, calib_yaw, calib_roll = [], [], []
    calib_gaze_x, calib_gaze_y = [], []

    baseline_ear = None
    baseline_mar = None
    baseline_pitch = None
    baseline_yaw = None
    baseline_roll = None
    baseline_gaze_x = None
    baseline_gaze_y = None
    adaptive_ear_threshold = None
    adaptive_mar_threshold = None

    # Temporal State Variables
    eyes_closed = False
    eye_close_start_time = None
    current_eye_closed_dur = 0.0
    last_blink_dur = 0.0
    blink_count = 0
    long_closure_count = 0
    blink_durations = deque(maxlen=30)  # rolling window of blink durations
    blink_timestamps = deque()          # rolling timestamps for rate/min

    mouth_open = False
    mouth_open_start_time = None
    current_mouth_open_dur = 0.0
    yawn_count = 0
    yawn_timestamps = deque()

    looking_away = False
    look_away_start_time = None
    current_look_away_dur = 0.0

    drowsiness_alarm_triggered = False
    yawn_alarm_triggered = False
    distraction_alarm_triggered = False

    fps = 0.0
    prev_frame_time = time.monotonic()
    monitoring_start_time = None
    last_log_time = 0.0

    print("\n[CALIBRATION PHASE]: Please sit comfortably and look forward at the road...")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("-> Video stream completed or interrupted.")
                break

            current_time = time.monotonic()

            # FPS Estimation (Exponential Moving Average)
            delta_t = current_time - prev_frame_time
            prev_frame_time = current_time
            if delta_t > 0:
                inst_fps = 1.0 / delta_t
                fps = inst_fps if fps == 0 else (0.90 * fps + 0.10 * inst_fps)

            # Flip camera horizontally for intuitive mirror view if live webcam
            if not video_path:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Face Landmark Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            active_alerts = []

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Extract Geometric Features
                left_ear, left_eye_pts = calculate_ear(landmarks, LEFT_EYE, width, height)
                right_ear, right_eye_pts = calculate_ear(landmarks, RIGHT_EYE, width, height)
                ear = (left_ear + right_ear) / 2.0

                mar, mouth_pts = calculate_mar(landmarks, width, height)
                raw_head_pose = estimate_raw_head_pose(landmarks, width, height)
                gaze_data = calculate_raw_gaze(landmarks, width, height)

                if not no_gui:
                    draw_eye_contour(frame, left_eye_pts)
                    draw_eye_contour(frame, right_eye_pts)
                    draw_mouth_contour(frame, mouth_pts)
                    draw_iris_marker(frame, gaze_data["left_center"])
                    draw_iris_marker(frame, gaze_data["right_center"])

                # --------------------------------------------------------------
                # STAGE 1: DYNAMIC BASELINE CALIBRATION (First 5 seconds)
                # --------------------------------------------------------------
                if not calibration_complete:
                    if calibration_start_time is None:
                        calibration_start_time = current_time

                    if raw_head_pose is not None:
                        calib_ear.append(ear)
                        calib_mar.append(mar)
                        calib_pitch.append(raw_head_pose["pitch"])
                        calib_yaw.append(raw_head_pose["yaw"])
                        calib_roll.append(raw_head_pose["roll"])
                        calib_gaze_x.append(gaze_data["horizontal_ratio"])
                        calib_gaze_y.append(gaze_data["vertical_ratio"])

                    elapsed_calib = current_time - calibration_start_time
                    remaining_calib = max(0.0, calibration_duration - elapsed_calib)

                    # Log baseline calibration frame (is_calibrated=False) if recording is active
                    feature_logger.log_frame(
                        current_time=current_time,
                        face_detected=True,
                        is_calibrated=False,
                        ear=ear,
                        mar=mar,
                        perclos=float("nan"),
                        blink_rate=float("nan"),
                        blink_duration=float("nan"),
                        eye_closure_duration=0.0,
                        yawn_rate=float("nan"),
                        yawn_duration=0.0,
                        head_pitch=raw_head_pose["pitch"] if raw_head_pose else float("nan"),
                        head_yaw=raw_head_pose["yaw"] if raw_head_pose else float("nan"),
                        head_roll=raw_head_pose["roll"] if raw_head_pose else float("nan"),
                        gaze_x=gaze_data["horizontal_ratio"],
                        gaze_y=gaze_data["vertical_ratio"],
                        looking_away_duration=0.0,
                        head_nodding=0,
                    )

                    if not no_gui:
                        cv2.putText(frame, "CALIBRATING DRIVER BASELINE...", (20, 40),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)
                        cv2.putText(frame, f"Look straight ahead. Progress: {elapsed_calib:.1f}s / {calibration_duration:.1f}s",
                                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        calib_rec_st = feature_logger.get_status()
                        if calib_rec_st.get("is_recording"):
                            cv2.putText(frame, f"REC [CALIBRATING - {calib_rec_st['sample_count']} smp | 10Hz]",
                                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

                    if elapsed_calib >= calibration_duration and len(calib_ear) >= 20:
                        baseline_ear = float(np.median(calib_ear))
                        baseline_mar = float(np.median(calib_mar))
                        baseline_pitch = float(np.median(calib_pitch))
                        baseline_yaw = float(np.median(calib_yaw))
                        baseline_roll = float(np.median(calib_roll))
                        baseline_gaze_x = float(np.median(calib_gaze_x))
                        baseline_gaze_y = float(np.median(calib_gaze_y))

                        adaptive_ear_threshold = clamp(baseline_ear * EAR_THRESHOLD_FACTOR, MIN_EAR_THRESHOLD, MAX_EAR_THRESHOLD)
                        adaptive_mar_threshold = clamp(baseline_mar + MAR_THRESHOLD_OFFSET, MIN_MAR_THRESHOLD, MAX_MAR_THRESHOLD)

                        calibration_complete = True
                        monitoring_start_time = current_time
                        print("\n========================================================")
                        print("              DRIVER CALIBRATION COMPLETE               ")
                        print("========================================================")
                        print(f"-> Baseline EAR:   {baseline_ear:.3f} | Adapted Threshold: {adaptive_ear_threshold:.3f}")
                        print(f"-> Baseline MAR:   {baseline_mar:.3f} | Adapted Threshold: {adaptive_mar_threshold:.3f}")
                        print(f"-> Neutral Pitch:  {baseline_pitch:+.1f}° | Yaw: {baseline_yaw:+.1f}°")
                        print(f"-> Neutral Gaze:   X={baseline_gaze_x:.3f}, Y={baseline_gaze_y:.3f}")
                        print("-> Real-Time Scientific Monitoring Started.\n")

                # --------------------------------------------------------------
                # STAGE 2: SCIENTIFIC MONITORING & TEMPORAL EVALUATION
                # --------------------------------------------------------------
                else:
                    # Head Pose & Gaze Classification
                    if raw_head_pose is not None:
                        head_pose = classify_head_pose(raw_head_pose, baseline_pitch, baseline_yaw, baseline_roll)
                        pitch, yaw, roll = head_pose["pitch"], head_pose["yaw"], head_pose["roll"]
                        head_dir = head_pose["direction"]
                    else:
                        pitch, yaw, roll = 0.0, 0.0, 0.0
                        head_dir = "UNKNOWN"

                    gaze_dir, _, _ = classify_gaze(gaze_data, baseline_gaze_x, baseline_gaze_y)

                    # Eyelid Closure State Machine
                    is_currently_closed = (ear < adaptive_ear_threshold)
                    if is_currently_closed:
                        if not eyes_closed:
                            eyes_closed = True
                            eye_close_start_time = current_time
                            drowsiness_alarm_triggered = False

                        if eye_close_start_time is not None:
                            current_eye_closed_dur = current_time - eye_close_start_time
                    else:
                        if eyes_closed:
                            closure_dur = current_time - eye_close_start_time if eye_close_start_time else 0.0
                            last_blink_dur = closure_dur
                            if MIN_BLINK_DURATION <= closure_dur <= MAX_BLINK_DURATION:
                                blink_count += 1
                                blink_timestamps.append(current_time)
                                blink_durations.append(closure_dur)
                            elif closure_dur > MAX_BLINK_DURATION:
                                long_closure_count += 1
                                blink_durations.append(closure_dur)

                        eyes_closed = False
                        eye_close_start_time = None
                        current_eye_closed_dur = 0.0
                        drowsiness_alarm_triggered = False

                    # Update PERCLOS
                    perclos = perclos_engine.update(current_time, is_currently_closed)

                    # Blink Dynamics
                    while blink_timestamps and current_time - blink_timestamps[0] > 60.0:
                        blink_timestamps.popleft()

                    obs_window = min(60.0, max(current_time - monitoring_start_time, 1.0))
                    blink_rate = (len(blink_timestamps) / obs_window) * 60.0
                    avg_blink_dur = float(np.mean(blink_durations)) if blink_durations else 0.20

                    # Head Droop / Nodding Detection
                    if nod_detector.update(current_time, pitch, eyes_closed):
                        active_alerts.append("HEAD NOD DETECTED")

                    # Yawning State Machine
                    is_mouth_open = (mar > adaptive_mar_threshold)
                    if is_mouth_open:
                        if not mouth_open:
                            mouth_open = True
                            mouth_open_start_time = current_time
                            yawn_alarm_triggered = False
                        if mouth_open_start_time is not None:
                            current_mouth_open_dur = current_time - mouth_open_start_time
                    else:
                        if mouth_open:
                            y_dur = current_time - mouth_open_start_time if mouth_open_start_time else 0.0
                            if y_dur >= MIN_YAWN_DURATION:
                                yawn_count += 1
                                yawn_timestamps.append(current_time)
                        mouth_open = False
                        mouth_open_start_time = None
                        current_mouth_open_dur = 0.0
                        yawn_alarm_triggered = False

                    while yawn_timestamps and current_time - yawn_timestamps[0] > 60.0:
                        yawn_timestamps.popleft()
                    recent_yawns = len(yawn_timestamps)

                    # Distraction State Machine (Look-Away)
                    head_away = (
                        abs(yaw) > HEAD_YAW_THRESHOLD
                        or pitch > HEAD_PITCH_DOWN_THRESHOLD
                        or pitch < -HEAD_PITCH_UP_THRESHOLD
                    )
                    gaze_away = (gaze_dir != "CENTER")
                    is_looking_away = (head_away or gaze_away)

                    if is_looking_away:
                        if not looking_away:
                            looking_away = True
                            look_away_start_time = current_time
                            distraction_alarm_triggered = False
                        if look_away_start_time is not None:
                            current_look_away_dur = current_time - look_away_start_time
                    else:
                        looking_away = False
                        look_away_start_time = None
                        current_look_away_dur = 0.0
                        distraction_alarm_triggered = False

                    distraction_score = distraction_engine.update(current_time, is_looking_away, current_look_away_dur)

                    # Multi-Modal Composite Fatigue Score
                    fatigue_score = FatigueScoreEngine.calculate(
                        perclos=perclos,
                        avg_blink_duration=avg_blink_dur,
                        recent_yawns_in_window=recent_yawns,
                        head_nod_count=nod_detector.total_nods,
                        current_eye_closed_duration=current_eye_closed_dur,
                    )

                    # ----------------------------------------------------------
                    # TIERED SCIENTIFIC ALERTS & ALARM TRIGGERS
                    # ----------------------------------------------------------
                    # Critical Acute Eye Closure (Microsleep Alert)
                    if eyes_closed and current_eye_closed_dur >= EYE_CLOSED_ALERT_SECONDS:
                        active_alerts.append("MICROSLEEP / EYE CLOSURE")
                        if not drowsiness_alarm_triggered:
                            trigger_alarm(alarm_path)
                            drowsiness_alarm_triggered = True

                    # Critical Prolonged Yawn Alert
                    if mouth_open and current_mouth_open_dur >= YAWN_ALERT_SECONDS:
                        active_alerts.append("PROLONGED YAWN")
                        if not yawn_alarm_triggered:
                            trigger_alarm(alarm_path)
                            yawn_alarm_triggered = True

                    # Distraction Alert
                    if looking_away and current_look_away_dur >= LOOK_AWAY_ALERT_SECONDS:
                        active_alerts.append("OFF-ROAD DISTRACTION")
                        if alarm_distraction and not distraction_alarm_triggered:
                            trigger_alarm(alarm_path)
                            distraction_alarm_triggered = True

                    # Overall High Fatigue Alarm Trigger
                    if fatigue_score >= FATIGUE_CRITICAL and not drowsiness_alarm_triggered:
                        active_alerts.append("CRITICAL FATIGUE")
                        trigger_alarm(alarm_path)
                        drowsiness_alarm_triggered = True

                    # Render Scientific HUD
                    # GRU Behavioural Feature Logging (Rate-controlled 10 Hz)
                    obs_yawn_rate = (recent_yawns / obs_window) * 60.0 if obs_window > 0 else 0.0
                    feature_logger.log_frame(
                        current_time=current_time,
                        face_detected=True,
                        is_calibrated=True,
                        ear=ear,
                        mar=mar,
                        perclos=perclos,
                        blink_rate=blink_rate,
                        blink_duration=last_blink_dur,
                        eye_closure_duration=current_eye_closed_dur,
                        yawn_rate=obs_yawn_rate,
                        yawn_duration=current_mouth_open_dur,
                        head_pitch=pitch,
                        head_yaw=yaw,
                        head_roll=roll,
                        gaze_x=gaze_data["horizontal_ratio"],
                        gaze_y=gaze_data["vertical_ratio"],
                        looking_away_duration=current_look_away_dur,
                        head_nodding=1 if nod_detector.nodding_active else 0,
                    )

                    # Render Scientific HUD
                    if not no_gui:
                        draw_hud(
                            frame=frame,
                            fatigue_score=fatigue_score,
                            distraction_score=distraction_score,
                            perclos=perclos,
                            ear=ear,
                            ear_t=adaptive_ear_threshold,
                            mar=mar,
                            mar_t=adaptive_mar_threshold,
                            blink_count=blink_count,
                            blink_rate=blink_rate,
                            avg_blink_dur=avg_blink_dur,
                            yawn_count=yawn_count,
                            head_nods=nod_detector.total_nods,
                            head_dir=head_dir,
                            pitch=pitch,
                            yaw=yaw,
                            gaze_dir=gaze_dir,
                            fps=fps,
                            active_alerts=active_alerts,
                            logger_status=feature_logger.get_status(),
                        )

                    # Diagnostic CSV Logging (sample every 0.5s or on alert)
                    if csv_writer and (current_time - last_log_time >= 0.5 or active_alerts):
                        last_log_time = current_time
                        csv_writer.writerow([
                            round(current_time, 2), round(fps, 1), round(ear, 3), round(adaptive_ear_threshold, 3),
                            round(mar, 3), round(adaptive_mar_threshold, 3), round(perclos * 100.0, 1),
                            round(fatigue_score, 1), round(distraction_score, 1), round(pitch, 1), round(yaw, 1),
                            head_dir, gaze_dir, blink_count, round(blink_rate, 1), round(avg_blink_dur, 2),
                            yawn_count, nod_detector.total_nods, ";".join(active_alerts)
                        ])

            else:
                # No Face Detected in frame
                feature_logger.log_frame(
                    current_time=current_time,
                    face_detected=False,
                    is_calibrated=calibration_complete,
                )

                if not calibration_complete:
                    calibration_start_time = None
                    calib_ear.clear()
                    calib_mar.clear()
                    calib_pitch.clear()
                    calib_yaw.clear()
                    calib_roll.clear()
                    calib_gaze_x.clear()
                    calib_gaze_y.clear()
                    if not no_gui:
                        cv2.putText(frame, "CALIBRATION: Please face the camera...", (20, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                else:
                    if not no_gui:
                        cv2.putText(frame, "NO FACE DETECTED", (20, 45),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

                eyes_closed = False
                eye_close_start_time = None
                mouth_open = False
                mouth_open_start_time = None
                looking_away = False
                look_away_start_time = None

            if not no_gui:
                cv2.imshow("DriveGuard AI - Personalized Driver Monitoring", frame)
                raw_key = cv2.waitKey(1)
                if raw_key != -1:
                    key = raw_key & 0xFF
                    if key == ord("q"):
                        print("\n-> User quit signal received ('q').")
                        break
                    elif key in (ord("r"), ord("R")):
                        is_rec = feature_logger.toggle_recording()
                        status_str = "STARTED" if is_rec else "STOPPED"
                        print(f"-> [Data Collection]: Recording {status_str} (Participant: {feature_logger.participant_id}, Activity: {feature_logger.activity_label})")
                    elif ord("1") <= key <= ord("8"):
                        act_idx = key - ord("0")
                        if feature_logger.set_activity_by_index(act_idx):
                            print(f"-> [Data Collection]: Activity set to -> '{feature_logger.activity_label}'")
                    elif key in (ord("p"), ord("P")):
                        try:
                            num_str = "".join(filter(str.isdigit, feature_logger.participant_id)) or "1"
                            new_p = f"P{int(num_str) + 1:03d}"
                        except Exception:
                            new_p = f"{feature_logger.participant_id}_next"
                        feature_logger.set_participant(new_p)
                        print(f"-> [Data Collection]: Participant set to -> '{new_p}'")

    finally:
        print("\n-> Shutting down DriveGuard AI subsystems...")
        camera.release()
        if not no_gui:
            cv2.destroyAllWindows()
        face_mesh.close()
        feature_logger.close()
        if csv_file:
            csv_file.close()

        print("\n========================================================")
        print("                FINAL SESSION STATISTICS                ")
        print("========================================================")
        print(f"Total Blinks:            {blink_count}")
        print(f"Long Closures (>0.4s):   {long_closure_count}")
        print(f"Total Yawns:             {yawn_count}")
        print(f"Total Head Nods:         {nod_detector.total_nods}")
        if calibration_complete:
            print("--------------------------------------------------------")
            print(f"Calibrated Baseline EAR: {baseline_ear:.3f} (Threshold: {adaptive_ear_threshold:.3f})")
            print(f"Calibrated Baseline MAR: {baseline_mar:.3f} (Threshold: {adaptive_mar_threshold:.3f})")
            print(f"Calibrated Neutral Pose: Pitch={baseline_pitch:+.1f}°, Yaw={baseline_yaw:+.1f}°")
            print(f"Calibrated Neutral Gaze: X={baseline_gaze_x:.3f}, Y={baseline_gaze_y:.3f}")
        print("========================================================\n")


# ==============================================================================
# CLI ARGUMENT PARSER & ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DriveGuard AI - Multi-Modal Scientific Driver Drowsiness and Distraction Monitoring System"
    )
    parser.add_argument(
        "-w", "--webcam", type=int, default=0, help="Webcam device index (default: 0)."
    )
    parser.add_argument(
        "-v", "--video", type=str, default=None, help="Optional path to input video file instead of live webcam."
    )
    parser.add_argument(
        "-a", "--alarm", type=str, default="Alert.wav", help="Path to alarm WAV file (default: Alert.wav)."
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Optional output CSV path for diagnostic telemetry logging."
    )
    parser.add_argument(
        "-c", "--calibration", type=float, default=CALIBRATION_SECONDS,
        help=f"Startup baseline calibration duration in seconds (default: {CALIBRATION_SECONDS}s)."
    )
    parser.add_argument(
        "--alarm-distraction", action="store_true", default=False,
        help="Sound buzzer when driver looks away off-road for > 2 seconds."
    )
    parser.add_argument(
        "--no-gui", action="store_true", default=False,
        help="Run without displaying OpenCV GUI window (useful for headless servers/benchmarks)."
    )
    # Behavioural Data Collection Arguments (GRU Dataset Preparation)
    parser.add_argument(
        "--record", action="store_true", default=False,
        help="Start recording 10 Hz behavioural features immediately on startup."
    )
    parser.add_argument(
        "--participant", type=str, default="P001",
        help="Participant identifier for dataset collection (default: P001)."
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Session identifier for dataset collection (default: auto timestamped)."
    )
    parser.add_argument(
        "--activity", type=str, default="normal_alert", choices=SUPPORTED_ACTIVITIES,
        help="Initial behavioural activity label (default: normal_alert)."
    )
    parser.add_argument(
        "--sampling-rate", type=float, default=10.0,
        help="Sampling rate in Hz for feature logging (default: 10.0 Hz)."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory to store raw CSV logs (default: data/raw)."
    )
    parser.add_argument(
        "--fatigue-label", type=str, default="unknown",
        help="Ground-truth fatigue state label if known (default: unknown)."
    )
    parser.add_argument(
        "--distraction-label", type=str, default="unknown",
        help="Ground-truth distraction state label if known (default: unknown)."
    )

    args = parser.parse_args()

    run_driveguard(
        webcam_idx=args.webcam,
        video_path=args.video,
        alarm_path=args.alarm,
        log_csv=args.log,
        calibration_duration=args.calibration,
        alarm_distraction=args.alarm_distraction,
        no_gui=args.no_gui,
        record=args.record,
        participant_id=args.participant,
        session_id=args.session,
        activity_label=args.activity,
        sampling_rate=args.sampling_rate,
        data_dir=args.data_dir,
        fatigue_label=args.fatigue_label,
        distraction_label=args.distraction_label,
    )