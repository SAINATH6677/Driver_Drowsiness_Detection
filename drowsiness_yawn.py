import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time

from collections import deque
from threading import Thread, Lock
from playsound3 import playsound


# ============================================================
# DRIVEGUARD AI
# PERSONALIZED DRIVER DROWSINESS + DISTRACTION MONITORING
#
# IMPLEMENTED:
#
# 1. Camera and face-landmark processing
# 2. Eye closure and blink detection
# 3. Yawn detection
# 4. Head-pose and gaze estimation
# 8. Basic personalized startup calibration
#
# NEXT:
#
# 5. Temporal behaviour analysis
# 6. Fatigue-score calculation
# 7. Distraction-score calculation
# 9. Risk prediction and intelligent alerting
# 10. Testing, evaluation and comparison
# ============================================================


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

CALIBRATION_SECONDS = 5.0


# ============================================================
# EYE CONFIGURATION
# ============================================================

# Personalized eye threshold:
#
# closed threshold =
# calibrated normal EAR * this factor
#
# Example:
# normal EAR = 0.28
# threshold = 0.28 * 0.72 = 0.2016

EAR_THRESHOLD_FACTOR = 0.72

# Safety limits for calculated threshold
MIN_EAR_THRESHOLD = 0.12
MAX_EAR_THRESHOLD = 0.28


# Eyes closed this long -> drowsiness warning
EYE_CLOSED_ALERT_SECONDS = 1.5


# Normal blink duration range
MIN_BLINK_DURATION = 0.08
MAX_BLINK_DURATION = 0.80


# ============================================================
# MOUTH / YAWN CONFIGURATION
# ============================================================

# Personalized MAR threshold:
#
# threshold = normal MAR + offset
#
MAR_THRESHOLD_OFFSET = 0.25

# Prevent threshold from becoming unrealistically low
MIN_MAR_THRESHOLD = 0.30

# Maximum allowed adaptive threshold
MAX_MAR_THRESHOLD = 0.65


# Minimum mouth-open duration considered a yawn
MIN_YAWN_DURATION = 0.80

# Warning during prolonged yawn
YAWN_ALERT_SECONDS = 1.20


# ============================================================
# HEAD POSE CONFIGURATION
# ============================================================

# These thresholds are now deviations from the person's
# calibrated forward-facing position.

HEAD_YAW_THRESHOLD = 25.0
HEAD_PITCH_THRESHOLD = 20.0
HEAD_ROLL_THRESHOLD = 25.0


# ============================================================
# GAZE CONFIGURATION
# ============================================================

# Gaze is now measured relative to the person's calibrated
# forward-looking gaze.

GAZE_HORIZONTAL_DEVIATION = 0.14
GAZE_VERTICAL_DEVIATION = 0.16


# ============================================================
# DISTRACTION CONFIGURATION
# ============================================================

LOOK_AWAY_ALERT_SECONDS = 2.0


# ============================================================
# ALARM GLOBALS
# ============================================================

alarm_lock = Lock()
alarm_playing = False


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# LANDMARK INDICES
# ============================================================

# EAR landmarks

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]


# Iris landmarks
# refine_landmarks=True required

LEFT_IRIS = [
    474, 475, 476, 477
]

RIGHT_IRIS = [
    469, 470, 471, 472
]


# Eye boundaries for gaze

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


# Mouth

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# Head pose

NOSE_TIP = 1
CHIN = 152

LEFT_EYE_OUTER_HEAD = 263
RIGHT_EYE_OUTER_HEAD = 33

LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "DriveGuard AI - Personalized Drowsiness "
        "and Distraction Detection"
    )
)

parser.add_argument(
    "-w",
    "--webcam",
    type=int,
    default=0,
    help="Webcam index. Default: 0."
)

parser.add_argument(
    "-a",
    "--alarm",
    type=str,
    default="Alert.wav",
    help="Path to alarm WAV file."
)

args = parser.parse_args()


# ============================================================
# ALARM FUNCTIONS
# ============================================================

def play_alarm(path):

    global alarm_playing

    with alarm_lock:

        if alarm_playing:
            return

        alarm_playing = True

    try:

        print("ALERT SOUND!")

        playsound(path)

    except Exception as error:

        print("Alarm error:", error)

    finally:

        with alarm_lock:
            alarm_playing = False


def trigger_alarm(path):

    global alarm_playing

    with alarm_lock:

        if alarm_playing:
            return

    thread = Thread(
        target=play_alarm,
        args=(path,),
        daemon=True
    )

    thread.start()


# ============================================================
# BASIC UTILITY FUNCTIONS
# ============================================================

def calculate_distance(point1, point2):

    p1 = np.array(
        point1,
        dtype=np.float32
    )

    p2 = np.array(
        point2,
        dtype=np.float32
    )

    return np.linalg.norm(
        p1 - p2
    )


def landmark_to_point(
    landmark,
    width,
    height
):

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    return (
        x,
        y
    )


def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


def angle_difference(
    current_angle,
    baseline_angle
):
    """
    Returns shortest angular difference.

    Useful because angles may wrap around at +/-180 degrees.
    """

    difference = (
        current_angle -
        baseline_angle +
        180.0
    ) % 360.0 - 180.0

    return difference


# ============================================================
# EAR
# ============================================================

def calculate_ear(
    landmarks,
    eye_indices,
    width,
    height
):

    points = []

    for index in eye_indices:

        point = landmark_to_point(
            landmarks[index],
            width,
            height
        )

        points.append(
            point
        )


    vertical_distance_1 = calculate_distance(
        points[1],
        points[5]
    )

    vertical_distance_2 = calculate_distance(
        points[2],
        points[4]
    )

    horizontal_distance = calculate_distance(
        points[0],
        points[3]
    )


    if horizontal_distance == 0:

        return (
            0.0,
            points
        )


    ear = (
        vertical_distance_1 +
        vertical_distance_2
    ) / (
        2.0 *
        horizontal_distance
    )


    return (
        float(ear),
        points
    )


# ============================================================
# MAR
# ============================================================

def calculate_mar(
    landmarks,
    width,
    height
):

    top1 = landmark_to_point(
        landmarks[13],
        width,
        height
    )

    bottom1 = landmark_to_point(
        landmarks[14],
        width,
        height
    )


    top2 = landmark_to_point(
        landmarks[82],
        width,
        height
    )

    bottom2 = landmark_to_point(
        landmarks[87],
        width,
        height
    )


    top3 = landmark_to_point(
        landmarks[312],
        width,
        height
    )

    bottom3 = landmark_to_point(
        landmarks[317],
        width,
        height
    )


    left = landmark_to_point(
        landmarks[MOUTH_LEFT],
        width,
        height
    )

    right = landmark_to_point(
        landmarks[MOUTH_RIGHT],
        width,
        height
    )


    vertical1 = calculate_distance(
        top1,
        bottom1
    )

    vertical2 = calculate_distance(
        top2,
        bottom2
    )

    vertical3 = calculate_distance(
        top3,
        bottom3
    )


    horizontal = calculate_distance(
        left,
        right
    )


    if horizontal == 0:

        return (
            0.0,
            (
                top1,
                bottom1,
                left,
                right
            )
        )


    average_vertical = (
        vertical1 +
        vertical2 +
        vertical3
    ) / 3.0


    mar = (
        average_vertical /
        horizontal
    )


    return (
        float(mar),
        (
            top1,
            bottom1,
            left,
            right
        )
    )


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_eye(
    frame,
    points
):

    points_array = np.array(
        points,
        dtype=np.int32
    )

    hull = cv2.convexHull(
        points_array
    )

    cv2.polylines(
        frame,
        [hull],
        True,
        (0, 255, 0),
        1
    )


def draw_mouth(
    frame,
    mouth_points
):

    top, bottom, left, right = mouth_points

    cv2.line(
        frame,
        top,
        bottom,
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        left,
        right,
        (0, 255, 0),
        2
    )


# ============================================================
# IRIS CENTER
# ============================================================

def calculate_iris_center(
    landmarks,
    iris_indices,
    width,
    height
):

    points = []

    for index in iris_indices:

        point = landmark_to_point(
            landmarks[index],
            width,
            height
        )

        points.append(
            point
        )


    points_array = np.array(
        points,
        dtype=np.float32
    )


    center = np.mean(
        points_array,
        axis=0
    )


    center = (
        int(center[0]),
        int(center[1])
    )


    return (
        center,
        points
    )


# ============================================================
# GAZE RATIO
# ============================================================

def calculate_eye_gaze_ratio(
    landmarks,
    iris_indices,
    outer_index,
    inner_index,
    top_index,
    bottom_index,
    width,
    height
):

    iris_center, iris_points = calculate_iris_center(
        landmarks,
        iris_indices,
        width,
        height
    )


    outer_point = landmark_to_point(
        landmarks[outer_index],
        width,
        height
    )

    inner_point = landmark_to_point(
        landmarks[inner_index],
        width,
        height
    )

    top_point = landmark_to_point(
        landmarks[top_index],
        width,
        height
    )

    bottom_point = landmark_to_point(
        landmarks[bottom_index],
        width,
        height
    )


    eye_left_x = min(
        outer_point[0],
        inner_point[0]
    )

    eye_right_x = max(
        outer_point[0],
        inner_point[0]
    )

    eye_top_y = min(
        top_point[1],
        bottom_point[1]
    )

    eye_bottom_y = max(
        top_point[1],
        bottom_point[1]
    )


    eye_width = (
        eye_right_x -
        eye_left_x
    )

    eye_height = (
        eye_bottom_y -
        eye_top_y
    )


    if eye_width <= 0:

        horizontal_ratio = 0.5

    else:

        horizontal_ratio = (
            iris_center[0] -
            eye_left_x
        ) / eye_width


    if eye_height <= 0:

        vertical_ratio = 0.5

    else:

        vertical_ratio = (
            iris_center[1] -
            eye_top_y
        ) / eye_height


    horizontal_ratio = float(
        np.clip(
            horizontal_ratio,
            0.0,
            1.0
        )
    )

    vertical_ratio = float(
        np.clip(
            vertical_ratio,
            0.0,
            1.0
        )
    )


    return (
        horizontal_ratio,
        vertical_ratio,
        iris_center,
        iris_points
    )


# ============================================================
# RAW GAZE
# ============================================================

def calculate_raw_gaze(
    landmarks,
    width,
    height
):

    (
        left_horizontal,
        left_vertical,
        left_center,
        left_points
    ) = calculate_eye_gaze_ratio(

        landmarks,

        LEFT_IRIS,

        LEFT_EYE_OUTER,
        LEFT_EYE_INNER,

        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,

        width,
        height
    )


    (
        right_horizontal,
        right_vertical,
        right_center,
        right_points
    ) = calculate_eye_gaze_ratio(

        landmarks,

        RIGHT_IRIS,

        RIGHT_EYE_OUTER,
        RIGHT_EYE_INNER,

        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,

        width,
        height
    )


    horizontal_ratio = (
        left_horizontal +
        right_horizontal
    ) / 2.0


    vertical_ratio = (
        left_vertical +
        right_vertical
    ) / 2.0


    return {

        "horizontal_ratio":
            float(horizontal_ratio),

        "vertical_ratio":
            float(vertical_ratio),

        "left_center":
            left_center,

        "right_center":
            right_center,

        "left_points":
            left_points,

        "right_points":
            right_points
    }


# ============================================================
# PERSONALIZED GAZE CLASSIFICATION
# ============================================================

def classify_gaze(
    gaze_data,
    baseline_gaze_x,
    baseline_gaze_y
):

    gaze_x = gaze_data[
        "horizontal_ratio"
    ]

    gaze_y = gaze_data[
        "vertical_ratio"
    ]


    delta_x = (
        gaze_x -
        baseline_gaze_x
    )

    delta_y = (
        gaze_y -
        baseline_gaze_y
    )


    # Vertical first because looking down
    # is especially relevant for driver distraction.

    if (
        delta_y >
        GAZE_VERTICAL_DEVIATION
    ):

        direction = "DOWN"


    elif (
        delta_y <
        -GAZE_VERTICAL_DEVIATION
    ):

        direction = "UP"


    elif (
        delta_x >
        GAZE_HORIZONTAL_DEVIATION
    ):

        direction = "RIGHT"


    elif (
        delta_x <
        -GAZE_HORIZONTAL_DEVIATION
    ):

        direction = "LEFT"


    else:

        direction = "CENTER"


    return (
        direction,
        delta_x,
        delta_y
    )


# ============================================================
# DRAW IRIS
# ============================================================

def draw_iris(
    frame,
    gaze_data
):

    cv2.circle(
        frame,
        gaze_data["left_center"],
        3,
        (255, 0, 255),
        -1
    )

    cv2.circle(
        frame,
        gaze_data["right_center"],
        3,
        (255, 0, 255),
        -1
    )


# ============================================================
# RAW HEAD POSE
# ============================================================

def estimate_raw_head_pose(
    landmarks,
    width,
    height
):

    image_points = np.array(
        [

            landmark_to_point(
                landmarks[NOSE_TIP],
                width,
                height
            ),

            landmark_to_point(
                landmarks[CHIN],
                width,
                height
            ),

            landmark_to_point(
                landmarks[LEFT_EYE_OUTER_HEAD],
                width,
                height
            ),

            landmark_to_point(
                landmarks[RIGHT_EYE_OUTER_HEAD],
                width,
                height
            ),

            landmark_to_point(
                landmarks[LEFT_MOUTH_CORNER],
                width,
                height
            ),

            landmark_to_point(
                landmarks[RIGHT_MOUTH_CORNER],
                width,
                height
            )

        ],
        dtype=np.float64
    )


    model_points = np.array(
        [

            (0.0, 0.0, 0.0),

            (0.0, -330.0, -65.0),

            (-225.0, 170.0, -135.0),

            (225.0, 170.0, -135.0),

            (-150.0, -150.0, -125.0),

            (150.0, -150.0, -125.0)

        ],
        dtype=np.float64
    )


    focal_length = float(
        width
    )


    center = (
        width / 2.0,
        height / 2.0
    )


    camera_matrix = np.array(
        [

            [
                focal_length,
                0.0,
                center[0]
            ],

            [
                0.0,
                focal_length,
                center[1]
            ],

            [
                0.0,
                0.0,
                1.0
            ]

        ],
        dtype=np.float64
    )


    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64
    )


    success, rotation_vector, translation_vector = cv2.solvePnP(

        model_points,

        image_points,

        camera_matrix,

        distortion_coefficients,

        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not success:

        return None


    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotation_matrix
    )


    pitch = float(
        angles[0]
    )

    yaw = float(
        angles[1]
    )

    roll = float(
        angles[2]
    )


    return {

        "pitch":
            pitch,

        "yaw":
            yaw,

        "roll":
            roll,

        "nose_point":
            (
                int(image_points[0][0]),
                int(image_points[0][1])
            )
    }


# ============================================================
# PERSONALIZED HEAD-POSE CLASSIFICATION
# ============================================================

def classify_head_pose(
    raw_head_pose,
    baseline_pitch,
    baseline_yaw,
    baseline_roll
):

    corrected_pitch = angle_difference(
        raw_head_pose["pitch"],
        baseline_pitch
    )

    corrected_yaw = angle_difference(
        raw_head_pose["yaw"],
        baseline_yaw
    )

    corrected_roll = angle_difference(
        raw_head_pose["roll"],
        baseline_roll
    )


    if (
        corrected_yaw >
        HEAD_YAW_THRESHOLD
    ):

        direction = "RIGHT"


    elif (
        corrected_yaw <
        -HEAD_YAW_THRESHOLD
    ):

        direction = "LEFT"


    elif (
        corrected_pitch >
        HEAD_PITCH_THRESHOLD
    ):

        direction = "DOWN"


    elif (
        corrected_pitch <
        -HEAD_PITCH_THRESHOLD
    ):

        direction = "UP"


    else:

        direction = "FORWARD"


    return {

        "pitch":
            corrected_pitch,

        "yaw":
            corrected_yaw,

        "roll":
            corrected_roll,

        "direction":
            direction
    }


# ============================================================
# CHECK ALARM FILE
# ============================================================

alarm_path = args.alarm


if not os.path.exists(
    alarm_path
):

    print()

    print(
        "ERROR: Alarm file not found!"
    )

    print(
        "Expected:"
    )

    print(
        os.path.abspath(
            alarm_path
        )
    )

    print()

    raise SystemExit(1)


print(
    "-> Alarm file:",
    os.path.abspath(
        alarm_path
    )
)


# ============================================================
# START MEDIAPIPE
# ============================================================

print(
    "-> Loading MediaPipe Face Mesh..."
)


face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# START CAMERA
# ============================================================

print(
    "-> Starting video stream..."
)


camera = cv2.VideoCapture(
    args.webcam
)


if not camera.isOpened():

    print()

    print(
        "ERROR: Could not open webcam."
    )

    print(
        "Try:"
    )

    print(
        "python drowsiness_yawn.py --webcam 1"
    )

    face_mesh.close()

    raise SystemExit(1)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# CALIBRATION VARIABLES
# ============================================================

calibration_complete = False

calibration_start_time = None


calibration_ear_values = []
calibration_mar_values = []

calibration_pitch_values = []
calibration_yaw_values = []
calibration_roll_values = []

calibration_gaze_x_values = []
calibration_gaze_y_values = []


baseline_ear = None
baseline_mar = None

baseline_pitch = None
baseline_yaw = None
baseline_roll = None

baseline_gaze_x = None
baseline_gaze_y = None


adaptive_ear_threshold = None
adaptive_mar_threshold = None


# ============================================================
# EYE / BLINK STATE
# ============================================================

eyes_closed = False

eye_close_start_time = None

current_eye_closed_duration = 0.0

last_blink_duration = 0.0

blink_count = 0

long_eye_closure_count = 0


blink_timestamps = deque()

monitoring_start_time = None

drowsiness_alarm_triggered = False


# ============================================================
# YAWN STATE
# ============================================================

mouth_open = False

mouth_open_start_time = None

current_mouth_open_duration = 0.0

last_yawn_duration = 0.0

yawn_count = 0

yawn_alarm_triggered = False


# ============================================================
# DISTRACTION STATE
# ============================================================

looking_away = False

look_away_start_time = None

current_look_away_duration = 0.0


# ============================================================
# FPS
# ============================================================

fps = 0.0

previous_frame_time = time.monotonic()


# ============================================================
# STARTUP MESSAGE
# ============================================================

print()

print(
    "=========================================="
)

print(
    "         DRIVEGUARD AI CALIBRATION"
)

print(
    "=========================================="
)

print()

print(
    "For the first 5 seconds:"
)

print(
    "1. Look directly forward."
)

print(
    "2. Keep your eyes normally open."
)

print(
    "3. Keep your mouth closed."
)

print(
    "4. Keep your head in a comfortable normal position."
)

print()

print(
    "Calibration starts when your face is detected."
)

print()

print(
    "Press 'q' to quit."
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:


        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        success, frame = camera.read()


        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break


        current_time = time.monotonic()


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        frame_delta = (
            current_time -
            previous_frame_time
        )

        previous_frame_time = (
            current_time
        )


        if frame_delta > 0:

            instantaneous_fps = (
                1.0 /
                frame_delta
            )


            if fps == 0:

                fps = (
                    instantaneous_fps
                )

            else:

                fps = (
                    0.90 * fps +
                    0.10 * instantaneous_fps
                )


        # ----------------------------------------------------
        # FLIP FRAME
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # RGB CONVERSION
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        rgb_frame.flags.writeable = False


        results = face_mesh.process(
            rgb_frame
        )


        rgb_frame.flags.writeable = True


        # ====================================================
        # FACE FOUND
        # ====================================================

        if results.multi_face_landmarks:

            face = (
                results.multi_face_landmarks[0]
            )

            landmarks = (
                face.landmark
            )


            # =================================================
            # CALCULATE EAR
            # =================================================

            left_ear, left_eye_points = calculate_ear(

                landmarks,

                LEFT_EYE,

                width,

                height
            )


            right_ear, right_eye_points = calculate_ear(

                landmarks,

                RIGHT_EYE,

                width,

                height
            )


            ear = (
                left_ear +
                right_ear
            ) / 2.0


            draw_eye(
                frame,
                left_eye_points
            )

            draw_eye(
                frame,
                right_eye_points
            )


            # =================================================
            # CALCULATE MAR
            # =================================================

            mar, mouth_points = calculate_mar(

                landmarks,

                width,

                height
            )


            draw_mouth(
                frame,
                mouth_points
            )


            # =================================================
            # RAW HEAD POSE
            # =================================================

            raw_head_pose = estimate_raw_head_pose(

                landmarks,

                width,

                height
            )


            # =================================================
            # RAW GAZE
            # =================================================

            gaze_data = calculate_raw_gaze(

                landmarks,

                width,

                height
            )


            draw_iris(
                frame,
                gaze_data
            )


            # =================================================
            # CALIBRATION MODE
            # =================================================

            if not calibration_complete:


                if calibration_start_time is None:

                    calibration_start_time = (
                        current_time
                    )


                if raw_head_pose is not None:


                    calibration_ear_values.append(
                        ear
                    )

                    calibration_mar_values.append(
                        mar
                    )


                    calibration_pitch_values.append(
                        raw_head_pose[
                            "pitch"
                        ]
                    )

                    calibration_yaw_values.append(
                        raw_head_pose[
                            "yaw"
                        ]
                    )

                    calibration_roll_values.append(
                        raw_head_pose[
                            "roll"
                        ]
                    )


                    calibration_gaze_x_values.append(
                        gaze_data[
                            "horizontal_ratio"
                        ]
                    )

                    calibration_gaze_y_values.append(
                        gaze_data[
                            "vertical_ratio"
                        ]
                    )


                elapsed_calibration = (
                    current_time -
                    calibration_start_time
                )


                remaining = max(
                    0.0,
                    CALIBRATION_SECONDS -
                    elapsed_calibration
                )


                cv2.putText(

                    frame,

                    "CALIBRATING DRIVER...",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 255),

                    2
                )


                cv2.putText(

                    frame,

                    "Look forward - eyes open - mouth closed",

                    (20, 75),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Remaining: {:.1f}s".format(
                        remaining
                    ),

                    (20, 105),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 255),

                    2
                )


                cv2.putText(

                    frame,

                    "EAR: {:.3f}".format(
                        ear
                    ),

                    (450, 30),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 0),

                    1
                )


                cv2.putText(

                    frame,

                    "MAR: {:.3f}".format(
                        mar
                    ),

                    (450, 55),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 0),

                    1
                )


                # ---------------------------------------------
                # FINISH CALIBRATION
                # ---------------------------------------------

                if (
                    elapsed_calibration >=
                    CALIBRATION_SECONDS
                    and
                    len(calibration_ear_values) >= 20
                    and
                    len(calibration_pitch_values) >= 20
                ):


                    # Median is more resistant to blinks,
                    # tiny movements and noisy frames.

                    baseline_ear = float(
                        np.median(
                            calibration_ear_values
                        )
                    )


                    baseline_mar = float(
                        np.median(
                            calibration_mar_values
                        )
                    )


                    baseline_pitch = float(
                        np.median(
                            calibration_pitch_values
                        )
                    )


                    baseline_yaw = float(
                        np.median(
                            calibration_yaw_values
                        )
                    )


                    baseline_roll = float(
                        np.median(
                            calibration_roll_values
                        )
                    )


                    baseline_gaze_x = float(
                        np.median(
                            calibration_gaze_x_values
                        )
                    )


                    baseline_gaze_y = float(
                        np.median(
                            calibration_gaze_y_values
                        )
                    )


                    # -----------------------------------------
                    # PERSONALIZED EAR THRESHOLD
                    # -----------------------------------------

                    adaptive_ear_threshold = (
                        baseline_ear *
                        EAR_THRESHOLD_FACTOR
                    )


                    adaptive_ear_threshold = clamp(

                        adaptive_ear_threshold,

                        MIN_EAR_THRESHOLD,

                        MAX_EAR_THRESHOLD
                    )


                    # -----------------------------------------
                    # PERSONALIZED MAR THRESHOLD
                    # -----------------------------------------

                    adaptive_mar_threshold = (
                        baseline_mar +
                        MAR_THRESHOLD_OFFSET
                    )


                    adaptive_mar_threshold = clamp(

                        adaptive_mar_threshold,

                        MIN_MAR_THRESHOLD,

                        MAX_MAR_THRESHOLD
                    )


                    calibration_complete = True

                    monitoring_start_time = (
                        current_time
                    )


                    # Reset all event states after calibration.

                    eyes_closed = False
                    eye_close_start_time = None
                    current_eye_closed_duration = 0.0

                    mouth_open = False
                    mouth_open_start_time = None
                    current_mouth_open_duration = 0.0

                    looking_away = False
                    look_away_start_time = None
                    current_look_away_duration = 0.0


                    print()

                    print(
                        "=========================================="
                    )

                    print(
                        "        CALIBRATION COMPLETE"
                    )

                    print(
                        "=========================================="
                    )

                    print()

                    print(
                        "Baseline EAR:",
                        round(
                            baseline_ear,
                            3
                        )
                    )

                    print(
                        "Eye threshold:",
                        round(
                            adaptive_ear_threshold,
                            3
                        )
                    )

                    print()

                    print(
                        "Baseline MAR:",
                        round(
                            baseline_mar,
                            3
                        )
                    )

                    print(
                        "Yawn threshold:",
                        round(
                            adaptive_mar_threshold,
                            3
                        )
                    )

                    print()

                    print(
                        "Baseline Pitch:",
                        round(
                            baseline_pitch,
                            2
                        )
                    )

                    print(
                        "Baseline Yaw:",
                        round(
                            baseline_yaw,
                            2
                        )
                    )

                    print(
                        "Baseline Roll:",
                        round(
                            baseline_roll,
                            2
                        )
                    )

                    print()

                    print(
                        "Baseline Gaze X:",
                        round(
                            baseline_gaze_x,
                            3
                        )
                    )

                    print(
                        "Baseline Gaze Y:",
                        round(
                            baseline_gaze_y,
                            3
                        )
                    )

                    print()

                    print(
                        "Monitoring started."
                    )

                    print()


            # =================================================
            # NORMAL MONITORING MODE
            # =================================================

            else:


                # =============================================
                # PERSONALIZED HEAD POSE
                # =============================================

                if raw_head_pose is not None:

                    head_pose = classify_head_pose(

                        raw_head_pose,

                        baseline_pitch,

                        baseline_yaw,

                        baseline_roll
                    )


                    pitch = (
                        head_pose[
                            "pitch"
                        ]
                    )

                    yaw = (
                        head_pose[
                            "yaw"
                        ]
                    )

                    roll = (
                        head_pose[
                            "roll"
                        ]
                    )

                    head_direction = (
                        head_pose[
                            "direction"
                        ]
                    )


                else:

                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0

                    head_direction = (
                        "UNKNOWN"
                    )


                # =============================================
                # PERSONALIZED GAZE
                # =============================================

                (
                    gaze_direction,
                    gaze_delta_x,
                    gaze_delta_y
                ) = classify_gaze(

                    gaze_data,

                    baseline_gaze_x,

                    baseline_gaze_y
                )


                # =============================================
                # EYE CLOSURE
                # =============================================

                if (
                    ear <
                    adaptive_ear_threshold
                ):


                    if not eyes_closed:

                        eyes_closed = True

                        eye_close_start_time = (
                            current_time
                        )

                        drowsiness_alarm_triggered = (
                            False
                        )


                    if eye_close_start_time is not None:

                        current_eye_closed_duration = (
                            current_time -
                            eye_close_start_time
                        )


                else:


                    if eyes_closed:


                        if eye_close_start_time is not None:

                            closure_duration = (
                                current_time -
                                eye_close_start_time
                            )

                        else:

                            closure_duration = 0.0


                        last_blink_duration = (
                            closure_duration
                        )


                        # -------------------------------------
                        # NORMAL BLINK
                        # -------------------------------------

                        if (
                            MIN_BLINK_DURATION
                            <= closure_duration
                            <= MAX_BLINK_DURATION
                        ):

                            blink_count += 1

                            blink_timestamps.append(
                                current_time
                            )


                        # -------------------------------------
                        # LONG CLOSURE
                        # -------------------------------------

                        elif (
                            closure_duration >
                            MAX_BLINK_DURATION
                        ):

                            long_eye_closure_count += 1


                    eyes_closed = False

                    eye_close_start_time = None

                    current_eye_closed_duration = 0.0

                    drowsiness_alarm_triggered = False


                # =============================================
                # BLINK RATE
                # =============================================

                while (
                    blink_timestamps
                    and
                    current_time -
                    blink_timestamps[0] >
                    60.0
                ):

                    blink_timestamps.popleft()


                if monitoring_start_time is not None:

                    observation_duration = (
                        current_time -
                        monitoring_start_time
                    )

                else:

                    observation_duration = 0.0


                effective_window = min(
                    60.0,
                    max(
                        observation_duration,
                        1.0
                    )
                )


                blink_rate = (
                    len(blink_timestamps) /
                    effective_window
                ) * 60.0


                # =============================================
                # DROWSINESS WARNING
                # =============================================

                if (
                    eyes_closed
                    and
                    current_eye_closed_duration >=
                    EYE_CLOSED_ALERT_SECONDS
                ):


                    cv2.putText(

                        frame,

                        "DROWSINESS ALERT!",

                        (20, 40),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.8,

                        (0, 0, 255),

                        2
                    )


                    if not drowsiness_alarm_triggered:

                        trigger_alarm(
                            alarm_path
                        )

                        drowsiness_alarm_triggered = (
                            True
                        )


                # =============================================
                # YAWN DETECTION
                # =============================================

                if (
                    mar >
                    adaptive_mar_threshold
                ):


                    if not mouth_open:

                        mouth_open = True

                        mouth_open_start_time = (
                            current_time
                        )

                        yawn_alarm_triggered = (
                            False
                        )


                    if mouth_open_start_time is not None:

                        current_mouth_open_duration = (
                            current_time -
                            mouth_open_start_time
                        )


                else:


                    if mouth_open:


                        if mouth_open_start_time is not None:

                            mouth_duration = (
                                current_time -
                                mouth_open_start_time
                            )

                        else:

                            mouth_duration = 0.0


                        if (
                            mouth_duration >=
                            MIN_YAWN_DURATION
                        ):

                            yawn_count += 1

                            last_yawn_duration = (
                                mouth_duration
                            )


                    mouth_open = False

                    mouth_open_start_time = None

                    current_mouth_open_duration = 0.0

                    yawn_alarm_triggered = False


                # =============================================
                # YAWN WARNING
                # =============================================

                if (
                    mouth_open
                    and
                    current_mouth_open_duration >=
                    YAWN_ALERT_SECONDS
                ):


                    cv2.putText(

                        frame,

                        "YAWN ALERT!",

                        (20, 75),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.8,

                        (0, 0, 255),

                        2
                    )


                    if not yawn_alarm_triggered:

                        trigger_alarm(
                            alarm_path
                        )

                        yawn_alarm_triggered = (
                            True
                        )


                # =============================================
                # DISTRACTION / LOOK-AWAY LOGIC
                # =============================================

                head_away = (

                    abs(yaw) >
                    HEAD_YAW_THRESHOLD

                    or

                    abs(pitch) >
                    HEAD_PITCH_THRESHOLD
                )


                gaze_away = (

                    gaze_direction
                    !=
                    "CENTER"
                )


                # Either strong head deviation or strong
                # personalized gaze deviation can indicate
                # looking away.

                currently_looking_away = (

                    head_away
                    or
                    gaze_away
                )


                if currently_looking_away:


                    if not looking_away:

                        looking_away = True

                        look_away_start_time = (
                            current_time
                        )


                    if look_away_start_time is not None:

                        current_look_away_duration = (
                            current_time -
                            look_away_start_time
                        )


                else:

                    looking_away = False

                    look_away_start_time = None

                    current_look_away_duration = 0.0


                # =============================================
                # DISTRACTION WARNING
                # =============================================

                if (
                    looking_away
                    and
                    current_look_away_duration >=
                    LOOK_AWAY_ALERT_SECONDS
                ):


                    cv2.putText(

                        frame,

                        "LOOKING AWAY!",

                        (20, 110),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.75,

                        (0, 0, 255),

                        2
                    )


                # =============================================
                # LEFT-SIDE INFORMATION
                # =============================================

                cv2.putText(

                    frame,

                    "Blinks: {}".format(
                        blink_count
                    ),

                    (20, 150),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Blink Rate: {:.1f}/min".format(
                        blink_rate
                    ),

                    (20, 175),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Last Blink: {:.2f}s".format(
                        last_blink_duration
                    ),

                    (20, 200),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Eye Closed: {:.2f}s".format(
                        current_eye_closed_duration
                    ),

                    (20, 225),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Long Closures: {}".format(
                        long_eye_closure_count
                    ),

                    (20, 250),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Yawns: {}".format(
                        yawn_count
                    ),

                    (20, 275),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Mouth Open: {:.2f}s".format(
                        current_mouth_open_duration
                    ),

                    (20, 300),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Away Time: {:.2f}s".format(
                        current_look_away_duration
                    ),

                    (20, 325),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.52,

                    (255, 255, 255),

                    1
                )


                # =============================================
                # RIGHT-SIDE INFORMATION
                # =============================================

                cv2.putText(

                    frame,

                    "EAR: {:.3f}".format(
                        ear
                    ),

                    (430, 25),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    (255, 255, 0),

                    1
                )


                cv2.putText(

                    frame,

                    "EAR T: {:.3f}".format(
                        adaptive_ear_threshold
                    ),

                    (430, 48),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (255, 255, 0),

                    1
                )


                cv2.putText(

                    frame,

                    "MAR: {:.3f}".format(
                        mar
                    ),

                    (430, 72),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    (255, 255, 0),

                    1
                )


                cv2.putText(

                    frame,

                    "MAR T: {:.3f}".format(
                        adaptive_mar_threshold
                    ),

                    (430, 95),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (255, 255, 0),

                    1
                )


                cv2.putText(

                    frame,

                    "Pitch D: {:.1f}".format(
                        pitch
                    ),

                    (430, 125),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Yaw D: {:.1f}".format(
                        yaw
                    ),

                    (430, 148),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Roll D: {:.1f}".format(
                        roll
                    ),

                    (430, 171),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Head: {}".format(
                        head_direction
                    ),

                    (430, 194),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 255, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Gaze: {}".format(
                        gaze_direction
                    ),

                    (430, 220),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (255, 0, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Gaze dX: {:.2f}".format(
                        gaze_delta_x
                    ),

                    (430, 243),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    (255, 0, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "Gaze dY: {:.2f}".format(
                        gaze_delta_y
                    ),

                    (430, 265),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.43,

                    (255, 0, 255),

                    1
                )


                cv2.putText(

                    frame,

                    "FPS: {:.1f}".format(
                        fps
                    ),

                    (520, 460),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    (255, 255, 255),

                    1
                )


                # =============================================
                # STATUS
                # =============================================

                if (
                    not eyes_closed
                    and
                    not looking_away
                    and
                    not mouth_open
                ):


                    cv2.putText(

                        frame,

                        "STATUS: NORMAL",

                        (20, 370),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (0, 255, 0),

                        2
                    )


        # ====================================================
        # NO FACE
        # ====================================================

        else:


            cv2.putText(

                frame,

                "NO FACE DETECTED",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 165, 255),

                2
            )


            # During calibration, require a continuous,
            # clearly visible face.

            if not calibration_complete:

                calibration_start_time = None

                calibration_ear_values.clear()
                calibration_mar_values.clear()

                calibration_pitch_values.clear()
                calibration_yaw_values.clear()
                calibration_roll_values.clear()

                calibration_gaze_x_values.clear()
                calibration_gaze_y_values.clear()


                cv2.putText(

                    frame,

                    "Calibration waiting for face...",

                    (20, 75),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 255),

                    1
                )


            # Reset temporary event states.
            # Do not reset accumulated blink/yawn totals.

            eyes_closed = False
            eye_close_start_time = None
            current_eye_closed_duration = 0.0

            mouth_open = False
            mouth_open_start_time = None
            current_mouth_open_duration = 0.0

            looking_away = False
            look_away_start_time = None
            current_look_away_duration = 0.0

            drowsiness_alarm_triggered = False
            yawn_alarm_triggered = False


        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(

            "DriveGuard AI - Personalized Driver Monitoring",

            frame
        )


        # ====================================================
        # KEY INPUT
        # ====================================================

        key = (
            cv2.waitKey(1) &
            0xFF
        )


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print()

    print(
        "-> Stopping DriveGuard AI..."
    )


    camera.release()

    cv2.destroyAllWindows()

    face_mesh.close()


    print(
        "-> System stopped successfully."
    )


    print()

    print(
        "FINAL SESSION STATISTICS"
    )

    print(
        "------------------------"
    )

    print(
        "Total blinks:",
        blink_count
    )

    print(
        "Long eye closures:",
        long_eye_closure_count
    )

    print(
        "Total yawns:",
        yawn_count
    )


    if calibration_complete:

        print()

        print(
            "PERSONALIZED BASELINE"
        )

        print(
            "---------------------"
        )

        print(
            "Baseline EAR:",
            round(
                baseline_ear,
                3
            )
        )

        print(
            "EAR threshold:",
            round(
                adaptive_ear_threshold,
                3
            )
        )

        print(
            "Baseline MAR:",
            round(
                baseline_mar,
                3
            )
        )

        print(
            "MAR threshold:",
            round(
                adaptive_mar_threshold,
                3
            )
        )

        print(
            "Baseline Pitch:",
            round(
                baseline_pitch,
                2
            )
        )

        print(
            "Baseline Yaw:",
            round(
                baseline_yaw,
                2
            )
        )

        print(
            "Baseline Gaze X:",
            round(
                baseline_gaze_x,
                3
            )
        )

        print(
            "Baseline Gaze Y:",
            round(
                baseline_gaze_y,
                3
            )
        )