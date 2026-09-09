
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time
from threading import Thread, Lock
from playsound3 import playsound


# ============================================================
# REAL-TIME DRIVER DROWSINESS DETECTION AND DISTRACTION SYSTEM
# MediaPipe + OpenCV + NumPy + playsound3
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Eye Aspect Ratio threshold.
# Lower EAR means the eyes are more closed.
EYE_AR_THRESH = 0.30

# Number of consecutive frames eyes must remain closed
# before the drowsiness alert is triggered.
EYE_AR_CONSEC_FRAMES = 40

# Mouth Aspect Ratio threshold for detecting yawning.
MAR_THRESH = 0.55

# Number of consecutive frames mouth must remain open
# before the yawn alert is triggered.
YAWN_CONSEC_FRAMES = 10


# ============================================================
# GLOBAL ALARM VARIABLES
# ============================================================

alarm_lock = Lock()
alarm_playing = False


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# MEDIAPIPE FACIAL LANDMARK INDICES
# ============================================================

# Left eye landmarks
LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

# Right eye landmarks
RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]

# Mouth landmarks
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

MOUTH_LEFT = 61
MOUTH_RIGHT = 291


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Real-Time Drowsiness Detection using MediaPipe"
)

parser.add_argument(
    "-w",
    "--webcam",
    type=int,
    default=0,
    help="Webcam index. Default is 0."
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
# ALARM FUNCTION
# ============================================================

def play_alarm(path):
    """
    Plays the alarm sound in a separate thread.
    """

    global alarm_playing

    with alarm_lock:

        if alarm_playing:
            return

        alarm_playing = True

    try:

        print("🔊 DROWSINESS/YAWN ALARM!")

        playsound(path)

    except Exception as error:

        print("Alarm error:", error)

    finally:

        with alarm_lock:
            alarm_playing = False


def trigger_alarm(path):
    """
    Starts the alarm without freezing the webcam.
    """

    global alarm_playing

    with alarm_lock:

        if alarm_playing:
            return

    alarm_thread = Thread(
        target=play_alarm,
        args=(path,),
        daemon=True
    )

    alarm_thread.start()


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(point1, point2):
    """
    Calculates Euclidean distance using NumPy.
    """

    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)

    return np.linalg.norm(p1 - p2)


# ============================================================
# GET PIXEL COORDINATES FROM MEDIAPIPE LANDMARK
# ============================================================

def landmark_to_point(landmark, width, height):

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)


# ============================================================
# EYE ASPECT RATIO (EAR)
# ============================================================

def calculate_ear(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:

        point = landmark_to_point(
            landmarks[index],
            width,
            height
        )

        points.append(point)

    # Eye landmark structure:
    #
    #       P2       P3
    #
    # P1 ---------------- P4
    #
    #       P6       P5

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

        return 0.0, points

    ear = (
        vertical_distance_1 +
        vertical_distance_2
    ) / (2.0 * horizontal_distance)

    return ear, points


# ============================================================
# MOUTH ASPECT RATIO (MAR)
# ============================================================

def calculate_mar(landmarks, width, height):
    """
    Calculates Mouth Aspect Ratio (MAR).

    Uses multiple vertical distances inside the mouth
    divided by the horizontal mouth width.
    """

    # Vertical mouth landmarks
    top1 = landmark_to_point(landmarks[13], width, height)
    bottom1 = landmark_to_point(landmarks[14], width, height)

    top2 = landmark_to_point(landmarks[82], width, height)
    bottom2 = landmark_to_point(landmarks[87], width, height)

    top3 = landmark_to_point(landmarks[312], width, height)
    bottom3 = landmark_to_point(landmarks[317], width, height)

    # Horizontal mouth landmarks
    left = landmark_to_point(
        landmarks[61],
        width,
        height
    )

    right = landmark_to_point(
        landmarks[291],
        width,
        height
    )

    # Vertical distances
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

    # Horizontal mouth width
    horizontal = calculate_distance(
        left,
        right
    )

    if horizontal == 0:
        return 0.0, (
            top1,
            bottom1,
            left,
            right
        )

    # Average vertical mouth opening
    average_vertical = (
        vertical1 +
        vertical2 +
        vertical3
    ) / 3.0

    # Mouth Aspect Ratio
    mar = average_vertical / horizontal

    return mar, (
        top1,
        bottom1,
        left,
        right
    )
# ============================================================
# DRAW EYE LANDMARKS
# ============================================================

def draw_eye(frame, points):

    points_array = np.array(
        points,
        dtype=np.int32
    )

    hull = cv2.convexHull(points_array)

    cv2.polylines(
        frame,
        [hull],
        True,
        (0, 255, 0),
        1
    )


# ============================================================
# DRAW MOUTH LANDMARKS
# ============================================================

def draw_mouth(frame, mouth_points):

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
# CHECK ALARM FILE
# ============================================================

alarm_path = args.alarm

if not os.path.exists(alarm_path):

    print()
    print("ERROR: Alarm file not found!")
    print(
        "Expected file:",
        os.path.abspath(alarm_path)
    )
    print()

    raise SystemExit(1)


print(
    "-> Alarm file:",
    os.path.abspath(alarm_path)
)


# ============================================================
# START MEDIAPIPE
# ============================================================

print("-> Loading MediaPipe Face Mesh...")


face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# START WEBCAM
# ============================================================

print("-> Starting Video Stream...")

camera = cv2.VideoCapture(
    args.webcam
)


if not camera.isOpened():

    print()
    print(
        "ERROR: Could not open webcam."
    )

    print(
        "Try another webcam index, for example:"
    )

    print(
        "python drowsiness_yawn.py --webcam 1"
    )

    print()

    face_mesh.close()

    raise SystemExit(1)


# Set webcam resolution
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# DETECTION COUNTERS
# ============================================================

eye_counter = 0

yawn_counter = 0


print("-> System started successfully!")
print()
print("Press 'q' to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # Read webcam frame
        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break


        # Mirror the webcam
        frame = cv2.flip(
            frame,
            1
        )


        # Get frame dimensions
        height, width = frame.shape[:2]


        # ====================================================
        # CONVERT BGR TO RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE FACE PROCESSING
        # ====================================================

        results = face_mesh.process(
            rgb_frame
        )


        # ====================================================
        # IF FACE IS DETECTED
        # ====================================================

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            landmarks = face.landmark


            # =================================================
            # CALCULATE LEFT EAR
            # =================================================

            left_ear, left_eye_points = calculate_ear(

                landmarks,

                LEFT_EYE,

                width,

                height
            )


            # =================================================
            # CALCULATE RIGHT EAR
            # =================================================

            right_ear, right_eye_points = calculate_ear(

                landmarks,

                RIGHT_EYE,

                width,

                height
            )


            # =================================================
            # FINAL EAR
            # =================================================

            ear = (
                left_ear +
                right_ear
            ) / 2.0


            # =================================================
            # DRAW EYES
            # =================================================

            draw_eye(
                frame,
                left_eye_points
            )

            draw_eye(
                frame,
                right_eye_points
            )


            # =================================================
            # CALCULATE MOUTH RATIO
            # =================================================

            mar, mouth_points = calculate_mar(

                landmarks,

                width,

                height
            )


            # =================================================
            # DRAW MOUTH
            # =================================================

            draw_mouth(
                frame,
                mouth_points
            )


            # =================================================
            # DROWSINESS DETECTION
            # =================================================

            if ear < EYE_AR_THRESH:

                eye_counter += 1

            else:

                eye_counter = 0


            # =================================================
            # DROWSINESS ALERT
            # =================================================

            if eye_counter >= EYE_AR_CONSEC_FRAMES:

                cv2.putText(

                    frame,

                    "DROWSINESS ALERT!",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 0, 255),

                    2
                )


                trigger_alarm(
                    alarm_path
                )


            # =================================================
            # YAWN DETECTION
            # =================================================

            if mar > MAR_THRESH:

                yawn_counter += 1

            else:

                yawn_counter = 0


            # =================================================
            # YAWN ALERT
            # =================================================

            if yawn_counter >= YAWN_CONSEC_FRAMES:

                cv2.putText(

                    frame,

                    "YAWN ALERT!",

                    (20, 75),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 0, 255),

                    2
                )


                trigger_alarm(
                    alarm_path
                )


            # =================================================
            # DISPLAY EAR
            # =================================================

            cv2.putText(

                frame,

                "EAR: {:.2f}".format(ear),

                (450, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2
            )


            # =================================================
            # DISPLAY MAR
            # =================================================

            cv2.putText(

                frame,

                "MAR: {:.2f}".format(mar),

                (450, 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2
            )


            # =================================================
            # DISPLAY COUNTERS
            # =================================================

            cv2.putText(

                frame,

                "Eye Frames: {}".format(
                    eye_counter
                ),

                (20, 110),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                1
            )


            cv2.putText(

                frame,

                "Yawn Frames: {}".format(
                    yawn_counter
                ),

                (20, 135),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                1
            )


        # ====================================================
        # NO FACE DETECTED
        # ====================================================

        else:

            eye_counter = 0

            yawn_counter = 0

            cv2.putText(

                frame,

                "NO FACE DETECTED",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 165, 255),

                2
            )


        # ====================================================
        # DISPLAY VIDEO
        # ====================================================

        cv2.imshow(

            "Real-Time Drowsiness Detection",

            frame
        )


        # ====================================================
        # KEYBOARD INPUT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # Press Q to quit
        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print()
    print("-> Stopping system...")

    camera.release()

    cv2.destroyAllWindows()

    face_mesh.close()

    print("-> System stopped successfully.")

