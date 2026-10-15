# Real-Time Drowsiness & Distraction Detection System (DriveGuard AI)

A multi-modal, scientific Driver Monitoring System (DMS) built using computer vision, Google MediaPipe Face Mesh, 3D Head Pose Estimation, PERCLOS, and multi-feature fatigue score fusion.

---

## Motivation & Scientific Background
According to the National Highway Traffic Safety Administration (NHTSA), tens of thousands of crashes every year are caused by drowsy or distracted drivers. Academic literature indexed on Google Scholar, IEEE, and transportation safety journals emphasizes that **single-metric thresholding (e.g., measuring Eye Aspect Ratio alone) is insufficient** due to individual anatomical differences, glasses, and natural blinks.

DriveGuard AI implements a **multi-modal fusion system** based on psychophysiological safety standards:
1. **PERCLOS (Percentage of Eyelid Closure over Time)**: Validated by the NHTSA and FHWA as the most reliable visual indicator of slow eyelid closures and fatigue onset (calculated over a rolling 60-second window).
2. **Blink Dynamics**: Analyzes blink frequency (blinks/min) and average blink duration ($t > 0.40\text{s}$ indicates micro-sleep / fatigue).
3. **Mouth Aspect Ratio (MAR) & Yawning Frequency**: Distinguishes normal speech from sustained yawning.
4. **3D Head Pose Estimation (Perspective-n-Point / SolvePnP)**: Continuously tracks Pitch, Yaw, and Roll Euler angles. Detects cervical muscle tone loss (**head droop / micro-sleep nod**).
5. **Iris Tracking & Gaze Classification**: Uses MediaPipe's sub-pupillary iris refinement to detect off-road glance distractions.
6. **Adaptive Startup Calibration**: Collects median baselines for the first 5 seconds to adjust personalized EAR/MAR thresholds for each driver.
7. **Composite Fatigue Index ($0 - 100\%$) & 4-Tier Risk Hierarchy**:
   - `NORMAL`: Driver attentive ($\text{Fatigue} < 35\%$).
   - `ADVISORY`: Mild fatigue indicators ($\text{Fatigue } 35 - 60\%$).
   - `WARNING`: High fatigue risk ($\text{Fatigue } 60 - 80\%$).
   - `CRITICAL`: Micro-sleep, sustained closure $\ge 1.5\text{s}$, or sustained yawn $\ge 1.2\text{s}$ triggering immediate audio alarm.

---

## Built With

* **[OpenCV](https://opencv.org/)** (`opencv-python 4.10.0+`): Real-time video capture, matrix transformations, Perspective-n-Point (SolvePnP) 3D head pose estimation, and HUD rendering.
* **[MediaPipe Face Mesh](https://developers.google.com/mediapipe)** (`mediapipe 0.10.21+`): 468+ high-density 3D facial landmarks with sub-millimeter iris contour refinement (`refine_landmarks=True`).
* **[NumPy](https://numpy.org/)**: Fast vectorized vector algebra, median calibration, and Euclidean distance computation.
* **[playsound3](https://github.com/gregn/playsound3)**: Non-blocking, cross-platform audio playback running on asynchronous daemon threads.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System.git
cd Real-Time-Drowsiness-Detection-System
```

### 2. Set Up Virtual Environment (Python 3.9+)
```bash
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
# or: venv\Scripts\activate # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Quick Start (Recommended)
```bash
./run.sh --webcam 0 --alarm Alert.wav
```
*(Or simply `python3 drowsiness_yawn.py --webcam 0 --alarm Alert.wav` — the system automatically detects the virtual environment)*

### Or with Virtual Environment Activated
```bash
source venv/bin/activate
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
```

### Options & Command-Line Arguments
```text
options:
  -h, --help            Show help message and exit
  -w WEBCAM, --webcam WEBCAM
                        Webcam device index (default: 0)
  -v VIDEO, --video VIDEO
                        Path to input video file (for offline testing/benchmarking)
  -a ALARM, --alarm ALARM
                        Path to alarm WAV file (default: Alert.wav)
  -l LOG, --log LOG     Path to output CSV file for diagnostic telemetry logging
  -c CALIBRATION, --calibration CALIBRATION
                        Startup baseline calibration duration in seconds (default: 5.0)
  --alarm-distraction   Sound buzzer when driver looks away off-road for > 2.0s
  --no-gui              Run headless without OpenCV window (useful for servers/evaluation)
```

### Example: Log Telemetry to CSV
```bash
python drowsiness_yawn.py -w 0 -a Alert.wav -l session_telemetry.csv
```

---

## Behavioural Feature Data Collection (for GRU Deep Learning)

DriveGuard AI includes an automatic behavioural feature data collection system designed for training **Gated Recurrent Unit (GRU)** and LSTM temporal sequence models.

### Target GRU Training Configuration
* **Sampling Frequency:** Exactly **10 Hz** ($\Delta t = 0.100\text{s}$ governed strictly by clock timestamps).
* **Sequence Window:** **5.0 seconds** ($50\text{ timesteps}$ per window).
* **Isolation:** Sequences are partitioned strictly by `participant_id` and `session_id` to prevent cross-subject leakage.

### How to Start Data Collection

#### 1. Interactive Collection Mode
Start the application with custom participant and session identifiers:
```bash
./run.sh --participant P001 --session S001
```
Use keyboard shortcuts in the OpenCV window to control collection dynamically:
* **`r` / `R`**: Toggle Recording **ON / OFF** (flushes to CSV immediately).
* **`1` – `8`**: Select Behavioural Activity label on the fly:
  * `1`: `normal_alert` (Normal/alert driving posture)
  * `2`: `normal_blinking` (Natural blinking)
  * `3`: `prolonged_eye_closure` (Simulated micro-sleep / slow closure)
  * `4`: `yawning` (Oral aperture / yawning)
  * `5`: `looking_left` (Left mirror / off-road glance)
  * `6`: `looking_right` (Right mirror / off-road glance)
  * `7`: `looking_down` (Phone / center console distraction)
  * `8`: `head_nodding` (Involuntary micro-sleep head droop)
* **`p` / `P`**: Increment/cycle participant ID (`P001` $\to$ `P002` $\to$ ...).
* **`q`**: Quit safely (auto-flushes and closes all CSV files).

#### 2. Auto-Record on Launch
```bash
./run.sh --participant P001 --session S001 --activity normal_alert --record
```

#### 3. Where Files Are Saved
Raw recordings are automatically written to:
```text
data/raw/{participant_id}_{session_id}_{activity_label}_{timestamp}.csv
```

### Dataset CSV Schema
Each row represents a 10 Hz timestep with numerical features in consistent order:

| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | Float | Monotonic session timestamp (seconds) |
| `participant_id` | String | Unique driver / participant code (e.g., `P001`) |
| `session_id` | String | Recording session identifier (e.g., `S001`) |
| `activity_label` | String | Active behavioural activity (`normal_alert`, `yawning`, etc.) |
| `face_detected` | Integer | `1` if face is tracked, `0` if face is occluded/absent |
| `is_calibrated` | Integer | `1` if post-calibration operational monitoring, `0` if baseline calibration |
| `ear` | Float | Eye Aspect Ratio |
| `mar` | Float | Mouth Aspect Ratio |
| `perclos` | Float | Rolling percentage of eyelid closure (0.0 – 1.0) |
| `blink_rate` | Float | Rolling blinks per minute |
| `blink_duration` | Float | Duration of last completed blink (seconds) |
| `eye_closure_duration`| Float | Duration eyes currently closed (seconds) |
| `yawn_rate` | Float | Rolling yawns per minute |
| `yawn_duration` | Float | Duration mouth currently open (seconds) |
| `head_pitch` | Float | Head Pitch angle in degrees |
| `head_yaw` | Float | Head Yaw angle in degrees |
| `head_roll` | Float | Head Roll angle in degrees |
| `gaze_x` | Float | Sub-pupillary iris horizontal position (0.0 – 1.0) |
| `gaze_y` | Float | Sub-pupillary iris vertical position (0.0 – 1.0) |
| `looking_away_duration`| Float| Off-road gaze / head deviation duration (seconds) |
| `head_nodding` | Integer | `1` if head nod / droop is active, `0` otherwise |
| `fatigue_label` | String | Ground-truth fatigue label (`unknown` unless labeled) |
| `distraction_label` | String | Ground-truth distraction label (`unknown` unless labeled) |

*Note: Missing or invalid face detections explicitly record `face_detected = 0` with `NaN` numerical feature placeholders to preserve time-series continuity for GRU masking.*

---

## Validating Datasets for GRU Training

DriveGuard AI includes `validate_dataset.py` to inspect data integrity before model training:

```bash
python validate_dataset.py data/raw/
```
The validator automatically verifies:
1. Column schema conformance against the standard GRU specification.
2. Strict monotonic timestamp sequencing and 10 Hz interval consistency ($\Delta t \approx 0.10\text{s}$).
3. Biological/geometric bounds for all physical features.
4. Calculation of ready **50-timestep (5.0s)** sequence chunks without crossing participant/session boundaries.

---

## Scientific Evaluation & Metrics

### 1. Eye Aspect Ratio (EAR)
$$\text{EAR} = \frac{\|p_1 - p_5\| + \|p_2 - p_4\|}{2 \cdot \|p_0 - p_3\|}$$

### 2. PERCLOS (Percentage of Eye Closure)
$$\text{PERCLOS} = \left(\frac{\text{Total eye closure time in window}}{\text{Total observation window (60s)}}\right) \times 100\%$$
- Standard NHTSA benchmark: $\text{PERCLOS} \ge 15\%$ indicates moderate drowsiness; $\ge 25 - 30\%$ indicates critical fatigue.

### 3. Head Nod & Droop Detection
By decomposing the 3D rotation matrix via `cv2.solvePnP` and `cv2.RQDecomp3x3`:
$$\Delta\text{Pitch} = (\text{Pitch}_{\text{current}} - \text{Pitch}_{\text{baseline}})$$
Downward head tilt exceeding $18^\circ$ sustained for $> 0.5\text{s}$ (or co-occurring with eye closure) triggers a **Head Nod Alert**.

---

## Academic References

1. **Wierwille, W. W., et al.** (1994). *Evaluation of Techniques for Detecting Driver Drowsiness under Simulated Conditions.* National Highway Traffic Safety Administration (NHTSA).
2. **Soukupová, T., & Čech, J.** (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop (CVWW).
3. **Manu, B. N.** (2016). *Facial Features Monitoring for Real Time Drowsiness Detection.* 12th International Conference on Innovations in Information Technology (IIT).
4. **Rahman, A.** (2015). *Real Time Drowsiness Detection using Eye Blink Monitoring.* National Software Engineering Conference (NSEC).
5. **Lugaresi, C., et al.** (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.

---

## Original Author & Credits
* Original Author: **Anshuman Srivastava**
* Modernized by: **Sainath** (MediaPipe FaceMesh, 3D SolvePnP Head Pose, PERCLOS, Multi-Modal Fusion, and Adaptive Calibration).
