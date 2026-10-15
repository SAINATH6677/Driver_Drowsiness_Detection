"""
================================================================================
DRIVEGUARD AI - BEHAVIOURAL FEATURE DATA LOGGER FOR TEMPORAL GRU TRAINING
================================================================================

This module provides rate-controlled (target 10 Hz) behavioural feature logging
for driver monitoring sessions. Logged data follows a consistent 23-column
numerical schema suitable for windowed sequence extraction (e.g., 50 timesteps /
5 seconds per sequence) for Gated Recurrent Unit (GRU) temporal deep learning.

Key Guarantees & Features:
- Preserves the standard 23-column research CSV schema.
- Accompanies every CSV recording with a sidecar JSON metadata manifest containing
  UTC start/end times, schema/software version, and activity transition boundaries.
- Rate-throttles data sampling to a nominal 10 Hz interval (0.10s) using monotonic
  clock timestamps, measuring the actual achieved sampling rate empirically.
- Strictly records missing or occluded face frames as explicit NaN values for all
  geometric and biological metrics (including head_nodding).
- Distinguishes baseline calibration frames (is_calibrated=0) from operational
  monitoring frames (is_calibrated=1).
- Guarantees complete participant and session segregation: changing participant ID
  automatically closes prior files and begins a new segregated recording.
================================================================================
"""

import csv
from datetime import datetime, timezone
import json
import os
import time
from typing import Optional, Dict, Any, List

# Software & Schema Versioning
SOFTWARE_VERSION = "DriveGuard AI 2.0"
SCHEMA_VERSION = "1.2.0"

# Standard Behavioural Activities for Data Collection
SUPPORTED_ACTIVITIES = [
    "normal_alert",           # Key 1: Normal alert baseline behavior
    "normal_blinking",        # Key 2: Regular natural blinking
    "prolonged_eye_closure",  # Key 3: Extended eye closure / micro-sleep simulation
    "yawning",                # Key 4: Oral aperture / yawning simulation
    "looking_left",           # Key 5: Left mirror / off-road glance
    "looking_right",          # Key 6: Right mirror / off-road glance
    "looking_down",           # Key 7: Downward gaze (e.g., dashboard / phone)
    "head_nodding",           # Key 8: Involuntary cervical drooping / nodding
]

# Standard CSV Header Schema (23 Columns)
FEATURE_SCHEMA = [
    "timestamp",
    "participant_id",
    "session_id",
    "activity_label",
    "face_detected",
    "is_calibrated",
    "ear",
    "mar",
    "perclos",
    "blink_rate",
    "blink_duration",
    "eye_closure_duration",
    "yawn_rate",
    "yawn_duration",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_x",
    "gaze_y",
    "looking_away_duration",
    "head_nodding",
    "fatigue_label",
    "distraction_label",
]


class FeatureLogger:
    """
    Manages structured, rate-controlled CSV data logging for GRU sequence training.
    
    Guarantees:
    - Target sampling frequency (nominal 10 Hz / 0.10s dt) rate-throttled by monotonic timestamps.
    - Sidecar JSON metadata capturing UTC start/end times, schema version, and activity boundaries.
    - Explicit handling of valid vs missing face detections with NaNs (including head_nodding).
    - Empirical measurement of actual achieved sampling rate and interval irregularities.
    - Segregated activity tracking vs ground truth labels.
    - Automated directory creation, non-overwriting filenames, and clean participant isolation.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        participant_id: str = "P001",
        session_id: Optional[str] = None,
        sampling_rate: float = 10.0,
        activity_label: str = "normal_alert",
        fatigue_label: str = "unknown",
        distraction_label: str = "unknown",
    ):
        self.data_dir = data_dir
        self.participant_id = participant_id
        self.session_id = session_id or datetime.now().strftime("S%Y%m%d_%H%M%S")
        self.sampling_rate = max(1.0, float(sampling_rate))
        self.sample_interval = 1.0 / self.sampling_rate

        self.activity_label = activity_label if activity_label in SUPPORTED_ACTIVITIES else "normal_alert"
        self.fatigue_label = fatigue_label
        self.distraction_label = distraction_label

        # Recording state
        self.is_recording = False
        self.file_handle = None
        self.csv_writer = None
        self.current_filepath = None
        self.metadata_filepath = None
        self.sample_count = 0
        self.first_sample_time = None
        self.last_sample_time = 0.0
        self.recording_start_time = 0.0
        self.irregular_intervals_count = 0
        self.utc_start_iso = None
        self.utc_start_epoch = None
        self.activity_transitions: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        os.makedirs(self.data_dir, exist_ok=True)

    def set_activity(self, activity: str) -> bool:
        """Sets the active recording activity label and records transition metadata if recording."""
        if activity in SUPPORTED_ACTIVITIES:
            if activity != self.activity_label:
                self.activity_label = activity
                if self.is_recording:
                    transition = {
                        "activity_label": activity,
                        "sample_index": self.sample_count,
                        "timestamp": round(float(self.last_sample_time), 4) if self.last_sample_time > 0 else None,
                    }
                    self.activity_transitions.append(transition)
                    self._save_metadata()
            return True
        return False

    def set_activity_by_index(self, index: int) -> bool:
        """Sets activity by 1-based index (1 through 8)."""
        if 1 <= index <= len(SUPPORTED_ACTIVITIES):
            return self.set_activity(SUPPORTED_ACTIVITIES[index - 1])
        return False

    def set_participant(self, participant_id: str):
        """Updates participant identifier without mixing data across participants."""
        clean_id = participant_id.strip()
        if clean_id and clean_id != self.participant_id:
            was_recording = self.is_recording
            if was_recording:
                self.stop_recording()
            self.participant_id = clean_id
            if was_recording:
                self.start_recording()

    def set_session(self, session_id: str):
        """Updates session identifier without mixing data across sessions."""
        clean_id = session_id.strip()
        if clean_id and clean_id != self.session_id:
            was_recording = self.is_recording
            if was_recording:
                self.stop_recording()
            self.session_id = clean_id
            if was_recording:
                self.start_recording()

    def _save_metadata(self):
        """Writes current metadata dictionary to sidecar JSON manifest."""
        if self.metadata_filepath:
            try:
                with open(self.metadata_filepath, mode="w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                print(f"[FeatureLogger Warning]: Error writing metadata sidecar: {e}")

    def start_recording(
        self,
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        activity_label: Optional[str] = None,
    ) -> str:
        """
        Starts a new recording session, initializes the CSV file and metadata sidecar JSON.
        Returns the path to the created CSV file.
        """
        if self.is_recording:
            self.stop_recording()

        if participant_id:
            clean_p = participant_id.strip()
            if clean_p:
                self.participant_id = clean_p
        if session_id:
            clean_s = session_id.strip()
            if clean_s:
                self.session_id = clean_s
        if activity_label and activity_label in SUPPORTED_ACTIVITIES:
            self.activity_label = activity_label

        now_utc = datetime.now(timezone.utc)
        self.utc_start_iso = now_utc.isoformat()
        self.utc_start_epoch = now_utc.timestamp()

        # Generate unique timestamped filename inside data/raw/
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.participant_id}_{self.session_id}_{self.activity_label}_{timestamp_str}"
        filename = f"{base_filename}.csv"
        self.current_filepath = os.path.join(self.data_dir, filename)
        collision_idx = 1
        while os.path.exists(self.current_filepath):
            filename = f"{base_filename}_{collision_idx}.csv"
            self.current_filepath = os.path.join(self.data_dir, filename)
            collision_idx += 1

        self.metadata_filepath = os.path.splitext(self.current_filepath)[0] + ".json"

        self.file_handle = open(self.current_filepath, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.file_handle)
        self.csv_writer.writerow(FEATURE_SCHEMA)
        self.file_handle.flush()

        self.is_recording = True
        self.sample_count = 0
        self.recording_start_time = time.time()
        self.first_sample_time = None
        self.last_sample_time = 0.0
        self.irregular_intervals_count = 0
        self.activity_transitions = [
            {
                "activity_label": self.activity_label,
                "sample_index": 0,
                "timestamp": None,
            }
        ]

        self.metadata = {
            "software_version": SOFTWARE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "initial_activity_label": self.activity_label,
            "target_sampling_rate_hz": self.sampling_rate,
            "nominal_sampling_interval_s": self.sample_interval,
            "timestamp_type": "monotonic_seconds (time.monotonic())",
            "utc_recording_start_time": self.utc_start_iso,
            "utc_recording_start_epoch": self.utc_start_epoch,
            "csv_filepath": self.current_filepath,
            "column_count": len(FEATURE_SCHEMA),
            "columns": FEATURE_SCHEMA,
            "activity_transitions": self.activity_transitions,
            "status": "recording",
        }
        self._save_metadata()

        print(f"\n[FeatureLogger]: Recording STARTED -> {self.current_filepath}")
        return self.current_filepath

    def stop_recording(self) -> Optional[str]:
        """Stops the current recording session, finalizes metadata sidecar, and flushes data to disk."""
        if not self.is_recording:
            return None

        saved_path = self.current_filepath
        count = self.sample_count
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0.0

        now_utc = datetime.now(timezone.utc)
        elapsed_monotonic = (self.last_sample_time - self.first_sample_time) if (self.first_sample_time and self.last_sample_time) else 0.0
        achieved_rate_hz = round((self.sample_count - 1) / elapsed_monotonic, 2) if (elapsed_monotonic > 0 and self.sample_count > 1) else (round(self.sample_count / duration, 2) if duration > 0 else 0.0)

        if self.file_handle:
            try:
                self.file_handle.flush()
                self.file_handle.close()
            except Exception as e:
                print(f"[FeatureLogger Warning]: Error closing file: {e}")

        # Finalize and persist sidecar metadata
        if self.metadata:
            self.metadata.update({
                "status": "completed",
                "utc_recording_end_time": now_utc.isoformat(),
                "utc_recording_end_epoch": now_utc.timestamp(),
                "duration_seconds": round(duration, 2),
                "elapsed_monotonic_seconds": round(elapsed_monotonic, 4),
                "total_samples": count,
                "achieved_sampling_rate_hz": achieved_rate_hz,
                "irregular_intervals_count": self.irregular_intervals_count,
                "activity_transitions": self.activity_transitions,
            })
            self._save_metadata()

        self.file_handle = None
        self.csv_writer = None
        self.is_recording = False

        print(f"[FeatureLogger]: Recording STOPPED. Saved {count} samples ({duration:.1f}s, achieved ~{achieved_rate_hz} Hz) to: {saved_path}")
        return saved_path

    def toggle_recording(self) -> bool:
        """Toggles recording state between ON and OFF."""
        if self.is_recording:
            self.stop_recording()
            return False
        else:
            self.start_recording()
            return True

    def log_frame(
        self,
        current_time: float,
        face_detected: bool,
        is_calibrated: bool = True,
        ear: Optional[float] = None,
        mar: Optional[float] = None,
        perclos: Optional[float] = None,
        blink_rate: Optional[float] = None,
        blink_duration: Optional[float] = None,
        eye_closure_duration: Optional[float] = None,
        yawn_rate: Optional[float] = None,
        yawn_duration: Optional[float] = None,
        head_pitch: Optional[float] = None,
        head_yaw: Optional[float] = None,
        head_roll: Optional[float] = None,
        gaze_x: Optional[float] = None,
        gaze_y: Optional[float] = None,
        looking_away_duration: Optional[float] = None,
        head_nodding: Optional[int] = None,
    ) -> bool:
        """
        Samples and records extracted behavioural features at the target sampling rate (nominal 10 Hz).
        
        Uses monotonic elapsed timestamps to govern rate throttling.
        If a face is not detected, safely records explicit NaN values for ALL biometric metrics
        (including head_nodding) with face_detected=0.
        Explicitly tracks calibration state (is_calibrated=1 vs 0) to prevent uncalibrated data mixing.
        
        Returns True if a sample was recorded on this invocation, False otherwise.
        """
        if not self.is_recording or self.csv_writer is None:
            return False

        # Strictly throttle sampling rate by timestamp (e.g., 0.1s for target 10 Hz) with 1ms float tolerance
        if self.last_sample_time > 0 and (current_time - self.last_sample_time) < (self.sample_interval - 0.001):
            return False

        if self.first_sample_time is None:
            self.first_sample_time = current_time
            if self.metadata and self.activity_transitions:
                self.activity_transitions[0]["timestamp"] = round(float(current_time), 4)

        if self.last_sample_time > 0:
            dt = current_time - self.last_sample_time
            if abs(dt - self.sample_interval) > 0.03:
                self.irregular_intervals_count += 1

        self.last_sample_time = current_time
        self.sample_count += 1

        sample_timestamp = round(float(current_time), 4)
        calib_val = 1 if is_calibrated else 0

        if face_detected:
            row = [
                sample_timestamp,
                self.participant_id,
                self.session_id,
                self.activity_label,
                1,
                calib_val,
                round(float(ear), 4) if ear is not None else float("nan"),
                round(float(mar), 4) if mar is not None else float("nan"),
                round(float(perclos), 4) if perclos is not None else float("nan"),
                round(float(blink_rate), 2) if blink_rate is not None else float("nan"),
                round(float(blink_duration), 4) if blink_duration is not None else float("nan"),
                round(float(eye_closure_duration), 4) if eye_closure_duration is not None else float("nan"),
                round(float(yawn_rate), 2) if yawn_rate is not None else float("nan"),
                round(float(yawn_duration), 4) if yawn_duration is not None else float("nan"),
                round(float(head_pitch), 2) if head_pitch is not None else float("nan"),
                round(float(head_yaw), 2) if head_yaw is not None else float("nan"),
                round(float(head_roll), 2) if head_roll is not None else float("nan"),
                round(float(gaze_x), 4) if gaze_x is not None else float("nan"),
                round(float(gaze_y), 4) if gaze_y is not None else float("nan"),
                round(float(looking_away_duration), 4) if looking_away_duration is not None else float("nan"),
                int(head_nodding) if head_nodding is not None else float("nan"),
                self.fatigue_label,
                self.distraction_label,
            ]
        else:
            # Explicit missing face representation
            # All facial measurements and head_nodding are recorded as NaN (not misleading zeros)
            row = [
                sample_timestamp,
                self.participant_id,
                self.session_id,
                self.activity_label,
                0,
                calib_val,
                float("nan"),  # ear
                float("nan"),  # mar
                float("nan"),  # perclos
                float("nan"),  # blink_rate
                float("nan"),  # blink_duration
                float("nan"),  # eye_closure_duration
                float("nan"),  # yawn_rate
                float("nan"),  # yawn_duration
                float("nan"),  # head_pitch
                float("nan"),  # head_yaw
                float("nan"),  # head_roll
                float("nan"),  # gaze_x
                float("nan"),  # gaze_y
                float("nan"),  # looking_away_duration
                float("nan"),  # head_nodding is NaN when face detection fails!
                self.fatigue_label,
                self.distraction_label,
            ]

        self.csv_writer.writerow(row)

        # Flush every 10 samples (1 second at nominal 10 Hz) to avoid data loss
        if self.sample_count % 10 == 0 and self.file_handle:
            self.file_handle.flush()

        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state dictionary for dashboard display."""
        duration = time.time() - self.recording_start_time if (self.is_recording and self.recording_start_time) else 0.0
        elapsed_mono = (self.last_sample_time - self.first_sample_time) if (self.first_sample_time and self.last_sample_time) else 0.0
        achieved_rate = round((self.sample_count - 1) / elapsed_mono, 2) if (elapsed_mono > 0 and self.sample_count > 1) else (round(self.sample_count / duration, 2) if duration > 0 else 0.0)
        return {
            "is_recording": self.is_recording,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "activity_label": self.activity_label,
            "sample_count": self.sample_count,
            "duration_s": duration,
            "filepath": self.current_filepath,
            "metadata_filepath": self.metadata_filepath,
            "sampling_rate": self.sampling_rate,
            "achieved_rate": achieved_rate,
            "irregular_intervals": self.irregular_intervals_count,
        }

    def close(self):
        """Cleanup upon application shutdown."""
        if self.is_recording:
            self.stop_recording()
