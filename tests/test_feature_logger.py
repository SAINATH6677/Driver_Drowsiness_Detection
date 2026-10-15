"""
================================================================================
DRIVEGUARD AI - UNIT TEST SUITE FOR FEATURELOGGER & DATASET VALIDATOR
================================================================================

This module tests the behavioural data logger, participant isolation,
missing-face NaN integrity, companion metadata manifests, interval quantiles,
empty file rejection, activity boundaries, and research sequence readiness.

IMPORTANT DISCLAIMER:
All synthetic numerical feature values in this test suite are strictly for
software verification (e.g. boundary conditions, column order, format parsing).
They do NOT represent clinical, physiological, or research ground truth.
================================================================================
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_logger import (
    FeatureLogger,
    FEATURE_SCHEMA,
    SUPPORTED_ACTIVITIES,
    SCHEMA_VERSION,
    SOFTWARE_VERSION,
)
from validate_dataset import (
    validate_csv_file,
    evaluate_temporal_sequences,
    NUMERICAL_FEATURE_COLUMNS,
    LEGACY_FEATURE_SCHEMA,
)


class TestFeatureLogger(unittest.TestCase):
    """Unit tests for FeatureLogger class and data collection mechanics."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    def test_schema_and_metadata_creation(self):
        """Verify 23-column CSV schema and companion JSON metadata generation."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="TEST_P01",
            session_id="TEST_S01",
            sampling_rate=10.0,
            activity_label="normal_alert",
        )
        csv_path = logger.start_recording()
        meta_path = logger.metadata_filepath

        self.assertTrue(os.path.exists(csv_path), "CSV file was not created.")
        self.assertTrue(os.path.exists(meta_path), "Metadata JSON sidecar was not created.")

        # Check CSV header
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, FEATURE_SCHEMA, "CSV header does not match FEATURE_SCHEMA.")
            self.assertEqual(len(header), 23, "CSV header must have exactly 23 columns.")

        # Check initial metadata JSON
        with open(meta_path, mode="r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["software_version"], SOFTWARE_VERSION)
            self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
            self.assertEqual(meta["participant_id"], "TEST_P01")
            self.assertEqual(meta["session_id"], "TEST_S01")
            self.assertEqual(meta["target_sampling_rate_hz"], 10.0)
            self.assertEqual(meta["status"], "recording")
            self.assertIn("utc_recording_start_time", meta)
            self.assertIn("utc_recording_start_epoch", meta)

        # Log 10 controlled synthetic frames at 10 Hz
        for i in range(10):
            logger.log_frame(
                current_time=100.0 + i * 0.10,
                face_detected=True,
                is_calibrated=True,
                ear=0.25,
                mar=0.30,
                perclos=0.05,
                head_pitch=1.0,
                head_yaw=0.0,
                head_roll=0.0,
                head_nodding=0,
            )

        saved_path = logger.stop_recording()
        self.assertEqual(saved_path, csv_path)

        # Check finalized metadata JSON
        with open(meta_path, mode="r", encoding="utf-8") as f:
            final_meta = json.load(f)
            self.assertEqual(final_meta["status"], "completed")
            self.assertEqual(final_meta["total_samples"], 10)
            self.assertIn("utc_recording_end_time", final_meta)
            self.assertIn("achieved_sampling_rate_hz", final_meta)
            self.assertAlmostEqual(final_meta["achieved_sampling_rate_hz"], 10.0, delta=0.5)

    def test_missing_face_nan_including_head_nodding(self):
        """Verify that when face is not detected, ALL numerical metrics (including head_nodding) are NaN."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="TEST_P02",
            session_id="TEST_S02",
        )
        csv_path = logger.start_recording()

        # Log a missing face frame
        logged = logger.log_frame(
            current_time=50.0,
            face_detected=False,
            is_calibrated=True,
        )
        self.assertTrue(logged)
        logger.stop_recording()

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            indices = {col: i for i, col in enumerate(header)}

            self.assertEqual(row[indices["face_detected"]], "0")
            self.assertEqual(row[indices["is_calibrated"]], "1")

            # Check that head_nodding is explicitly NaN (not misleading 0)
            head_nod_val = float(row[indices["head_nodding"]])
            self.assertTrue(math.isnan(head_nod_val), "head_nodding must be NaN when face detection fails.")

            # Check that all other numerical feature columns are NaN
            for col in NUMERICAL_FEATURE_COLUMNS:
                val = float(row[indices[col]])
                self.assertTrue(math.isnan(val), f"Column '{col}' must be NaN on missing face frame.")

    def test_rate_throttling_and_intervals(self):
        """Verify rate-limiting logic throttles rapid frames while permitting target ~10 Hz interval."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            sampling_rate=10.0,  # 0.10s interval
        )
        logger.start_recording()

        # Frame 1 at t = 1.00s -> should succeed
        self.assertTrue(logger.log_frame(current_time=1.00, face_detected=True, ear=0.3))

        # Frame 2 at t = 1.05s (< 0.10s) -> should be throttled / skipped
        self.assertFalse(logger.log_frame(current_time=1.05, face_detected=True, ear=0.3))

        # Frame 3 at t = 1.10s (>= 0.10s) -> should succeed
        self.assertTrue(logger.log_frame(current_time=1.10, face_detected=True, ear=0.3))

        # Frame 4 at t = 1.30s (irregular interval: 0.20s > 0.10s by 0.10s) -> should succeed and flag irregular
        self.assertTrue(logger.log_frame(current_time=1.30, face_detected=True, ear=0.3))
        self.assertGreaterEqual(logger.irregular_intervals_count, 1)

        logger.stop_recording()
        self.assertEqual(logger.sample_count, 3)

    def test_participant_switching_isolation(self):
        """Verify participant switching closes old CSV and opens new file without mixing data."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="PARTICIPANT_A",
            session_id="SESS_01",
        )
        file_a = logger.start_recording()

        # Log 5 frames for PARTICIPANT_A
        for i in range(5):
            logger.log_frame(current_time=10.0 + i * 0.10, face_detected=True, ear=0.25)

        # Switch participant to PARTICIPANT_B mid-recording
        logger.set_participant("PARTICIPANT_B")
        file_b = logger.current_filepath

        # Files must be different
        self.assertNotEqual(file_a, file_b, "Switching participant must create a distinct CSV file.")
        self.assertTrue(os.path.exists(file_a), "Old CSV must remain intact on disk.")
        self.assertTrue(os.path.exists(file_b), "New CSV must be created.")

        # Log 5 frames for PARTICIPANT_B
        for i in range(5):
            logger.log_frame(current_time=20.0 + i * 0.10, face_detected=True, ear=0.30)
        logger.stop_recording()

        # Verify file_a contains ONLY PARTICIPANT_A
        with open(file_a, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            p_idx = header.index("participant_id")
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            for r in rows:
                self.assertEqual(r[p_idx], "PARTICIPANT_A")

        # Verify file_b contains ONLY PARTICIPANT_B
        with open(file_b, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            p_idx = header.index("participant_id")
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            for r in rows:
                self.assertEqual(r[p_idx], "PARTICIPANT_B")

    def test_activity_transitions_and_boundaries(self):
        """Verify that activity switching records transitions in metadata manifest and row labels."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="ACT_P01",
            activity_label="normal_alert",
        )
        csv_path = logger.start_recording()

        # 3 samples of normal_alert
        for i in range(3):
            logger.log_frame(current_time=1.0 + i * 0.10, face_detected=True, ear=0.3)

        # Switch to yawning
        logger.set_activity("yawning")
        for i in range(3):
            logger.log_frame(current_time=1.3 + i * 0.10, face_detected=True, ear=0.3)

        # Switch by index to head_nodding (8)
        logger.set_activity_by_index(8)
        for i in range(3):
            logger.log_frame(current_time=1.6 + i * 0.10, face_detected=True, ear=0.3)

        logger.stop_recording()

        # Check metadata JSON transitions
        meta_path = os.path.splitext(csv_path)[0] + ".json"
        with open(meta_path, mode="r", encoding="utf-8") as f:
            meta = json.load(f)
            transitions = meta["activity_transitions"]
            self.assertGreaterEqual(len(transitions), 3)
            self.assertEqual(transitions[0]["activity_label"], "normal_alert")
            self.assertEqual(transitions[1]["activity_label"], "yawning")
            self.assertEqual(transitions[2]["activity_label"], "head_nodding")

        # Check CSV rows have correct activity labels
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            act_idx = header.index("activity_label")
            rows = list(reader)
            self.assertEqual(len(rows), 9)
            self.assertEqual([r[act_idx] for r in rows[:3]], ["normal_alert"] * 3)
            self.assertEqual([r[act_idx] for r in rows[3:6]], ["yawning"] * 3)
            self.assertEqual([r[act_idx] for r in rows[6:9]], ["head_nodding"] * 3)

    def test_empty_csv_rejected(self):
        """Verify validate_dataset.py explicitly rejects an empty CSV file (0 data rows)."""
        empty_csv = os.path.join(self.data_dir, "empty_test.csv")
        with open(empty_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_SCHEMA)

        res = validate_csv_file(empty_csv)
        self.assertFalse(res["is_valid"], "Empty CSV must be marked invalid.")
        self.assertFalse(res["structural_validity"], "Empty CSV must fail structural validity.")
        self.assertEqual(res["total_rows"], 0)
        self.assertTrue(any("0 data rows" in err for err in res["errors"]))

    def test_legacy_schema_unknown_calibration(self):
        """Verify legacy 22-column schema reports unknown calibration status instead of assuming calibrated."""
        legacy_csv = os.path.join(self.data_dir, "legacy_test.csv")
        with open(legacy_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LEGACY_FEATURE_SCHEMA)
            # Write 5 valid rows
            for i in range(5):
                writer.writerow([
                    round(100.0 + i * 0.10, 4), "P_LEGACY", "S_LEGACY", "normal_alert", 1,
                    0.28, 0.30, 0.05, 15.0, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.5, 0.5, 0.0, 0, "unknown", "unknown"
                ])

        res = validate_csv_file(legacy_csv)
        self.assertTrue(res["structural_validity"], "Legacy CSV structure should pass with warning.")
        self.assertEqual(res["schema_version_detected"], "22-column (legacy)")
        self.assertEqual(res["unknown_calibration_rows"], 5)
        self.assertEqual(res["calibrated_rows"], 0)
        self.assertIn("unknown", res["calibration_status"])

    def test_activity_boundary_sequence_separation(self):
        """Verify temporal sequence readiness never combines different activities in the same sequence."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="SEQ_P01",
            session_id="SEQ_S01",
        )
        csv_path = logger.start_recording()

        # 60 samples (6.0s) of normal_alert -> yields 1 full 5.0s sequence
        for i in range(60):
            logger.log_frame(
                current_time=100.0 + i * 0.10,
                face_detected=True,
                is_calibrated=True,
                ear=0.28,
                mar=0.30,
            )

        # Switch to yawning for 60 samples (6.0s) -> yields 1 full 5.0s sequence
        logger.set_activity("yawning")
        for i in range(60):
            logger.log_frame(
                current_time=106.0 + i * 0.10,
                face_detected=True,
                is_calibrated=True,
                ear=0.28,
                mar=0.70,
            )

        # Switch to looking_down for only 20 samples (2.0s) -> too short for 5.0s sequence!
        logger.set_activity("looking_down")
        for i in range(20):
            logger.log_frame(
                current_time=112.0 + i * 0.10,
                face_detected=True,
                is_calibrated=True,
                ear=0.28,
                mar=0.30,
            )

        logger.stop_recording()

        res = validate_csv_file(csv_path)
        t_ready = res["temporal_readiness"]

        self.assertTrue(t_ready["is_ready"])
        self.assertEqual(t_ready["total_non_overlapping"], 2)
        # Sequence breakdown must be strictly segregated by activity!
        self.assertEqual(t_ready["sequences_by_activity"].get("normal_alert"), 1)
        self.assertEqual(t_ready["sequences_by_activity"].get("yawning"), 1)
        self.assertNotIn("looking_down", t_ready["sequences_by_activity"])
        self.assertGreaterEqual(t_ready["discarded_short_segments"], 1)

    def test_temporal_sequence_readiness_gap_rejection(self):
        """Verify that timestamp gaps > 0.25s break sequences rather than falsely creating a continuous window."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="GAP_P01",
            session_id="GAP_S01",
        )
        csv_path = logger.start_recording()

        # Segment A: 3.0s of normal_alert (30 frames)
        for i in range(30):
            logger.log_frame(current_time=10.0 + i * 0.10, face_detected=True, is_calibrated=True, ear=0.28)

        # Large Gap of 0.80s (> 0.25s threshold)
        gap_start = 13.80

        # Segment B: 3.0s of normal_alert (30 frames)
        for i in range(30):
            logger.log_frame(current_time=gap_start + i * 0.10, face_detected=True, is_calibrated=True, ear=0.28)

        logger.stop_recording()

        res = validate_csv_file(csv_path)
        t_ready = res["temporal_readiness"]

        # Neither segment alone meets the 5.0s threshold (both are 3.0s)
        # Because the 0.80s gap breaks them, they must NOT combine into a 6.0s window!
        self.assertEqual(t_ready["total_non_overlapping"], 0)
        self.assertEqual(t_ready["discarded_short_segments"], 2)

    def test_missing_face_sequence_exclusion(self):
        """Verify that missing face frames break temporal sequence continuity."""
        logger = FeatureLogger(
            data_dir=self.data_dir,
            participant_id="OCC_P01",
            session_id="OCC_S01",
        )
        csv_path = logger.start_recording()

        # 30 frames with face (3.0s)
        for i in range(30):
            logger.log_frame(current_time=1.0 + i * 0.10, face_detected=True, is_calibrated=True, ear=0.28)

        # 5 frames missing face
        for i in range(5):
            logger.log_frame(current_time=4.0 + i * 0.10, face_detected=False, is_calibrated=True)

        # 30 frames with face (3.0s)
        for i in range(30):
            logger.log_frame(current_time=4.5 + i * 0.10, face_detected=True, is_calibrated=True, ear=0.28)

        logger.stop_recording()

        res = validate_csv_file(csv_path)
        t_ready = res["temporal_readiness"]

        # 3.0s + occlusion + 3.0s -> neither face-tracked chunk reaches 5.0s
        self.assertEqual(t_ready["total_non_overlapping"], 0)


if __name__ == "__main__":
    unittest.main()
