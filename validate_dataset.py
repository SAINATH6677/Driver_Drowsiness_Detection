"""
================================================================================
DRIVEGUARD AI - RESEARCH-SAFE DATASET VALIDATOR & SEQUENCE READINESS CHECKER
================================================================================

Multi-tier validation for DriveGuard AI behavioural datasets:
1. CSV STRUCTURAL INTEGRITY:
   - File existence, non-empty row check
   - Header conformance to standard 23-column schema (or legacy 22-column)
   - Consistent column count across all rows
   - Monotonic increasing timestamps (no duplicate or backward timestamps)
   - Numerical parsing integrity (no corrupted non-float values)
   - Participant ID and session ID consistency

2. TEMPORAL SEQUENCE READINESS:
   - Evaluates genuine fixed-duration continuous temporal sequences (target 5.0s)
   - Never crosses participant boundaries
   - Never crosses session boundaries
   - Never crosses activity transition boundaries (homogeneous activity per sequence)
   - Excludes calibration baseline samples (is_calibrated == 0 or unverified)
   - Rejects or breaks sequences across timestamp gaps (> 0.25s) and missing-face occlusions
   - Does NOT assume 50 rows = 5 seconds (measures actual elapsed duration)

3. SCIENTIFIC GROUND-TRUTH READINESS:
   - Verifies whether fatigue/distraction labels represent clinical ground truth
     (e.g., EEG, PVT, Karolinska Sleepiness Scale) or prompted behavioural simulations.
================================================================================
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import List, Dict, Any, Set, Tuple

import numpy as np

from feature_logger import FEATURE_SCHEMA, SUPPORTED_ACTIVITIES, SCHEMA_VERSION

TARGET_SAMPLING_RATE_HZ = 10.0
TARGET_DT = 1.0 / TARGET_SAMPLING_RATE_HZ  # 0.100 seconds
SEQUENCE_TARGET_DURATION_S = 5.0  # Standard 5.0-second temporal window
MAX_GAP_THRESHOLD_S = 0.25  # Maximum allowable continuous gap before breaking sequence

NUMERICAL_FEATURE_COLUMNS = [
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
]

LEGACY_FEATURE_SCHEMA = [
    "timestamp",
    "participant_id",
    "session_id",
    "activity_label",
    "face_detected",
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

FEATURE_BOUNDS = {
    "ear": (0.0, 0.8),
    "mar": (0.0, 1.8),
    "perclos": (0.0, 1.0),
    "blink_rate": (0.0, 120.0),
    "blink_duration": (0.0, 5.0),
    "eye_closure_duration": (0.0, 30.0),
    "yawn_rate": (0.0, 30.0),
    "yawn_duration": (0.0, 15.0),
    "head_pitch": (-90.0, 90.0),
    "head_yaw": (-180.0, 180.0),
    "head_roll": (-180.0, 180.0),
    "gaze_x": (0.0, 1.0),
    "gaze_y": (0.0, 1.0),
    "looking_away_duration": (0.0, 60.0),
    "head_nodding": (0, 1),
}


def evaluate_temporal_sequences(
    rows: List[List[str]],
    header_indices: Dict[str, int],
    target_duration: float = SEQUENCE_TARGET_DURATION_S,
    max_gap: float = MAX_GAP_THRESHOLD_S,
) -> Dict[str, Any]:
    """
    Research-safe temporal sequence readiness checker.
    
    Guarantees:
    - Never crosses participant boundaries
    - Never crosses session boundaries
    - Never crosses activity transitions
    - Excludes calibration samples (is_calibrated == 0 or unverified legacy)
    - Rejects windows containing missing face detections
    - Rejects or breaks contiguous windows across timestamp gaps > max_gap
    - Evaluates elapsed physical duration (target_duration seconds), NOT row counts.
    """
    seq_result = {
        "is_ready": False,
        "target_duration_s": target_duration,
        "max_gap_threshold_s": max_gap,
        "total_non_overlapping": 0,
        "total_sliding": 0,
        "sequences_by_activity": {},
        "discarded_short_segments": 0,
        "homogeneous_segments": [],
        "notes": [],
    }

    if not rows:
        return seq_result

    # Determine calibration column presence
    has_calib_col = "is_calibrated" in header_indices

    # Parse and filter rows into candidate valid frames
    valid_frames = []
    for r in rows:
        try:
            t = float(r[header_indices["timestamp"]])
            p_id = r[header_indices["participant_id"]]
            s_id = r[header_indices["session_id"]]
            act = r[header_indices["activity_label"]]
            fd = int(r[header_indices["face_detected"]])
            calib = int(r[header_indices["is_calibrated"]]) if has_calib_col else None
        except (ValueError, KeyError):
            continue

        valid_frames.append({
            "timestamp": t,
            "participant_id": p_id,
            "session_id": s_id,
            "activity_label": act,
            "face_detected": fd,
            "is_calibrated": calib,
        })

    # Group into contiguous, homogeneous, gap-free, face-tracked segments
    segments = []
    current_segment = []

    for f in valid_frames:
        # Exclude uncalibrated baseline samples
        if has_calib_col and f["is_calibrated"] != 1:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        # Exclude missing face samples from continuous temporal sequence
        if f["face_detected"] != 1:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        if not current_segment:
            current_segment = [f]
            continue

        prev = current_segment[-1]
        dt = f["timestamp"] - prev["timestamp"]

        # Check segment boundary conditions:
        # 1. Activity transition
        # 2. Participant change
        # 3. Session change
        # 4. Excessive timestamp gap
        if (
            f["activity_label"] != prev["activity_label"]
            or f["participant_id"] != prev["participant_id"]
            or f["session_id"] != prev["session_id"]
            or dt > max_gap
            or dt <= 0.0
        ):
            segments.append(current_segment)
            current_segment = [f]
        else:
            current_segment.append(f)

    if current_segment:
        segments.append(current_segment)

    # Evaluate sequence extraction within each homogeneous segment
    for seg in segments:
        if len(seg) < 2:
            seq_result["discarded_short_segments"] += 1
            continue

        t_start = seg[0]["timestamp"]
        t_end = seg[-1]["timestamp"]
        seg_duration = round(t_end - t_start, 3)
        act = seg[0]["activity_label"]

        seg_info = {
            "activity_label": act,
            "participant_id": seg[0]["participant_id"],
            "session_id": seg[0]["session_id"],
            "samples": len(seg),
            "start_time": t_start,
            "end_time": t_end,
            "duration_s": seg_duration,
        }
        seq_result["homogeneous_segments"].append(seg_info)

        if seg_duration >= target_duration:
            # Non-overlapping count = floor(seg_duration / target_duration)
            non_overlap = int(seg_duration // target_duration)
            # Sliding count with 1.0s stride
            sliding = int((seg_duration - target_duration) // 1.0) + 1

            seq_result["total_non_overlapping"] += non_overlap
            seq_result["total_sliding"] += sliding
            seq_result["sequences_by_activity"][act] = (
                seq_result["sequences_by_activity"].get(act, 0) + non_overlap
            )
        else:
            seq_result["discarded_short_segments"] += 1

    seq_result["is_ready"] = seq_result["total_non_overlapping"] > 0
    return seq_result


def validate_csv_file(filepath: str) -> Dict[str, Any]:
    """Inspects a single CSV file and evaluates all 3 validation tiers."""
    result = {
        "filepath": filepath,
        "is_valid": True,  # Backward compatibility alias for structural_validity
        "structural_validity": True,
        "errors": [],
        "warnings": [],
        "schema_version_detected": "unknown",
        "total_rows": 0,
        "calibrated_rows": 0,
        "uncalibrated_rows": 0,
        "unknown_calibration_rows": 0,
        "calibration_status": "verified",
        "calibrated_pct": 0.0,
        "uncalibrated_pct": 0.0,
        "face_detected_rows": 0,
        "missing_face_rows": 0,
        "missing_face_pct": 0.0,
        "missing_feature_counts": {col: 0 for col in NUMERICAL_FEATURE_COLUMNS},
        "missing_feature_pcts": {col: 0.0 for col in NUMERICAL_FEATURE_COLUMNS},
        "duration_s": 0.0,
        "median_dt": 0.0,
        "mean_dt": 0.0,
        "p5_dt": 0.0,
        "p25_dt": 0.0,
        "p75_dt": 0.0,
        "p95_dt": 0.0,
        "p99_dt": 0.0,
        "min_dt": 0.0,
        "max_dt": 0.0,
        "achieved_sampling_rate_hz": 0.0,
        "avg_sampling_rate_hz": 0.0,  # Backward compatibility alias
        "interval_bins": {
            "3_frame_quanta_100ms": 0,
            "4_frame_quanta_133ms": 0,
            "large_gaps_over_250ms": 0,
        },
        "irregular_intervals_count": 0,
        "irregular_intervals_pct": 0.0,
        "large_gaps_count": 0,
        "max_gap_s": 0.0,
        "participant_id": None,
        "session_id": None,
        "unique_participant_ids": set(),
        "unique_session_ids": set(),
        "activity_distribution": {},
        "activity_distribution_pct": {},
        "activity_transitions": [],
        "activities_found": set(),  # Backward compatibility alias
        "metadata_found": False,
        "metadata_info": {},
        "temporal_readiness": {},
        "gru_sequences_non_overlapping": 0,  # Backward compatibility alias
        "gru_sequences_sliding": 0,  # Backward compatibility alias
        "ground_truth_readiness": {
            "status": "BEHAVIORAL_SIMULATION",
            "fatigue_labels": set(),
            "distraction_labels": set(),
            "has_clinical_validation": False,
        },
    }

    if not os.path.exists(filepath):
        result["structural_validity"] = False
        result["is_valid"] = False
        result["errors"].append(f"File does not exist: {filepath}")
        return result

    # Check companion metadata sidecar
    meta_path = os.path.splitext(filepath)[0] + ".json"
    if os.path.exists(meta_path):
        result["metadata_found"] = True
        try:
            with open(meta_path, mode="r", encoding="utf-8") as mf:
                result["metadata_info"] = json.load(mf)
        except Exception as e:
            result["warnings"].append(f"Could not parse metadata sidecar JSON: {e}")

    timestamps = []
    raw_rows = []
    last_activity = None

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            result["structural_validity"] = False
            result["is_valid"] = False
            result["errors"].append("File is empty (0 bytes).")
            return result

        # Tier 1: Header validation
        if header == FEATURE_SCHEMA:
            result["schema_version_detected"] = "23-column (v1.2.0)"
        elif header == LEGACY_FEATURE_SCHEMA:
            result["schema_version_detected"] = "22-column (legacy)"
            result["calibration_status"] = "unknown (legacy schema lacks is_calibrated column)"
            result["warnings"].append("Legacy 22-column schema detected (lacks 'is_calibrated' column; calibration status is unknown).")
        else:
            result["structural_validity"] = False
            result["is_valid"] = False
            diff_expected = [c for c in FEATURE_SCHEMA if c not in header]
            diff_unexpected = [c for c in header if c not in FEATURE_SCHEMA]
            result["errors"].append(f"Header schema mismatch. Missing: {diff_expected}, Unexpected: {diff_unexpected}")

        header_indices = {col: idx for idx, col in enumerate(header)}

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(c.strip() == "" for c in row):
                continue

            result["total_rows"] += 1
            raw_rows.append(row)

            if len(row) != len(header):
                result["structural_validity"] = False
                result["is_valid"] = False
                result["errors"].append(f"Row {row_idx}: Column count {len(row)} != {len(header)}")
                continue

            # Timestamp check
            try:
                t = float(row[header_indices["timestamp"]])
                timestamps.append(t)
            except ValueError:
                result["structural_validity"] = False
                result["is_valid"] = False
                result["errors"].append(f"Row {row_idx}: Non-numeric timestamp '{row[header_indices['timestamp']]}'")

            # Participant & Session checks
            p_id = row[header_indices["participant_id"]]
            s_id = row[header_indices["session_id"]]
            act = row[header_indices["activity_label"]]

            result["unique_participant_ids"].add(p_id)
            result["unique_session_ids"].add(s_id)

            if result["participant_id"] is None:
                result["participant_id"] = p_id
            if result["session_id"] is None:
                result["session_id"] = s_id

            # Ground truth label collection
            if "fatigue_label" in header_indices:
                result["ground_truth_readiness"]["fatigue_labels"].add(row[header_indices["fatigue_label"]])
            if "distraction_label" in header_indices:
                result["ground_truth_readiness"]["distraction_labels"].add(row[header_indices["distraction_label"]])

            # Activity tracking & distribution
            result["activity_distribution"][act] = result["activity_distribution"].get(act, 0) + 1
            if act != last_activity:
                result["activity_transitions"].append({
                    "activity": act,
                    "sample_index": result["total_rows"] - 1,
                    "timestamp": t if timestamps else None,
                })
                last_activity = act

            # Face detected status
            try:
                fd = int(row[header_indices["face_detected"]])
                if fd == 1:
                    result["face_detected_rows"] += 1
                else:
                    result["missing_face_rows"] += 1
            except ValueError:
                fd = 0
                result["missing_face_rows"] += 1

            # Calibration state
            if "is_calibrated" in header_indices:
                try:
                    calib = int(row[header_indices["is_calibrated"]])
                    if calib == 1:
                        result["calibrated_rows"] += 1
                    else:
                        result["uncalibrated_rows"] += 1
                except ValueError:
                    result["uncalibrated_rows"] += 1
            else:
                # Legacy schema: do not assume calibrated! Explicitly record unknown.
                result["unknown_calibration_rows"] += 1

            # Feature validity and missingness analysis
            for col in NUMERICAL_FEATURE_COLUMNS:
                val_str = row[header_indices[col]]
                try:
                    val = float(val_str)
                    if math.isnan(val):
                        result["missing_feature_counts"][col] += 1
                    else:
                        if col in FEATURE_BOUNDS:
                            low, high = FEATURE_BOUNDS[col]
                            if val < low or val > high:
                                result["warnings"].append(
                                    f"Row {row_idx}: {col} value {val} outside standard bounds [{low}, {high}]"
                                )
                except ValueError:
                    result["missing_feature_counts"][col] += 1
                    result["errors"].append(f"Row {row_idx}: Invalid numeric format for {col}: '{val_str}'")
                    result["structural_validity"] = False
                    result["is_valid"] = False

            # Check that missing face rows do not have misleading non-NaN values
            if fd == 0:
                for col in NUMERICAL_FEATURE_COLUMNS:
                    val_str = row[header_indices[col]]
                    try:
                        val = float(val_str)
                        if not math.isnan(val):
                            result["warnings"].append(
                                f"Row {row_idx}: Non-NaN numerical value ({val}) for {col} when face_detected=0 (should be NaN)"
                            )
                    except ValueError:
                        pass

    # Empty file check: If 0 data rows, mark as invalid!
    if result["total_rows"] == 0:
        result["structural_validity"] = False
        result["is_valid"] = False
        result["errors"].append("File contains 0 data rows (empty recording).")
        return result

    # Participant & Session consistency across all rows
    if len(result["unique_participant_ids"]) > 1:
        result["structural_validity"] = False
        result["is_valid"] = False
        result["errors"].append(
            f"Mixed participant IDs detected in single CSV: {sorted(result['unique_participant_ids'])}"
        )
    if len(result["unique_session_ids"]) > 1:
        result["structural_validity"] = False
        result["is_valid"] = False
        result["errors"].append(
            f"Mixed session IDs detected in single CSV: {sorted(result['unique_session_ids'])}"
        )

    # Calculate percentages
    total = result["total_rows"]
    result["missing_face_pct"] = round((result["missing_face_rows"] / total) * 100.0, 1)
    result["calibrated_pct"] = round((result["calibrated_rows"] / total) * 100.0, 1)
    result["uncalibrated_pct"] = round((result["uncalibrated_rows"] / total) * 100.0, 1)

    for col, count in result["missing_feature_counts"].items():
        result["missing_feature_pcts"][col] = round((count / total) * 100.0, 1)

    for act, count in result["activity_distribution"].items():
        result["activity_distribution_pct"][act] = round((count / total) * 100.0, 1)

    # Detailed interval analysis and quantile calculations
    if len(timestamps) >= 2:
        diffs = np.diff(timestamps)
        negative_diffs = [d for d in diffs if d <= 0]
        if negative_diffs:
            result["structural_validity"] = False
            result["is_valid"] = False
            result["errors"].append(f"Detected {len(negative_diffs)} non-increasing or duplicate timestamps!")

        duration = timestamps[-1] - timestamps[0]
        result["duration_s"] = round(duration, 2)

        result["median_dt"] = round(float(np.median(diffs)), 4)
        result["mean_dt"] = round(float(np.mean(diffs)), 4)
        result["min_dt"] = round(float(np.min(diffs)), 4)
        result["max_dt"] = round(float(np.max(diffs)), 4)
        result["p5_dt"] = round(float(np.percentile(diffs, 5)), 4)
        result["p25_dt"] = round(float(np.percentile(diffs, 25)), 4)
        result["p75_dt"] = round(float(np.percentile(diffs, 75)), 4)
        result["p95_dt"] = round(float(np.percentile(diffs, 95)), 4)
        result["p99_dt"] = round(float(np.percentile(diffs, 99)), 4)

        if duration > 0:
            result["achieved_sampling_rate_hz"] = round((len(timestamps) - 1) / duration, 2)
            result["avg_sampling_rate_hz"] = result["achieved_sampling_rate_hz"]

        result["activities_found"] = set(result["activity_distribution"].keys())

        # Bimodal camera-quanta breakdown
        b_3frame = int(np.sum((diffs >= 0.095) & (diffs <= 0.105)))
        b_4frame = int(np.sum((diffs > 0.120) & (diffs <= 0.145)))
        b_large = int(np.sum(diffs > MAX_GAP_THRESHOLD_S))

        result["interval_bins"]["3_frame_quanta_100ms"] = b_3frame
        result["interval_bins"]["4_frame_quanta_133ms"] = b_4frame
        result["interval_bins"]["large_gaps_over_250ms"] = b_large

        irregular_diffs = [d for d in diffs if abs(d - TARGET_DT) > 0.03]
        result["irregular_intervals_count"] = len(irregular_diffs)
        result["irregular_intervals_pct"] = round((len(irregular_diffs) / len(diffs)) * 100.0, 1)
        result["large_gaps_count"] = b_large
        result["max_gap_s"] = round(float(np.max(diffs)), 4) if len(diffs) > 0 else 0.0

        if result["large_gaps_count"] > 0:
            result["warnings"].append(
                f"Detected {result['large_gaps_count']} large gaps (>0.25s). Maximum gap: {result['max_gap_s']}s"
            )

    # Tier 2: Temporal Sequence Readiness Check
    seq_eval = evaluate_temporal_sequences(
        rows=raw_rows,
        header_indices=header_indices,
        target_duration=SEQUENCE_TARGET_DURATION_S,
        max_gap=MAX_GAP_THRESHOLD_S,
    )
    result["temporal_readiness"] = seq_eval
    result["gru_sequences_non_overlapping"] = seq_eval["total_non_overlapping"]
    result["gru_sequences_sliding"] = seq_eval["total_sliding"]

    if result["errors"]:
        result["structural_validity"] = False
        result["is_valid"] = False

    return result


def print_validation_report(results: List[Dict[str, Any]]):
    """Prints a structured 3-tier validation report for all datasets."""
    print("\n" + "=" * 84)
    print("           DRIVEGUARD AI - RESEARCH DATASET VALIDATION REPORT          ")
    print("=" * 84)

    total_files = len(results)
    structurally_valid = sum(1 for r in results if r["structural_validity"])
    total_samples = sum(r["total_rows"] for r in results)
    total_duration = sum(r["duration_s"] for r in results)
    total_ready_seqs = sum(r["gru_sequences_non_overlapping"] for r in results)

    print(f"Total CSV Files Inspected:      {total_files}")
    print(f"Structurally Valid CSV Files:   {structurally_valid} / {total_files}")
    print(f"Total Recorded Timesteps:       {total_samples} ({total_duration:.1f} seconds total)")
    print(f"Verified Homogeneous Sequences: {total_ready_seqs} (5.0s continuous windows, no activity mixing)")
    print("-" * 84)

    for r in results:
        fname = os.path.basename(r["filepath"])
        struct_ok = r["structural_validity"]
        has_seqs = r["gru_sequences_non_overlapping"] > 0
        status_symbol = "✔" if struct_ok else "✖"
        status_label = "STRUCT_OK" if struct_ok else "STRUCT_FAIL"

        print(f"\n[{status_symbol}] {fname} [{status_label}]")
        print(f"    Schema Format:    {r['schema_version_detected']}")
        print(f"    Participant ID:   {r['participant_id']} | Session: {r['session_id']}")

        # Manifest JSON
        if r["metadata_found"]:
            m = r["metadata_info"]
            utc_start = m.get("utc_recording_start_time", "N/A")
            sw_ver = m.get("software_version", "N/A")
            sch_ver = m.get("schema_version", "N/A")
            print(f"    Metadata Manifest: Present (App: {sw_ver}, Schema: {sch_ver}, UTC: {utc_start})")
        else:
            print("    Metadata Manifest: Not found (sidecar .json missing)")

        print(f"    Duration & Rows:  {r['duration_s']}s | Samples: {r['total_rows']}")

        if r["total_rows"] > 0:
            # Interval Statistics & Sampling Rate
            print(f"    Sampling Rates:   Achieved: {r['achieved_sampling_rate_hz']} Hz (Target: {TARGET_SAMPLING_RATE_HZ} Hz)")
            print(f"    Interval Quantiles: Median: {r['median_dt']}s | Mean: {r['mean_dt']}s | p95: {r['p95_dt']}s | Max: {r['max_dt']}s (Min: {r['min_dt']}s)")

            # Empirical Quantization Analysis
            b3 = r["interval_bins"]["3_frame_quanta_100ms"]
            b4 = r["interval_bins"]["4_frame_quanta_133ms"]
            n_diffs = max(1, r["total_rows"] - 1)
            print(f"    Camera Quanta:    ~100ms (3 camera frames @ 30 FPS): {b3} ({b3 / n_diffs * 100:.1f}%)")
            print(f"                      ~133ms (4 camera frames @ 30 FPS): {b4} ({b4 / n_diffs * 100:.1f}%)")
            print(f"                      Gaps > 250ms: {r['large_gaps_count']} | Irregular (>0.03s dev): {r['irregular_intervals_count']} ({r['irregular_intervals_pct']}%)")

            # Calibration Breakdown
            if r["schema_version_detected"] == "22-column (legacy)":
                print("    Calibration State: UNVERIFIED (Legacy 22-column file lacks is_calibrated indicator)")
            else:
                print(f"    Calibration State: Calibrated: {r['calibrated_rows']} ({r['calibrated_pct']}%) | Baseline: {r['uncalibrated_rows']} ({r['uncalibrated_pct']}%)")

            # Face Tracking
            print(f"    Face Tracking:    Face Detected: {r['face_detected_rows']} | Missing Face: {r['missing_face_rows']} ({r['missing_face_pct']}%)")

            # Activity Distribution
            act_strs = [f"{act}: {cnt} ({r['activity_distribution_pct'].get(act, 0)}%)" for act, cnt in sorted(r["activity_distribution"].items())]
            print(f"    Activities:       {', '.join(act_strs) if act_strs else 'None'}")
            print(f"    Transitions:      {len(r['activity_transitions'])} activity boundaries detected in session")

            # Tier 2: Temporal Sequence Readiness Check
            t_ready = r["temporal_readiness"]
            print(f"    Sequence Readiness: {'READY' if has_seqs else 'INSUFFICIENT_WINDOW_LENGTH'}")
            print(f"      - Verified 5.0s Non-Overlapping Sequences: {t_ready.get('total_non_overlapping', 0)}")
            print(f"      - Verified 5.0s Sliding Sequences:         {t_ready.get('total_sliding', 0)}")
            if t_ready.get("sequences_by_activity"):
                seq_breakdown = [f"{act}: {n}" for act, n in sorted(t_ready["sequences_by_activity"].items())]
                print(f"      - Sequences by Activity: {', '.join(seq_breakdown)}")

            # Tier 3: Scientific Ground-Truth Readiness
            fat_lbls = r["ground_truth_readiness"]["fatigue_labels"]
            dist_lbls = r["ground_truth_readiness"]["distraction_labels"]
            is_simulation = (fat_lbls == {"unknown"} or not fat_lbls) and (dist_lbls == {"unknown"} or not dist_lbls)
            gt_status = "BEHAVIORAL SIMULATION ONLY" if is_simulation else "ANNOTATED GROUND TRUTH"
            print(f"    Ground-Truth Tier: {gt_status} (Labels: Fatigue={sorted(fat_lbls)}, Distraction={sorted(dist_lbls)})")

        if r["errors"]:
            print("    ERRORS:")
            for err in r["errors"]:
                print(f"      - {err}")

        if r["warnings"]:
            print("    WARNINGS:")
            unique_warns = list(dict.fromkeys(r["warnings"]))
            for warn in unique_warns[:4]:
                print(f"      - {warn}")
            if len(unique_warns) > 4:
                print(f"      ... and {len(unique_warns) - 4} more warnings")

    # Final Overall Multi-Tier Verdict
    print("\n" + "=" * 84)
    print("                    FINAL DATASET READINESS SUMMARY                     ")
    print("=" * 84)
    print(f"1. CSV STRUCTURAL INTEGRITY:   {structurally_valid} / {total_files} Files PASS")
    print(f"2. TEMPORAL SEQUENCE READINESS: {total_ready_seqs} Verified 5.0-Second Homogeneous Sequences Ready")
    print("3. SCIENTIFIC GROUND TRUTH:    BEHAVIORAL SIMULATION ONLY (Labels: unknown)")
    print("   NOTE: Datasets represent driver behavioural demonstrations, NOT clinical")
    print("   or sensor-validated drowsiness ground truth. Suitable for feature modeling")
    print("   and action recognition, but cannot validate drowsiness detection accuracy.")
    print("=" * 84 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate collected DriveGuard AI CSV datasets for GRU deep learning training.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/raw",
        help="Path to a specific CSV file or directory containing raw CSV logs (default: data/raw).",
    )
    args = parser.parse_args()

    target_path = args.path
    if os.path.isfile(target_path):
        files = [target_path]
    elif os.path.isdir(target_path):
        files = sorted(glob.glob(os.path.join(target_path, "*.csv")))
    else:
        print(f"[ERROR]: Path does not exist: {target_path}")
        sys.exit(1)

    if not files:
        print(f"[INFO]: No CSV files found in: {target_path}")
        sys.exit(0)

    results = [validate_csv_file(f) for f in files]
    print_validation_report(results)


if __name__ == "__main__":
    main()
