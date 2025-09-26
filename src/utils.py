# -*- coding: utf-8 -*-
"""
Modules and Key Functions for the BIDS Conversion Pipeline
The system processes raw data files, performs necessary signal interpolation, and generates various output files including EDF, JSON, and TSV files.
"""

import logging
import warnings
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Sequence, Union, List
import bids_config as config

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyedflib

# Global constants to avoid redundant hard-coded values
DEFAULT_SAMPLING_FREQUENCY = 1000
DEFAULT_LOW_CUTOFF = 2
DEFAULT_HIGH_CUTOFF = 325

def setup_logging(info_log: str = "process.log", error_log: str = "error.log") -> logging.Logger:
    """
    Set up the logger for the application.

    Parameters:
    - info_log: File path for INFO level logs.
    - error_log: File path for WARNING and higher level logs.

    Returns:
    - Configured Logger object.
    """
    logger = logging.getLogger("bids_logger")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Info file handler
    fh_info = logging.FileHandler(info_log, mode='w', encoding='utf-8')
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(formatter)
    logger.addHandler(fh_info)

    # Error file handler
    fh_error = logging.FileHandler(error_log, mode='w', encoding='utf-8')
    fh_error.setLevel(logging.WARNING)
    fh_error.setFormatter(formatter)
    logger.addHandler(fh_error)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# Ignore warnings from pyedflib
warnings.filterwarnings('ignore')


def init_subject_qc_table(bids_root, sub_id):
    qc_dir = Path(bids_root) / "derivatives" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_tsv = qc_dir / f"sub-{sub_id}_missingdata_qc.tsv"   # 每个 subject 单独一个文件
    fieldnames = [
        "subject","session","task", "run",
        "n_samples",
        "missing_samples","missing_pct",
        "interpolated_samples","interpolated_pct",
        "missing_reason"
    ]

    with qc_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
    return qc_tsv

def generate_qc_row(qc_tsv_path, data_st, sub_id, ses_id, mapped_task_name, run_num, reason="occasional packet loss due to varying wireless conditions"):
    signals = data_st["signals"]  # (N, C)
    interp_mask = data_st["interp_mask"]  # (N,)
    N = int(signals.shape[0])

    if getattr(interp_mask, "ndim", 1) == 2:
        interp_t = (interp_mask == 1).any(axis=1)
    else:
        interp_t = (interp_mask == 1)

    if N == 0:
        interpolated_samples = 0
        interpolated_pct = 0.00
        n_samples = 0
    else:
        interpolated_samples = int(np.count_nonzero(interp_t))
        interpolated_pct = round(100.0 * interpolated_samples / N, 2)
        n_samples = N
        
    missing_samples = interpolated_samples
    missing_pct = interpolated_pct

    row = {
        "subject": f"sub-{sub_id}",
        "session": f"ses-{ses_id}",
        "task": mapped_task_name,
        "run": f"run-{run_num:02d}",
        "n_samples": n_samples,
        "missing_samples": missing_samples,
        "missing_pct": f"{missing_pct:.2f}",
        "interpolated_samples": interpolated_samples,
        "interpolated_pct": f"{interpolated_pct:.2f}",
        "missing_reason": reason,
    }
    fieldnames = [
        "subject","session","task", "run",
        "n_samples",
        "missing_samples","missing_pct",
        "interpolated_samples","interpolated_pct",
        "missing_reason"
    ]
    with qc_tsv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)

def get_post_op_day(date_str: str, surgery_date_str: str = "20240501") -> int:
    """
    Calculate the post-operative day based on measurement date and surgery date.

    Parameters:
    - date_str: Measurement date in "YYYYMMDD" format.
    - surgery_date_str: Surgery date in "YYYYMMDD" format (default: "20240501").

    Returns:
    - The number of days after surgery as an integer.
    """
    fmt = "%Y%m%d"
    measure_date = datetime.strptime(date_str, fmt)
    surgery_date = datetime.strptime(surgery_date_str, fmt)
    return (measure_date - surgery_date).days

def extract_date_from_filename(filename: str) -> datetime:
    """
    Extract date and time information from the filename and convert to a datetime object.

    Parameters:
    - filename: Filename containing date/time info (e.g., "cortec_20240501T123456...").

    Returns:
    - A datetime object.
    """
    prefix = "cortec_"
    if filename.startswith(prefix):
        filename = filename[len(prefix):]
    date_part, time_part = filename.split("T")
    if 'B' in time_part:
        time_part = time_part.split('B')[0]
    datetime_str = date_part + time_part
    return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")

def load_signals_cortec(data_fpath: str, interp_method: str = "none") -> Dict[str, Any]:
    """
    Load Cortec data from the corresponding .hdr and .bin files, and perform interpolation if required.

    Parameters:
    - data_fpath: Path to the data file (must end with .bin or .hdr).
    - interp_method: Interpolation method. Currently supports "linear+nearest".

    Returns:
    - A dictionary containing signal data and metadata.
    """
    logging.info(f"Loading signals from {data_fpath} with interpolation method: {interp_method}")

    if not data_fpath.endswith((".bin", ".hdr")):
        raise ValueError(f"{data_fpath}: Invalid file extension")

    base_fpath = os.path.splitext(data_fpath)[0]
    hdr_fpath = f"{base_fpath}.hdr"
    bin_fpath = f"{base_fpath}.bin"

    # Read the header file
    with open(hdr_fpath, "r", encoding="utf-8") as f:
        hdr_lines = f.readlines()

    main_header = hdr_lines[0].strip()
    reference_channel_line = hdr_lines[1].strip() if len(hdr_lines) >= 2 else None
    amplification = hdr_lines[2].strip() if len(hdr_lines) >= 3 else None
    ground_line = hdr_lines[3].strip() if len(hdr_lines) >= 4 else None
    ground = ground_line.lower() if ground_line and ground_line.lower() in ["true", "false"] else "false"

    header_parts = main_header.split(";")
    if len(header_parts) != 7:
        raise ValueError(f"{hdr_fpath}: Header format is incorrect")
    sampling_rate = float(header_parts[1])
    threshold_high = float(header_parts[2])
    threshold_low = float(header_parts[3])
    ch_num_signal = int(header_parts[4])
    ch_num_logic = int(header_parts[5])
    ch_num_total = ch_num_signal + ch_num_logic
    channel_names = header_parts[6].split(":")
    if len(channel_names) != ch_num_total:
        raise ValueError(f"{hdr_fpath}: Number of channels mismatch. Expected: {ch_num_total}, Actual: {len(channel_names)}")

    # Process the reference channel
    if reference_channel_line and reference_channel_line.isdigit():
        ref_idx = int(reference_channel_line)
        reference_channel = channel_names[ref_idx] if 0 <= ref_idx < ch_num_total else None
    else:
        reference_channel = None

    # Read the binary file
    with open(bin_fpath, "rb") as f:
        data_uint8 = f.read()

    byte_num_per_sample = 8 + 4 + 4 * ch_num_total
    remainder = len(data_uint8) % byte_num_per_sample
    if remainder > 0:
        data_uint8 = data_uint8[:-remainder]

    samples = len(data_uint8) // byte_num_per_sample
    data_uint8 = np.frombuffer(data_uint8, dtype=np.uint8).reshape(samples, byte_num_per_sample)

    unix_time_ms = np.frombuffer(data_uint8[:, 0:8].tobytes(), dtype=np.int64)
    sample_index = np.frombuffer(data_uint8[:, 8:12].tobytes(), dtype=np.uint32)
    signals = np.frombuffer(data_uint8[:, 12:].tobytes(), dtype=np.float32).reshape(samples, ch_num_total)

    data_st = {
        "sampling_rate": sampling_rate,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "channel_num_signal": ch_num_signal,
        "channel_num_trigger": ch_num_logic,
        "channel_names": channel_names,
        "signals": signals,
        "unix_time_ms": unix_time_ms,
        "sample_index": sample_index,
        "reference_channel": reference_channel,
        "amplification": amplification,
        "ground": ground
    }

    # Perform interpolation if requested
    if interp_method == "linear+nearest":
        interpolated_signals, interp_mask = interpolate_signals(signals, sample_index, channel_names)
        data_st["signals"] = interpolated_signals
        data_st["interp_mask"] = interp_mask

    logging.info(f"Successfully loaded signal data from {data_fpath}")
    return data_st

def interpolate_signals(signals: np.ndarray, sample_indices: np.ndarray, channel_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform interpolation on the input signals.
    Channels starting with "CH" are interpolated using linear interpolation,
    while other channels use nearest-neighbor interpolation.

    Parameters:
    - signals: Original signal array (num_samples x num_channels)
    - sample_indices: Array of original sampling indices
    - channel_names: List of channel names

    Returns:
    - interpolated_signals: The interpolated signal array
    - interp_mask: A mask array indicating interpolated samples (1 for interpolated, 0 for original)
    """
    logging.info("Starting signal interpolation.")

    if signals.size == 0:
        return signals, np.zeros_like(signals, dtype=np.int16)

    # Use the actual first sample index as the start to avoid unwanted extrapolation.
    start_idx = sample_indices[0]
    end_idx = sample_indices[-1]
    full_range = np.arange(start_idx, end_idx + 1)
    num_samples = len(full_range)
    num_channels = signals.shape[1]

    interpolated_signals = np.zeros((num_samples, num_channels), dtype=np.float32)
    interp_mask = np.zeros((num_samples, num_channels), dtype=np.int16)

    for ch_idx, ch_name in enumerate(channel_names):
        # Select interpolation method based on channel name.
        method = "linear" if ch_name.startswith("CH") else "nearest"
        interp_func = RegularGridInterpolator(
            (sample_indices,),
            signals[:, ch_idx],
            method=method,
            bounds_error=False,
            fill_value=None
        )
        # Interpolate on the full range.
        interpolated_channel = interp_func(full_range)
        interpolated_signals[:, ch_idx] = interpolated_channel

        # Create mask: mark points not in the original sample_indices as interpolated.
        original_mask = np.isin(full_range, sample_indices)
        interp_mask[:, ch_idx] = (~original_mask).astype(np.int16)

    logging.info("Signal interpolation completed.")
    return interpolated_signals, interp_mask

def append_interp_mask_to_signals(signals: np.ndarray, interp_mask: np.ndarray, channel_names: list) -> Tuple[np.ndarray, list]:
    """
    Append a combined interpolation mask as an additional channel to the signals.
    The combined mask is computed as the logical OR across all channels of the interpolation mask.

    Parameters:
    - signals: Original signal array (num_samples x num_channels)
    - interp_mask: Interpolation mask array of the same shape as signals
    - channel_names: List of original channel names

    Returns:
    - new_signals: Updated signals array with the appended mask column
    - new_channel_names: Updated list of channel names with "Misc_InterpMask" added
    """
    # Combine the mask along channels: if any channel was interpolated at a given sample, mark it as 1.
    combined_mask = np.any(interp_mask, axis=1).astype(np.int16).reshape(-1, 1)
    new_signals = np.hstack([signals, combined_mask])
    new_channel_names = channel_names.copy()
    new_channel_names.append("Misc_InterpMask")
    return new_signals, new_channel_names


def create_edf_file(data_st: Dict[str, Any], edf_file_path: str, start_datetime: datetime,
                    patient_name: str, patient_code: str) -> None:
    """
    Generate an EDF file from the provided signal data and metadata.

    Parameters:
    - data_st: Dictionary containing signal data and metadata, including:
        - "signals": The signal array (num_samples x num_channels)
        - "channel_names": List of channel names
        - "sampling_rate": Sampling frequency
        - "interp_mask": Interpolation mask array (always present, even if all zeros)
    - edf_file_path: Output path for the EDF file
    - start_datetime: Recording start datetime
    - patient_name: Patient/subject name
    - patient_code: Patient code in BIDS format
    """
    logging.info(f"Generating EDF file: {edf_file_path} (Patient: {patient_name})")
    out_dir = Path(edf_file_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = data_st["signals"]
    channel_names = data_st["channel_names"]
    sample_frequency = data_st["sampling_rate"]

    # Append the interpolation mask channel.
    # The 'interp_mask' is always present (it will be all zeros if no interpolation occurred).
    if "interp_mask" in data_st:
        interp_mask = data_st["interp_mask"]
        if interp_mask.shape != signals.shape:
            logging.error("Interpolation mask shape does not match signals. Skipping mask addition.")
        else:
            signals, channel_names = append_interp_mask_to_signals(signals, interp_mask, channel_names)

    n_channels = len(channel_names)
    edf_writer = pyedflib.EdfWriter(edf_file_path, n_channels=n_channels,
                                    file_type=pyedflib.FILETYPE_EDFPLUS)
    edf_writer.setPatientName(patient_name)
    edf_writer.setPatientCode(patient_code)
    edf_writer.setSex(0)  # Set sex to 0 (assumed default)
    edf_writer.setStartdatetime(start_datetime)

    channel_info = []
    for i, channel_name in enumerate(channel_names):
        # Compute physical minimum and maximum from the signals.
        physical_min = np.nanmin(signals[:, i])
        physical_max = np.nanmax(signals[:, i])
        if np.isnan(physical_min) or np.isnan(physical_max) or (physical_min == physical_max):
            physical_min, physical_max = -1000, 1000

        # Set dimension based on channel type.
        if channel_name == "Misc_InterpMask":
            dimension = "binary"
        elif channel_name.startswith("CH"):
            dimension = "uV"
        else:
            dimension = "trigger"

        ch_dict = {
            "label": channel_name,
            "dimension": dimension,
            "sample_frequency": sample_frequency,
            "physical_max": physical_max,
            "physical_min": physical_min,
            "digital_max": 32767,
            "digital_min": -32768,
        }
        channel_info.append(ch_dict)

    edf_writer.setSignalHeaders(channel_info)
    edf_writer.writeSamples(signals.T)
    edf_writer.writeAnnotation(0, -1, "Start Recording")
    edf_writer.close()

    logging.info(f"EDF file generated successfully: {edf_file_path}")

def load_samples(samples_path: Path) -> Dict:
    """
    load small dataset samples.json
    """
    if not samples_path.exists():
        return {}
    with samples_path.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    out = {}
    for subj, items in (data.get("subjects") or {}).items():
        out[subj] = {}
        for it in items or []:
            pod = it.get("post_op_day")
            if pod is None:
                continue
            tasks = it.get("tasks")
            out[subj][int(pod)] = {"tasks": tasks}
    return out


def generate_events_json(tsv_path: Path, mapped_task_name: str, logger) -> None:
    """
    If a same-named events.json does not exist, generate a minimal description based on the task type template.
    Does not read or depend on the TSV header; does not correct or modify the TSV.
    Templates use the "mapped task name" as the key:
      rest, sep, listening(=piano), pressing, reaching, word (if enabled)
    """
    # Target json path (still inferred by _events.tsv naming convention)
    json_path = tsv_path.with_name(tsv_path.name.replace("_events.tsv", "_events.json"))

    # Skip if already exists
    if json_path.exists():
        logger.info(f"[events.json] already exists, skip: {json_path}")
        return

    # Normalize task name (ensure mapped name is used)
    key = (mapped_task_name or "").lower()

    # Get template from config; generate entirely from template
    tmpl = config.templates.get(key)

    if not tmpl:
        logger.warning(f"[events.json] no template found for task '{key}', writing minimal stub: {json_path}")
        out = {}
    else:
        # If template is mutable, copy it to avoid external contamination
        try:
            import copy
            out = copy.deepcopy(tmpl)
        except Exception:
            out = dict(tmpl)

    # Optional recommended key (keep)
    out.setdefault("StimulusPresentation", {"OperatingSystem": "n/a", "SoftwareName": "n/a"})

    # Write file
    try:
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(out, jf, indent=4, ensure_ascii=False)
        logger.info(f"[events.json] generated: {json_path}")
    except Exception as e:
        logger.error(f"[events.json] write failed: {e}")

def create_ieeg_json_file(
    data_meta: dict,
    out_json_path: str,
    *,
    task_name: str = "",
    task_description: str = "",
    instructions: str = "n/a",
    power_line_frequency: int = 60,
    software_filters: str = "n/a",
    manufacturer: str = "CorTec GmbH",
    manufacturers_model_name: str = "Brain Interchange ONE",
    institution_name: str = "Osaka University Graduate School of Medicine",
    institutional_department_name: str = "n/a",
    institution_address: str = "n/a",
    # —— Only valid for ECoG; others fixed at 0 —__
    misc_channel_count_default: int = 0,  # Used only as fallback if TR cannot be determined from channel_names
    # —— Recording and hardware —__
    recording_type: str = "continuous",
    software_versions: str = "Unknown",
    ieeg_placement_scheme: str = "",
    ieeg_reference: str = (
        "The electrode arrays [CH11, CH27] were positioned into subdural space over the "
        "sensorimotor cortex on both hemispheres."
    ),
    electrode_manufacturer: str = "CorTec GmbH",
    electrode_manufacturers_model_name: str = "Brain Interchange ONE",
    ieeg_ground: str = "Upper back",
    electrical_stimulation: bool = False,
    electrical_stimulation_parameters: str = "n/a",
    # —— Minimal recommended keys —__
    device_serial_number: str = "n/a",
    ieeg_electrode_groups: str = "n/a",
    subject_artefact_description: str = "n/a",
) -> None:
    # Sampling rate and duration
    sfreq = data_meta.get("sampling_rate", DEFAULT_SAMPLING_FREQUENCY)
    signals = data_meta.get("signals")
    if signals is not None and hasattr(signals, "shape") and signals.shape:
        try:
            n_samples = int(signals.shape[0])
            recording_duration_sec = float(n_samples) / float(sfreq) if sfreq else 0.0
        except Exception:
            recording_duration_sec = 0.0
    else:
        recording_duration_sec = 0.0

    # —— Channel counts (only rely on two items you actually have + name list for TR detection) —__
    ecog_count = int(data_meta.get("channel_num_signal", 0) or 0)
    logic_total = int(data_meta.get("channel_num_trigger", 0) or 0)  # Historical naming: this is actually "total logic"
    names = data_meta.get("channel_names") or []

    # Count TR channels
    try:
        trigger_count = sum(1 for n in names if isinstance(n, str) and n.upper().startswith("TR") and n[2:].isdigit())
    except Exception:
        trigger_count = 0

    # If no names can be used for detection, fallback to 0 and treat logic_total as misc
    if trigger_count == 0 and not any((isinstance(n, str) and n.upper().startswith("TR")) for n in names):
        # No recognizable TR names
        misc_count = max(logic_total - trigger_count, 0) if logic_total else int(misc_channel_count_default or 0)
    else:
        misc_count = max(logic_total - trigger_count, 0)

    ieeg_json_dict = {
        # Core task metadata
        "TaskName": task_name,
        "TaskDescription": task_description,
        "Instructions": instructions,
        "SamplingFrequency": sfreq,
        "PowerLineFrequency": power_line_frequency,
        "HardwareFilters": {
            "HighpassFilter": {"CutoffFrequency": DEFAULT_LOW_CUTOFF},
            "LowpassFilter": {"CutoffFrequency": DEFAULT_HIGH_CUTOFF},
        },
        "SoftwareFilters": software_filters,
        "Manufacturer": manufacturer,
        "ManufacturersModelName": manufacturers_model_name,
        "InstitutionName": institution_name,
        "InstitutionalDepartmentName": institutional_department_name,
        "InstitutionAddress": institution_address,

        # —— Minimal recommended keys —__
        "DeviceSerialNumber": device_serial_number,
        "iEEGElectrodeGroups": ieeg_electrode_groups,
        "SubjectArtefactDescription": subject_artefact_description,

        # —— Channel counts —__
        "ECOGChannelCount": ecog_count,
        "SEEGChannelCount": 0,
        "EEGChannelCount": 0,
        "EOGChannelCount": 0,
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "TriggerChannelCount": trigger_count,
        "MiscChannelCount": misc_count,

        # —— Recording properties and electrode info —__
        "RecordingDuration": recording_duration_sec,
        "RecordingType": recording_type,
        "SoftwareVersions": software_versions,
        "iEEGPlacementScheme": ieeg_placement_scheme,
        "iEEGReference": ieeg_reference,
        "ElectrodeManufacturer": electrode_manufacturer,
        "ElectrodeManufacturersModelName": electrode_manufacturers_model_name,
        "iEEGGround": ieeg_ground,
        "ElectricalStimulation": bool(electrical_stimulation),
        "ElectricalStimulationParameters": electrical_stimulation_parameters,
    }

    out_dir = os.path.dirname(out_json_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(ieeg_json_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"iEEG JSON file generated successfully: {out_json_path}")
    except Exception as e:
        logging.error(f"Failed to write iEEG JSON file: {e}")

def unify_channel_data(broken_channels: list,
                       progressive_channels: dict,
                       default_channels: Dict[str, Dict[str, Any]],
                       measurement_date: datetime) -> Dict[str, Dict[str, Any]]:
    """
    Update the default channel configuration based on broken and progressive channel info.

    This function always includes the 'Misc_InterpMask' channel because the interpolation mask exists
    even when no interpolation is performed (in such cases, all mask values are 0).

    Parameters:
    - broken_channels: List of broken channels.
    - progressive_channels: Dictionary with progressive channel information.
    - default_channels: Default channel configuration obtained from get_default_channel_data().
    - measurement_date: The measurement date as a datetime object.

    Returns:
    - Updated channel configuration dictionary.
    """
    # Create a copy of the default configuration to avoid modifying the original.
    unified_channels = default_channels.copy()

    # Update broken channels
    for ch in broken_channels:
        if ch in unified_channels:
            unified_channels[ch]["status"] = "bad"
            unified_channels[ch]["status_description"] = "Hardware failure from the beginning"
        else:
            unified_channels[ch] = {
                "name": ch,
                "type": "ECOG",
                "units": "uV",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "sampling_frequency": 1000,
                "group": "n/a",
                "status": "bad",
                "status_description": "Hardware failure from the beginning"
            }

    # Update progressive channels info
    for ch, ch_info in progressive_channels.items():
        q_start_str = ch_info.get("QuestionStartDate")
        b_after_str = ch_info.get("BadAfter")
        if not q_start_str:
            continue
        q_start_dt = datetime.strptime(q_start_str, "%Y%m%d")
        if b_after_str:
            bad_after_dt = datetime.strptime(b_after_str, "%Y%m%d")
        else:
            bad_after_dt = None

        # If current date is before QuestionStartDate, maintain good status (unless already bad)
        if measurement_date < q_start_dt:
            if ch not in unified_channels or unified_channels[ch]["status"] != "bad":
                unified_channels[ch] = unified_channels.get(ch, {"name": ch})
                unified_channels[ch]["status"] = "good"
                unified_channels[ch]["status_description"] = "Stable before questionStart"
        else:
            # After questionStart, set as bad if past BadAfter, otherwise as questionable.
            if bad_after_dt and measurement_date >= bad_after_dt:
                unified_channels[ch] = unified_channels.get(ch, {"name": ch})
                unified_channels[ch]["status"] = "bad"
                unified_channels[ch]["status_description"] = f"Turned bad after {b_after_str}"
            else:
                if ch not in unified_channels or unified_channels[ch]["status"] != "bad":
                    unified_channels[ch] = unified_channels.get(ch, {"name": ch})
                    unified_channels[ch]["status"] = "bad"
                    # unified_channels[ch]["status"] = "questionable"
                    unified_channels[ch]["status_description"] = f"Impedance rising since {q_start_str}"

    # Always include the interpolation mask channel since interp_mask always exists.
    unified_channels["Misc_InterpMask"] = {
        "name": "Misc_InterpMask",
        "type": "MISC",
        "units": "binary",
        "low_cutoff": "n/a",
        "high_cutoff": "n/a",
        "sampling_frequency": default_channels.get("CH01", {}).get("sampling_frequency", 1000),
        "group": "n/a",
        "status": "good",
        "status_description": "Interpolation mask channel (1 indicates interpolated sample, 0 indicates original)"
    }

    return unified_channels

def get_default_channel_data(channel_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of default channel information.

    Returns:
    - A dictionary with channel names as keys and default channel settings as values.
    """
    default_channels = {}
    for ch_name in channel_names:
        if ch_name.startswith("CH"):
            default_channels[ch_name] = {
                "name": ch_name,
                "type": "ECOG",
                "units": "uV",
                "low_cutoff": DEFAULT_LOW_CUTOFF,
                "high_cutoff": DEFAULT_HIGH_CUTOFF,
                "sampling_frequency": DEFAULT_SAMPLING_FREQUENCY,
                "group": "n/a",
                "status": "good",
            }
        elif ch_name.startswith("TR"):
            default_channels[ch_name] = {
                "name": ch_name,
                "type": "TRIG",
                "units": "n/a",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "sampling_frequency": DEFAULT_SAMPLING_FREQUENCY,
                "group": "n/a",
                "status": "good",
            }
        elif "Misc_InterpMask" in ch_name:
            default_channels["Misc_InterpMask"] = {
                "name": "Misc_InterpMask",
                "type": "MISC",
                "units": "binary",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "sampling_frequency": DEFAULT_SAMPLING_FREQUENCY,
                "group": "n/a",
                "status": "good",
                "status_description": "Interpolation mask channel (1 indicates interpolated sample, 0 indicates original)"
            }
        else:
            default_channels[ch_name] = {
                "name": ch_name,
                "type": "MISC",
                "units": "n/a",
                "low_cutoff": "n/a",
                "high_cutoff": "n/a",
                "sampling_frequency": DEFAULT_SAMPLING_FREQUENCY,
                "group": "n/a",
                "status": "good",
            }
    return default_channels

def create_channels_tsv_file(out_tsv_path: str, data_st: dict, measurement_date: datetime,
                             broken_channels: Optional[list] = None,
                             progressive_channels: Optional[Dict[str, Dict[str, str]]] = None) -> None:
    """
    Generate channels.tsv file by unifying channel configuration.

    Parameters:
    - data_st: Dictionary containing raw channel information, such as data_st["channel_names"] and sampling rate.
    - broken_channels: List of broken channels.
    - progressive_channels: Dictionary with progressive channel information.
    """
    # Get the default channel configuration.
    default_channels = get_default_channel_data(data_st["channel_names"])

    # Generate unified channel configuration
    unified_channels = unify_channel_data(broken_channels or [],
                                          progressive_channels or {},
                                          default_channels,
                                          measurement_date)

    out_dir = os.path.dirname(out_tsv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "group", "status", "status_description"])
        for ch_name in unified_channels.keys():
            ch_info = unified_channels[ch_name]
            writer.writerow([
                ch_info.get("name", ch_name),
                ch_info.get("type", "n/a"),
                ch_info.get("units", "n/a"),
                ch_info.get("low_cutoff", "n/a"),
                ch_info.get("high_cutoff", "n/a"),
                ch_info.get("sampling_frequency", "n/a"),
                ch_info.get("group", "n/a"),
                ch_info.get("status", "n/a"),
                ch_info.get("status_description", "n/a")
            ])
    logging.info(f"Channels TSV file generated successfully: {out_tsv_path}")

def load_impedance_as_dict(
    date_str: str,
    impedance_source_dir: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Read the latest impedance CSV for the given date and return a mapping
    from channel name to impedance value.
    
    Behavior:
      - Looks for files in `impedance_source_dir` whose names start with `date_str`
        and end with `.csv`, e.g., "20230718T131216.csv".
      - Sorts the matched files lexicographically and uses the last one
        as the latest measurement of that day.
      - Expects exactly 32 rows in the CSV, one per channel CH01..CH32.
      - Returns a dict: {"CH01": float_value, ..., "CH32": float_value}.
      - If no valid impedance values are found, returns all channels with NaN.
    
    Args:
        date_str: Date prefix string "YYYYMMDD" used to filter files.
        impedance_source_dir: Directory containing impedance CSV files.
        logger: Logger for info/warning/error messages.
    
    Returns:
        Dict[str, float]: Mapping from channel name to impedance value.
    """
    try:
        all_files = os.listdir(impedance_source_dir)
    except Exception as e:
        logger.error(f"Failed to list files in {impedance_source_dir}: {e}")
        return {f"CH{i+1:02d}": "n/a" for i in range(32)}

    # Filter CSVs for the given date prefix
    impedance_files = [
        f for f in all_files
        if f.endswith(".csv")
        and f.startswith(date_str)
        and os.path.isfile(os.path.join(impedance_source_dir, f))
    ]

    if not impedance_files:
        logger.info(f"No impedance files found for date {date_str}.")
        return {f"CH{i+1:02d}": "n/a" for i in range(32)}

    # Use the latest file (lexicographically greatest) for the day
    latest_file = sorted(impedance_files)[-1]
    latest_path = os.path.join(impedance_source_dir, latest_file)

    try:
        with open(latest_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            rows = list(reader)
    except Exception as e:
        logger.error(f"Error reading {latest_path}: {e}")
        return {f"CH{i+1:02d}": "n/a" for i in range(32)}

    if len(rows) != 32:
        logger.warning(
            f"{latest_path} does not have 32 rows (actual: {len(rows)})."
        )
        return {f"CH{i+1:02d}": "n/a" for i in range(32)}

    impedance: Dict[str, float] = {}
    for idx, row in enumerate(rows):
        ch_name = f"CH{idx+1:02d}"
        if not row:
            impedance[ch_name] = "n/a"
            continue
        raw_val = row[0].strip()
        try:
            val = float(raw_val)
        except ValueError:
            logger.warning(
                f"{latest_path}: cannot parse impedance '{raw_val}' for {ch_name}; set NaN."
            )
            val = "n/a"
        impedance[ch_name] = val

    # If everything is NaN
    if np.all(np.isnan(list(impedance.values()))):
        logger.info(f"No valid impedance values parsed from {latest_path}; returning all NaN.")

    return impedance

def create_coordsystem_json_fallback(sub_id: str,
                                     ses_id: Optional[str] = None,
                                     bids_root: str = ".",
                                     logger: logging.Logger = None,
                                     space: Optional[str] = None) -> str:
    """
    Generate sub-<id>_ses-<id>_coordsystem.json (fallback, no coordinates) for each session.
    Core keys:
      - iEEGCoordinateSystem: "Other"
      - iEEGCoordinateSystemDescription: explanation for "no coordinates"
      - iEEGCoordinateUnits: "n/a"
    If the file already exists, do not overwrite.
    """
    space_tag = f"_space-{space}" if space else ""

    if ses_id is None:
        ieeg_dir = os.path.join(bids_root, f"sub-{sub_id}", "ieeg")
        out_name = f"sub-{sub_id}{space_tag}_coordsystem.json"
    else:
        ieeg_dir = os.path.join(bids_root, f"sub-{sub_id}", f"ses-{ses_id}", "ieeg")
        out_name = f"sub-{sub_id}_ses-{ses_id}{space_tag}_coordsystem.json"

    os.makedirs(ieeg_dir, exist_ok=True)
    out_path = os.path.join(ieeg_dir, out_name)

    if os.path.exists(out_path):
        logger.info(f"[coordsystem.json] already exists, skip: {out_path}")
        return out_path

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config.coord_description, f, indent=4, ensure_ascii=False)
        logger.info(f"[coordsystem.json] generated: {out_path}")
    except Exception as e:
        logger.error(f"[coordsystem.json] write failed: {e}")

    return out_path

def create_electrodes_tsv(sub_id: str,
                          ses_id: Optional[str] = None,
                          bids_root: str = ".",
                          logger: Optional[logging.Logger] = None,                          
                          space: Optional[str] = None,
                          impedance: Optional[Dict[str, float]] = None,
                          ) -> str:
    """
    生成 electrodes.tsv（包含 x/y/z），基于 bids_config.COORDINATE（新结构）：
      COORDINATE = {
        "defaults": {"size": 2.0, "material": "Pt/Ir 90/10", "type": "ECoG", ...可选manufacturer},
        "group_defaults": {"Left_Sheet": {"hemisphere": "L"}, ...},
        "electrodes": [{"name":"CH01","x":0.0,"y":0.0,"z":0.0,"group":"Left_Sheet"}, ...]
      }

    若 bids_config.COORDINATE 不存在，则回退为：
      name_list = CH01..CH32，x/y/z 均为 "n/a"。

    - ses_id 为 None 时输出到 subject 层级:
        sub-<id>/ieeg/sub-<id>[_space-<label>]_electrodes.tsv
    - ses_id 不为空时输出到 session 层级:
        sub-<id>/ses-<ses>/ieeg/sub-<id>_ses-<ses>[_space-<label>]_electrodes.tsv

    注意：请确保同级目录下提供 coordsystem.json 以声明坐标系与单位。
    """
    bids_root_path = Path(bids_root)
    space_tag = f"_space-{space}" if space else ""
    if ses_id is None:
        ieeg_dir = bids_root_path / f"sub-{sub_id}" / "ieeg"
        out_path = ieeg_dir / f"sub-{sub_id}{space_tag}_electrodes.tsv"
    else:
        ieeg_dir = bids_root_path / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg"
        out_path = ieeg_dir / f"sub-{sub_id}_ses-{ses_id}{space_tag}_electrodes.tsv"

    ieeg_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info(f"[electrodes.tsv] already exists, skip: {out_path}")
        return str(out_path)

    # 尝试读取 bids_config.COORDINATE（新结构）
    config_data = None
    try:
        config_data = getattr(config, "COORDINATE", None)
    except Exception as e:
        logger.warning(f"[electrodes.tsv] bids_config not available or import failed: {e}")

    # Prepare header
    header = ["name", "x", "y", "z", "size", "material", "group", "hemisphere", "type", "impedance"]
    rows = []
    if isinstance(config_data, dict) and isinstance(config_data.get("electrodes"), list):
        defaults = config_data.get("defaults", {}) or {}
        group_defaults = config_data.get("group_defaults", {}) or {}

        default_size = defaults.get("size", "n/a")
        default_material = defaults.get("material", "n/a")
        default_type = defaults.get("type", "n/a")

        for ele in config_data["electrodes"]:
            name = ele.get("name", "n/a")
            # 坐标允许缺失则写 'n/a'
            x = ele.get("x", "n/a")
            y = ele.get("y", "n/a")
            z = ele.get("z", "n/a")
            group = ele.get("group", "n/a")

            # hemisphere 来自 group_defaults[group].hemisphere，否则 'n/a'
            hemi = "n/a"
            if group in group_defaults:
                hemi = group_defaults[group].get("hemisphere", "n/a")

            # 可允许电极级别覆盖（若你后续扩展在 electrode 上直接提供 size/material/type/manufacturer）
            size = ele.get("size", default_size)
            material = ele.get("material", default_material)
            etype = ele.get("type", default_type)

            # 保证数字型坐标被正常写出（否则为 'n/a'）
            def _num_or_na(v):
                try:
                    return float(v)
                except Exception:
                    return "n/a"

            x = _num_or_na(x)
            y = _num_or_na(y)
            z = _num_or_na(z)
            
            # impedance 来自传入的 impedance dict，否则 'n/a'
            if impedance and isinstance(impedance, dict) and name in impedance: 
                try:
                    imp_val = float(impedance[name])
                except Exception:
                    imp_val = "n/a"
            rows.append([name, x, y, z, size, material, group, hemi, etype,imp_val])

    else:
        # 回退逻辑：无 COORDINATE，就写 CH01..CH32，x/y/z 全 'n/a'
        name_list = [f"CH{i:02d}" for i in range(1, 33)]
        for nm in name_list:
            imp_val = "n/a"
            if impedance is not None and nm in impedance:
                try:
                    imp_val = float(impedance[nm])
                except Exception:
                    imp_val = "n/a"
            rows.append([nm, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", imp_val])
        logger.info("[electrodes.tsv] COORDINATE not found; wrote fallback CH01..CH32 with 'n/a' coords.")

    # 写文件
    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)
        logger.info(f"[electrodes.tsv] generated: {out_path}")
    except Exception as e:
        logger.error(f"[electrodes.tsv] write failed: {e}")

    # Friendly reminder: Please ensure the coordsystem.json at the same level sets the coordinate system and units (e.g., Units: 'mm')
    return str(out_path)

def create_electrodes_json(sub_id: str,
                            ses_id: Optional[str] = None,
                            bids_root: str = ".",
                            logger: logging.Logger = None,
                            space: Optional[str] = None) -> str:
    """
    Generate sub-<id>_ses-<id>_electrodes.json for each sub-<id>_ses-<id>_electrodes.tsv.
    If the file already exists, do not overwrite.
    """
    space_tag = f"_space-{space}" if space else ""

    if ses_id is None:
        ieeg_dir = os.path.join(bids_root, f"sub-{sub_id}", "ieeg")
        out_name = f"sub-{sub_id}{space_tag}_electrodes.json"
    else:
        ieeg_dir = os.path.join(bids_root, f"sub-{sub_id}", f"ses-{ses_id}", "ieeg")
        out_name = f"sub-{sub_id}_ses-{ses_id}{space_tag}_electrodes.json"

    os.makedirs(ieeg_dir, exist_ok=True)
    out_path = os.path.join(ieeg_dir, out_name)

    if os.path.exists(out_path):
        logger.info(f"[electrodes.json] already exists, skip: {out_path}")
        return out_path

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config.electrodes_description, f, indent=4, ensure_ascii=False)
        logger.info(f"[electrodes.json] generated: {out_path}")
    except Exception as e:
        logger.error(f"[electrodes.json] write failed: {e}")

    return out_path

def detect_01010101_pattern(trigger_data: np.ndarray, logger: logging.Logger, threshold_count: int = 50) -> bool:
    """
    Detect if the trigger data contains a significant number of alternating patterns (e.g., 01010101).

    Parameters:
    - trigger_data: A numpy array containing trigger data (0s and 1s). Can be 1D or 2D (each column is a channel).
    - logger: Logger object for debug logging.
    - threshold_count: Threshold count; if the total number of detected patterns across channels
                        meets or exceeds this value, return True.

    Returns:
    - True if the number of alternating patterns meets or exceeds the threshold; otherwise False.
    """
    count_pattern = 0

    # If trigger_data is 2D, iterate over each channel.
    if trigger_data.ndim == 2:
        for col in range(trigger_data.shape[1]):
            col_data = trigger_data[:, col]
            try:
                # Create sliding windows of length 8 from the 1D column data.
                windows = np.lib.stride_tricks.sliding_window_view(col_data, window_shape=8)
                diffs = np.diff(windows, axis=1)
                # For an alternating pattern, all differences should be non-zero.
                valid_windows = np.all(diffs != 0, axis=1)
                count_pattern += np.sum(valid_windows)
            except Exception:
                # Fallback for older numpy versions or errors: manual loop.
                for i in range(len(col_data) - 7):
                    segment = col_data[i:i+8]
                    if all(segment[j] != segment[j+1] for j in range(7)):
                        count_pattern += 1
    else:
        # For 1D array input.
        try:
            windows = np.lib.stride_tricks.sliding_window_view(trigger_data, window_shape=8)
            diffs = np.diff(windows, axis=1)
            valid_windows = np.all(diffs != 0, axis=1)
            count_pattern = np.sum(valid_windows)
        except Exception:
            for i in range(len(trigger_data) - 7):
                segment = trigger_data[i:i+8]
                if all(segment[j] != segment[j+1] for j in range(7)):
                    count_pattern += 1

    logger.debug(f"Number of alternating patterns detected: {count_pattern}")
    return count_pattern >= threshold_count
