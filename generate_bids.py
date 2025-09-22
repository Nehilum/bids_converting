# -*- coding: utf-8 -*-
"""
Optimized generate_bids.py
Function:
    Reads experimental data (e.g., from subjects "Boss" and "Carol"), converts it to BIDS-compliant format,
    and generates corresponding EDF, JSON, and TSV files.
"""

import logging
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import chardet
import argparse
import yaml

# Import required functions from the utility module
from utils_test_v1 import (
    setup_logging,
    extract_date_from_filename,
    load_signals_cortec,
    create_edf_file,
    create_ieeg_json_file,
    create_channels_tsv_file,
    get_post_op_day,
    detect_01010101_pattern,
    create_impedance_tsv,
    generate_events_json, # <--- 新增
    load_samples, # <--- 新增
    create_electrodes_tsv,      # <--- 新增
    create_coordsystem_json_fallback      # <--- 新增
)

# Mapping from subject name to BIDS ID
MAP_SUB_NAME_TO_ID = {
    "Carol": "monkeyc",
    "Boss": "monkeyb"
}
MONKEY_NAMES = ["Boss", "Carol"]

# Task mapping information (only tasks with use_flag=True are processed)
TASK_MAPPING_INFO = {
    "association": {
        "use_flag": False,
        "mapped_name": "pressing",
        "description": "",
    },
    "association_electric": {
        "use_flag": False,
        "mapped_name": "association_electric",
        "description": "",
    },
    "piano": {
        "use_flag": True,
        "mapped_name": "listening",
        "description": "",
    },
    "pressing": {
        "use_flag": True,
        "mapped_name": "pressing",
        "description": "",
    },
    "reaching": {
        "use_flag": True,
        "mapped_name": "reaching",
        "description": "",
    },
    "rest": {
        "use_flag": True,
        "mapped_name": "rest",
        "description": "",
    },
    "SEP": {
        "use_flag": True,
        "mapped_name": "sep",
        "description": "",
    },
    "word": {
        "use_flag": False,
        "mapped_name": "word",
        "description": "",
    }
}

# Data and output directories
DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/01_Data")
BIDS_DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/BIDS_test_clean")
CONFIG_FILE_PATH = Path("..") / "config.json"
SAMPLES_FILE_DEFAULT = Path("samples.yaml")

def main():
    """
    Main function: Process data for each subject and convert to BIDS format.
    """
    # Set up logging for both file and console outputs.
    logger = setup_logging(info_log="process.log", error_log="error.log")
    logger.info("Starting BIDS data conversion process.")

    # CLI 参数：支持 --samples
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=str, default=None,
                    help="Path to samples.yaml. If set, export only listed subject/post_op_day/tasks.")
    args = ap.parse_args()


    # Load configuration file
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file {CONFIG_FILE_PATH}: {e}")
        return

    # 读取样例过滤
    samples_filter = {}
    if args.samples:
        samples_filter = load_samples(Path(args.samples))
    else:
        # 也支持“默认路径存在就启用”的懒加载
        if SAMPLES_FILE_DEFAULT.exists():
            samples_filter = load_samples(SAMPLES_FILE_DEFAULT)
    if samples_filter:
        logger.info(f"Samples filtering enabled: {samples_filter}")
    else:
        logger.info("Samples filtering not enabled (full export).")

    # Process each subject
    for monkey_name in MONKEY_NAMES:
        process_subject(monkey_name, config, logger, samples_filter)

def process_subject(monkey_name: str, config: dict, logger: logging.Logger,
                    samples_filter: Dict) -> None:
    """
    Process all data for a single subject.
    
    Parameters:
    - monkey_name: Subject's name (e.g., "Boss" or "Carol")
    - config: Configuration loaded from the config file
    - logger: Logger object for logging messages
    """
    logger.info(f"Starting processing for subject {monkey_name}")
    sub_id = MAP_SUB_NAME_TO_ID.get(monkey_name, monkey_name.lower())
    subject_cfg = config.get("subjects", {}).get(monkey_name, {})

    # Retrieve subject-specific configuration: broken channels, progressive channels, and date range.
    broken_ch_list = subject_cfg.get("BrokenChannels", [])
    progressive_channels = subject_cfg.get("ProgressiveChannels", {})
    start_date_str = subject_cfg.get("start_date")  # e.g., "20221222"
    end_date_str   = subject_cfg.get("end_date")    # e.g., "20240501"
    op_day_str     = subject_cfg.get("OperationDay")  # e.g., "20221222"

    try:
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
        op_date = datetime.strptime(op_day_str, "%Y%m%d")
    except Exception as e:
        logger.error(f"Date format error for subject {monkey_name}: {e}")
        return
    
    # Get the subject's condition directory
    condition_folder_path = DATA_DIR_PATH / "Condition" / monkey_name
    if not condition_folder_path.is_dir():
        logger.error(f"Condition folder does not exist: {condition_folder_path}")
        return

    # Get all date folders in the condition directory
    dates = [d.name for d in condition_folder_path.iterdir() if d.is_dir()]
    logger.debug(f"Found date folders: {dates}")

    try:
        dates_sorted = sorted(dates, key=lambda d: datetime.strptime(d, "%Y%m%d"))
    except Exception as e:
        logger.error(f"Failed to sort date folders: {e}")
        return

    # Filter folders within the specified date range
    dates_filtered = [d for d in dates_sorted if start_date <= datetime.strptime(d, "%Y%m%d") <= end_date]
    logger.debug(f"Date folders within specified range: {dates_filtered}")

    # Process each date folder
    for date_str in dates_filtered:
        process_date(monkey_name, sub_id, date_str, op_day_str,
             broken_ch_list, progressive_channels, logger,
             samples_filter.get(monkey_name, {}))

def process_date(monkey_name: str, sub_id: str, date_str: str, op_day_str: str,
                 broken_ch_list: list, progressive_channels: dict, logger: logging.Logger,
                 subj_samples: Dict[int, Dict]) -> None:
    """
    Process all JSON files in a single date folder and generate corresponding BIDS files.
    
    Parameters:
    - monkey_name: Subject name
    - sub_id: BIDS subject ID
    - date_str: Date folder name in "YYYYMMDD" format
    - op_day_str: Operation day as a string (e.g., "20221222")
    - broken_ch_list: List of broken channels
    - progressive_channels: Progressive channel information
    - logger: Logger object for logging messages
    """
    logger.info(f"Processing date folder {date_str}")
    current_date = datetime.strptime(date_str, "%Y%m%d")
    post_op_day = get_post_op_day(date_str, op_day_str)
    ses_id = f"day{post_op_day:02d}"
    # 若启用样例过滤：只保留当前 subject 下指定的 post_op_day
    # subj_samples 形如 {149: {"tasks": ["rest"]}, 337: {"tasks": ["reaching"]}}
    allowed_tasks_for_day = None
    if subj_samples:
        if post_op_day not in subj_samples:
            logger.info(f"[samples] skip {monkey_name} {date_str} (post_op_day={post_op_day})")
            return
        allowed_tasks_for_day = subj_samples[post_op_day].get("tasks")  # 可能是 None

    json_folder_path = DATA_DIR_PATH / "Condition" / monkey_name / date_str
    if not json_folder_path.is_dir():
        logger.error(f"JSON folder does not exist: {json_folder_path}")
        return

    json_files = [f for f in json_folder_path.iterdir() if f.is_file() and f.suffix == ".json"]
    task_type_infos = {}
    scans_info = []  # To store information for scans.tsv

    # Process each JSON file to classify by task type
    for json_file in json_files:
        logger.info(f"Processing JSON file: {json_file.name}")
        try:
            raw_data = json_file.read_bytes()
            encoding = chardet.detect(raw_data)["encoding"]
            with json_file.open("r", encoding=encoding) as f:
                condition_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read/parse {json_file}: {e}")
            continue

        task_type = condition_data.get("Task Type")
        if task_type in TASK_MAPPING_INFO and TASK_MAPPING_INFO[task_type]["use_flag"]:
            task_type_infos.setdefault(task_type, []).append(json_file.stem)

    # Generate BIDS files for each task type
    for task_type, file_name_list in task_type_infos.items():
        # # Sort the list of file base names in chronological order based on the extracted datetime.
        file_name_list.sort(key=extract_date_from_filename)
        # # Initialize the run counter. This counter will be incremented for each file, 
        # ensuring that each run (i.e., each separate recording session for the same task on the same date) 
        # is assigned a unique sequential run number.
        run_num = 0
        mapped_task_name = TASK_MAPPING_INFO[task_type]["mapped_name"]
        task_description = TASK_MAPPING_INFO[task_type]["description"]
        # 若启用样例过滤，且为本 day 指定了 tasks，则仅保留这些任务（按映射后的名字匹配）
        if allowed_tasks_for_day is not None:
            if allowed_tasks_for_day and (mapped_task_name not in set(allowed_tasks_for_day)):
                logger.info(f"[samples] skip task {mapped_task_name} on post_op_day={post_op_day}")
                continue

        # Iterate through each base file name in the sorted list.
        for base_file_name in file_name_list:
            # Increment the run number for each file processed.
            run_num += 1
            edf_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_ieeg.edf"
            edf_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / edf_file_name

            bin_file_path = DATA_DIR_PATH / "CortecData" / monkey_name / date_str / f"{base_file_name}.bin"

            try:
                data_st = load_signals_cortec(str(bin_file_path), interp_method="linear+nearest")
            # Data quality control: Check trigger signal for anomalies in channels TR01 to TR04.
            # Extract indices of channels whose names are TR01, TR02, TR03, or TR04 from the channel names list.
                channel_names = data_st["channel_names"]
                trigger_indices = [i for i, ch in enumerate(channel_names) if ch in ["TR01", "TR02", "TR03", "TR04"]]
                if trigger_indices:
                    # Extract trigger data only from TR01 to TR04 channels.
                    trigger_data = data_st["signals"][:, trigger_indices]
                    if detect_01010101_pattern(trigger_data, logger):
                        logger.warning(f"{bin_file_path} trigger signal shows excessive alternating patterns, indicating possible data anomaly.")
                exp_start_datetime = extract_date_from_filename(base_file_name)
                create_edf_file(data_st, str(edf_file_path), exp_start_datetime, monkey_name, f"sub-{sub_id}")
            except Exception as e:
                logger.error(f"Error processing {bin_file_path}: {e}")
                continue

            # Copy events file
            events_file_name = f"{base_file_name}_events.tsv"
            events_file_path = DATA_DIR_PATH / "Events" / monkey_name / date_str / events_file_name
            bids_events_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_events.tsv"
            bids_events_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_events_file_name
            try:
                shutil.copy(str(events_file_path), str(bids_events_file_path))
            except Exception as e:
                logger.error(f"Failed to copy events file {events_file_path}: {e}")
            # generate events.json if not exists
            try:
                shutil.copy(str(events_file_path), str(bids_events_file_path))
                # 复制成功后：若无 json 则按任务模板生成
                generate_events_json(Path(bids_events_file_path), mapped_task_name, logger)
            except Exception as e:
                logger.error(f"Failed to copy events file {events_file_path}: {e}")

            # Generate channels.tsv file using unified channel configuration
            bids_channels_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_channels.tsv"
            bids_channels_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_channels_file_name
            try:
                create_channels_tsv_file(str(bids_channels_file_path), data_st,
                                           measurement_date=current_date,
                                           broken_channels=broken_ch_list,
                                           progressive_channels=progressive_channels)
            except Exception as e:
                logger.error(f"Failed to generate channels.tsv: {e}")

            # Generate iEEG JSON file
            bids_json_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_ieeg.json"
            bids_json_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_json_file_name
            try:
                create_ieeg_json_file(data_st, str(bids_json_file_path),
                                      task_name=mapped_task_name,
                                      task_description=task_description)
            except Exception as e:
                logger.error(f"Failed to generate iEEG JSON file: {e}")

            # Record scan information for scans.tsv
            relative_edf_path = str(Path("ieeg") / edf_file_name).replace("\\", "/")
            acquisition_time_str = exp_start_datetime.isoformat()
            scans_info.append([relative_edf_path, acquisition_time_str])

            # Generate impedance TSV file if impedance data exists
            create_impedance_tsv(sub_id, post_op_day, date_str,
                                 str(DATA_DIR_PATH / "Impedance" / monkey_name),
                                 str(BIDS_DATA_DIR_PATH), logger)

    # Generate scans.tsv file for the session if scans_info is not empty
    if scans_info:
        ses_dir_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}"
        if not ses_dir_path.exists():
            ses_dir_path.mkdir(parents=True, exist_ok=True)
        scans_tsv_name = f"sub-{sub_id}_ses-{ses_id}_scans.tsv"
        scans_tsv_path = ses_dir_path / scans_tsv_name

        try:
            with scans_tsv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["filename", "acq_time"])
                for row in scans_info:
                    writer.writerow(row)
            logger.info(f"Scans TSV file generated successfully: {scans_tsv_path}")
        except Exception as e:
            logger.error(f"Failed to generate scans.tsv: {e}")
        
         # ===== 新增：每个 session 生成 electrodes.tsv（无坐标）与 coordsystem.json（fallback） =====
        try:
            create_electrodes_tsv(
                sub_id=sub_id,
                ses_id=ses_id,
                bids_root=str(BIDS_DATA_DIR_PATH),
                logger=logger,
                material="Pt/Ir",        
                manufacturer="CorTec GmbH",
                group="n/a",
                hemisphere="n/a"
            )
            create_coordsystem_json_fallback(
                sub_id=sub_id,
                ses_id=ses_id,
                bids_root=str(BIDS_DATA_DIR_PATH),
                logger=logger
            )
        except Exception as e:
            logger.error(f"Failed to generate electrodes/coordsystem for {sub_id} {ses_id}: {e}")


if __name__ == "__main__":
    main()
