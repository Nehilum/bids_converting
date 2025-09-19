# -*- coding: utf-8 -*-
"""
最適化された utils.py モジュール
機能：
　信号データの読み込み、補間、EDFファイル生成、iEEG JSONファイル生成、
　channels.tsv生成など、各種補助関数を提供する。
"""

import logging
import warnings
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyedflib

# ログ記録器の設定（メインプログラムで setup_logging を呼び出して統一的に使用）
def setup_logging(info_log: str = "process.log", error_log: str = "error.log") -> logging.Logger:
    """
    ログ記録器を設定する関数。
    引数：
        info_log: infoレベルのログを出力するファイルパス
        error_log: errorレベル以上のログを出力するファイルパス
    戻り値：
        設定済みの Logger オブジェクト
    """
    logger = logging.getLogger("bids_logger")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # infoログ用ハンドラ
    fh_info = logging.FileHandler(info_log, mode='w', encoding='utf-8')
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(formatter)
    logger.addHandler(fh_info)

    # errorログ用ハンドラ
    fh_error = logging.FileHandler(error_log, mode='w', encoding='utf-8')
    fh_error.setLevel(logging.WARNING)
    fh_error.setFormatter(formatter)
    logger.addHandler(fh_error)

    # コンソール用ハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# pyedflib に関する警告を無視する
warnings.filterwarnings('ignore')


def get_post_op_day(date_str: str, surgery_date_str: str = "20240501") -> int:
    """
    測定日と手術日から術後日数を計算する関数
    引数：
        date_str: 測定日（"YYYYMMDD"形式）
        surgery_date_str: 手術日（"YYYYMMDD"形式）、デフォルトは "20240501"
    戻り値：
        術後日数（整数）
    """
    fmt = "%Y%m%d"
    measure_date = datetime.strptime(date_str, fmt)
    surgery_date = datetime.strptime(surgery_date_str, fmt)
    return (measure_date - surgery_date).days


def extract_date_from_filename(filename: str) -> datetime:
    """
    ファイル名から日付および時刻情報を抽出し、datetimeオブジェクトに変換する
    引数：
        filename: 日付時刻情報を含むファイル名（例："cortec_20240501T123456..."）
    戻り値：
        datetimeオブジェクト
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
    Cortec データを読み込み、対応する .hdr と .bin ファイルを処理し、必要に応じて補間を実施する
    引数：
        data_fpath: データファイルのパス（.bin または .hdr で終わる必要があります）
        interp_method: 補間方法。現在は "linear+nearest" に対応
    戻り値：
        信号データとメタ情報を含む辞書
    """
    logging.info(f"{data_fpath} から信号を読み込みます。補間方法: {interp_method}")
    
    if not data_fpath.endswith((".bin", ".hdr")):
        raise ValueError(f"{data_fpath}: 無効なファイル拡張子です")

    base_fpath = os.path.splitext(data_fpath)[0]
    hdr_fpath = f"{base_fpath}.hdr"
    bin_fpath = f"{base_fpath}.bin"

    # hdr ファイルの読み込み
    with open(hdr_fpath, "r", encoding="utf-8") as f:
        hdr_lines = f.readlines()
    main_header = hdr_lines[0].strip()
    reference_channel_line = hdr_lines[1].strip() if len(hdr_lines) >= 2 else None
    amplification = hdr_lines[2].strip() if len(hdr_lines) >= 3 else None
    ground_line = hdr_lines[3].strip() if len(hdr_lines) >= 4 else None
    ground = ground_line.lower() if ground_line and ground_line.lower() in ["true", "false"] else "false"

    header_parts = main_header.split(";")
    if len(header_parts) != 7:
        raise ValueError(f"{hdr_fpath}: ヘッダーのフォーマットが正しくありません")
    sampling_rate = float(header_parts[1])
    threshold_high = float(header_parts[2])
    threshold_low = float(header_parts[3])
    ch_num_signal = int(header_parts[4])
    ch_num_logic = int(header_parts[5])
    ch_num_total = ch_num_signal + ch_num_logic
    channel_names = header_parts[6].split(":")
    if len(channel_names) != ch_num_total:
        raise ValueError(f"{hdr_fpath}: チャネル数が一致しません。期待: {ch_num_total}個、実際: {len(channel_names)}個")

    # 参照チャネルの処理
    if reference_channel_line and reference_channel_line.isdigit():
        ref_idx = int(reference_channel_line)
        if 0 <= ref_idx < ch_num_total:
            reference_channel = channel_names[ref_idx]
        else:
            reference_channel = None
    else:
        reference_channel = None

    # bin ファイルの読み込み
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

    if interp_method == "linear+nearest":
        interpolated_signals, interp_mask = interpolate_signals(signals, sample_index, channel_names)
        data_st["signals"] = interpolated_signals
        data_st["interp_mask"] = interp_mask

    logging.info(f"{data_fpath} の信号データを正常に読み込みました")
    return data_st


def interpolate_signals(signals: np.ndarray, sample_indices: np.ndarray, channel_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    信号の補間処理を実施する。 "CH" で始まるチャネルは線形補間、その他のチャネルは最近傍補間を使用する
    引数：
        signals: 元の信号配列（サンプル数 × チャネル数）
        sample_indices: 元のサンプリングインデックスの配列
        channel_names: チャネル名のリスト
    戻り値：
        補間後の信号配列と、補間されたサンプルを示すマスク（補間点は1、元の点は0）
    """
    logging.info("信号の補間処理を開始します")
    if signals.size == 0:
        return signals, np.zeros_like(signals, dtype=np.int16)
    
    full_range = np.arange(1, sample_indices[-1] + 1)
    num_samples = len(full_range)
    num_channels = signals.shape[1]
    interpolated_signals = np.zeros((num_samples, num_channels), dtype=np.float32)
    interp_mask = np.zeros((num_samples, num_channels), dtype=np.int16)

    for ch_idx, ch_name in enumerate(channel_names):
        method = "linear" if ch_name.startswith("CH") else "nearest"
        interp_func = RegularGridInterpolator((sample_indices,), signals[:, ch_idx],
                                              method=method, bounds_error=False, fill_value=None)
        interpolated_channel = interp_func(full_range)
        interpolated_signals[:, ch_idx] = interpolated_channel
        # 元のサンプルでない点を補間点としてマークする
        original_mask = np.isin(full_range, sample_indices)
        interp_mask[:, ch_idx] = (~original_mask).astype(np.int16)

    logging.info("信号の補間処理が完了しました")
    return interpolated_signals, interp_mask


def create_edf_file(data_st: Dict[str, Any], edf_file_path: str, start_datetime: datetime,
                    patient_name: str, patient_code: str) -> None:
    """
    指定された信号データとメタ情報から EDF ファイルを生成する
    引数：
        data_st: 信号データおよびメタ情報を含む辞書
        edf_file_path: EDF ファイルの保存パス
        start_datetime: 記録開始時刻
        patient_name: 患者（被験者）の名前
        patient_code: 患者コード（BIDS形式）
    """
    logging.info(f"EDF ファイルを生成します: {edf_file_path}（患者: {patient_name}）")
    out_dir = Path(edf_file_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = data_st["signals"]
    channel_names = data_st["channel_names"]
    sample_frequency = data_st["sampling_rate"]

    # 補間マスクが存在する場合、全チャネルのマスクを1列にまとめて追加する
    if "interp_mask" in data_st:
        interp_mask = data_st["interp_mask"]
        if interp_mask.shape != signals.shape:
            logging.error("interp_mask の形状が一致しません。mask の追加をスキップします。")
        else:
            # 各サンプルについて、いずれかのチャネルで補間が行われたかを判定
            # すなわち、行ごとに np.any() を取って1次元にまとめ、列ベクトルに変換する
            combined_mask = np.any(interp_mask, axis=1).astype(np.int16).reshape(-1, 1)
            signals = np.hstack([signals, combined_mask])
            channel_names.append("Misc_InterpMask")

    n_channels = len(channel_names)
    edf_writer = pyedflib.EdfWriter(edf_file_path, n_channels=n_channels,
                                    file_type=pyedflib.FILETYPE_EDFPLUS)
    edf_writer.setPatientName(patient_name)
    edf_writer.setPatientCode(patient_code)
    edf_writer.setSex(0)  # ここではすべて女性として 0 を設定
    edf_writer.setStartdatetime(start_datetime)

    channel_info = []
    for i, channel_name in enumerate(channel_names):
        physical_min = np.nanmin(signals[:, i])
        physical_max = np.nanmax(signals[:, i])
        if np.isnan(physical_min) or np.isnan(physical_max) or (physical_min == physical_max):
            physical_min, physical_max = -1000, 1000

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
    logging.info(f"EDF ファイルが正常に生成されました: {edf_file_path}")


def extract_datetime_from_filename(filename: str) -> datetime:
    """
    ファイル名から日付・時刻情報を抽出し、datetimeオブジェクトに変換する
    引数：
        filename: 日付・時刻情報を含むファイル名
    戻り値：
        datetimeオブジェクト
    """
    base_filename = Path(filename).name
    datetime_str = base_filename.split("T")[0] + base_filename.split("T")[1][:6]
    return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")


def convert_all_bin_to_edf(original_data_dir: str, output_parent_dir: str) -> None:
    """
    指定ディレクトリ内のすべての .bin ファイルを EDF 形式に変換し、指定ディレクトリに出力する
    引数：
        original_data_dir: 元データディレクトリのパス
        output_parent_dir: 出力先ディレクトリのパス
    """
    logging.info(f"{original_data_dir} 内の全 .bin ファイルを EDF 形式に変換し、{output_parent_dir} に保存します")
    for root, dirs, files in os.walk(original_data_dir):
        for file in files:
            if file.endswith(".bin"):
                bin_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, original_data_dir)
                output_dir = os.path.join(output_parent_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                edf_filename = os.path.splitext(file)[0] + ".edf"
                edf_file_path = os.path.join(output_dir, edf_filename)

                if 'Boss' in root:
                    patient_name = 'Boss'
                    patient_code = 'sub-02'
                elif 'Carol' in root:
                    patient_name = 'Carol'
                    patient_code = 'sub-01'
                else:
                    continue

                try:
                    data_st = load_signals_cortec(bin_file_path, interp_method="linear+nearest")
                    start_datetime = extract_datetime_from_filename(bin_file_path)
                    create_edf_file(data_st, edf_file_path, start_datetime, patient_name, patient_code)
                except Exception as e:
                    logging.error(f"{bin_file_path} の処理中にエラーが発生しました: {e}")


def create_ieeg_json_file(
    bin_file_path: str,
    out_json_path: str,
    task_name: str = "",
    task_description: str = "",
    instructions: str = "n/a",
    sampling_frequency: int = 1000,
    power_line_frequency: int = 60,
    hardware_filters: str = "High pass filter cut-off: ~2 Hz, Low pass filter cut-off: 325 Hz",
    software_filters: str = "n/a",
    manufacturer: str = "CorTec GmbH, Freiburg, Germany",
    manufacturers_model_name: str = "Brain Interchange ONE",
    institution_name: str = "Osaka University Graduate School of Medicine",
    institution_address: str = "n/a",
    ecog_channel_count: int = 32,
    seeg_channel_count: int = 0,
    eeg_channel_count: int = 0,
    eog_channel_count: int = 0,
    ecg_channel_count: int = 0,
    emg_channel_count: int = 0,
    misc_channel_count: int = 0,
    trigger_channel_count: int = 17,
    recording_type: str = "continuous",
    software_versions: str = "Unknown",
    ieeg_placement_scheme: str = "The electrode arrays were positioned into subdural space over the sensorimotor cortex on both hemispheres.",
    ieeg_reference: str = "参考電極の詳細はドキュメントを参照してください",
    electrode_manufacturer: str = "CorTec GmbH, Freiburg, Germany",
    electrode_manufacturers_model_name: str = "Brain Interchange ONE",
    ieeg_ground: str = "上背部",
    electrical_stimulation: bool = False,
    electrical_stimulation_parameters: str = "n/a"
) -> None:
    """
    bin ファイルおよびその他のメタデータから、BIDS規格に準拠した iEEG JSON ファイルを生成する
    引数：
        bin_file_path: 元の bin ファイルのパス
        out_json_path: 出力先 JSON ファイルのパス
        その他の引数：JSON 内に記載する各種メタ情報
    """
    try:
        data_st = load_signals_cortec(bin_file_path, interp_method="linear+nearest")
    except Exception as e:
        logging.error(f"{bin_file_path} からデータを読み込むのに失敗しました: {e}")
        return

    signals = data_st["signals"]
    n_samples = signals.shape[0]  # signals の行数＝サンプル数
    recording_duration_sec = n_samples / float(sampling_frequency)

    ieeg_json_dict = {
        "TaskName": task_name,
        "TaskDescription": task_description,
        "Instructions": instructions,
        "SamplingFrequency": sampling_frequency,
        "PowerLineFrequency": power_line_frequency,
        "HardwareFilters": hardware_filters,
        "SoftwareFilters": software_filters,
        "Manufacturer": manufacturer,
        "ManufacturersModelName": manufacturers_model_name,
        "InstitutionName": institution_name,
        "InstitutionAddress": institution_address,
        "ECOGChannelCount": ecog_channel_count,
        "SEEGChannelCount": seeg_channel_count,
        "EEGChannelCount": eeg_channel_count,
        "EOGChannelCount": eog_channel_count,
        "ECGChannelCount": ecg_channel_count,
        "EMGChannelCount": emg_channel_count,
        "MiscChannelCount": misc_channel_count,
        "TriggerChannelCount": trigger_channel_count,
        "RecordingDuration": recording_duration_sec,
        "RecordingType": recording_type,
        "SoftwareVersions": software_versions,
        "iEEGPlacementScheme": ieeg_placement_scheme,
        "iEEGReference": ieeg_reference,
        "SoftwareReferenceChannel": data_st["reference_channel"],
        "ElectrodeManufacturer": electrode_manufacturer,
        "ElectrodeManufacturersModelName": electrode_manufacturers_model_name,
        "iEEGGround": ieeg_ground,
        "GroundUsed": data_st["ground"],
        "ElectricalStimulation": electrical_stimulation,
        "ElectricalStimulationParameters": electrical_stimulation_parameters,
        "SessionLabelDescription": "session labeled by post-operative day",
        "Amplification": data_st["amplification"]
    }

    out_dir = os.path.dirname(out_json_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(ieeg_json_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"iEEG JSON ファイルの生成に成功しました: {out_json_path}")
    except Exception as e:
        logging.error(f"iEEG JSON ファイルの書き込みに失敗しました: {e}")


def get_default_channel_data() -> Dict[str, Dict[str, Any]]:
    """
    デフォルトのチャネル情報を辞書形式で返す関数
    戻り値：
        チャネル名をキーとしたデフォルトチャネル情報の辞書
    """
    default_channels = {}
    # ECOG チャネル（CH01～CH32）
    for i in range(1, 33):
        ch_name = f"CH{i:02d}"
        default_channels[ch_name] = {
            "name": ch_name,
            "type": "ECOG",
            "units": "uV",
            "low_cutoff": "2Hz",
            "high_cutoff": "325Hz",
            "sampling_frequency": 1000,
            "group": "n/a",
            "status": "good",
        }
    # EXT チャネル
    default_channels["EXT01"] = {
        "name": "EXT01",
        "type": "MISC",
        "units": "n/a",
        "low_cutoff": "n/a",
        "high_cutoff": "n/a",
        "sampling_frequency": 1000,
        "group": "n/a",
        "status": "good",
    }
    # TR チャネル（TR01～TR16）
    for i in range(1, 17):
        tr_name = f"TR{i:02d}"
        default_channels[tr_name] = {
            "name": tr_name,
            "type": "MISC",
            "units": "n/a",
            "low_cutoff": "n/a",
            "high_cutoff": "n/a",
            "sampling_frequency": 1000,
            "group": "n/a",
            "status": "good",
        }
    # Interpolation mask チャネル
    default_channels["Misc_InterpMask"] = {
        "name": "Misc_InterpMask",
        "type": "MISC",
        "units": "binary",
        "low_cutoff": "n/a",
        "high_cutoff": "n/a",
        "sampling_frequency": 1000,
        "group": "n/a",
        "status": "good",
        "status_description": "Interpolation mask channel (1=interpolated sample, 0=original)"
    }
    return default_channels


def create_channels_tsv_file(
    out_tsv_path: str,
    data_st: dict,
    # updated_channels 可能由外部传入, 用于标记 bad / questionable / reference 等
    updated_channels: Optional[Dict[str, Dict[str, str]]] = None,
    # 当下(本session)的软件参考通道, 若没有则为 None
    software_ref_channel: Optional[str] = None,
):    
    """
    生成 channels.tsv, 包含以下字段:
      name, type, units, low_cutoff, high_cutoff, sampling_frequency, group, status, status_description
    
    参数:
    - data_st: 
        必须包含:
          data_st["channel_names"]: 不含插值掩码通道(原始通道列表)
          data_st["sampling_rate"]
          data_st["interp_mask"] (可选, 如果存在, 说明edf里有Misc_InterpMask)
    - updated_channels:
        一个dict, 键=channel_name, 值={ "status":..., "status_description":... }等
        若某通道在 updated_channels 里没有, 则保持默认
    - software_ref_channel:
        若有指定 CHxx 作为软件参考, 在此通道的status_description中注明
    """

    channel_data = get_default_channel_data()

    if updated_channels:
        for ch_name, new_info in updated_channels.items():
            if ch_name in channel_data:
                channel_data[ch_name].update(new_info)
                logging.info(f"チャネル {ch_name} の情報を更新しました: {new_info}")
            else:
                logging.warning(f"チャネル {ch_name} はデフォルト設定に存在しないため、更新をスキップします。")

    out_dir = os.path.dirname(out_tsv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "group", "status", "status_description"])
        for ch_name in sorted(channel_data.keys()):
            ch_info = channel_data[ch_name]
            writer.writerow([
                ch_info["name"],
                ch_info["type"],
                ch_info["units"],
                ch_info["low_cutoff"],
                ch_info["high_cutoff"],
                ch_info["sampling_frequency"],
                ch_info["group"],
                ch_info["status"],
                ch_info["status_description"]
            ])
    logging.info(f"Channels TSV ファイルの生成に成功しました: {out_tsv_path}")


def create_impedance_tsv(sub_id: str, ses_day: int, date_str: str, impedance_source_dir: str,
                         bids_output_dir: str, logger: logging.Logger) -> None:
    """
    生成阻抗 TSV 文件（BIDS格式）

    输入文件：
      - 阻抗文件存放在阻抗源目录 impedance_source_dir，该目录结构为：
            /work/project/ECoG_Monkey/01_Data/Impedance/<MonkeyName>
      - 阻抗文件命名格式为 "YYYYMMDDTHHMMSS.csv"（例如 "20230718T131216.csv"），
        每个文件中包含 32 行数据，每行对应 CH01 至 CH32 的阻抗值（单位为 uA）。

    输出文件：
      - 在 BIDS 输出目录的 session 文件夹中（例如 bids_output_dir/sub-<sub_id>/ses-day??），
        生成一个 TSV 文件，文件名为 "sub-<sub_id>_ses-day??_impedance.tsv"。
      - TSV 文件包含三列：channel_name, impedance_uA, measurement_time。
      - 如果一天内有多个阻抗文件，则每个文件生成 32 行，measurement_time 列写入对应的时间信息。
    """
     # 在阻抗源目录中筛选出文件名以当前日期 (date_str) 开头的 CSV 文件
    impedance_files = [
        f for f in os.listdir(impedance_source_dir)
        if os.path.isfile(os.path.join(impedance_source_dir, f))
           and f.endswith(".csv")
           and f.startswith(date_str)
    ]
    
    if not impedance_files:
        logger.info(f"指定日付 {date_str} の阻抗ファイルが見つからないため、スキップします。")
        return

    tsv_rows = []
    # 对所有阻抗文件进行处理，按文件名排序（可选）
    for imp_file in sorted(impedance_files):
        # 阻抗文件命名格式为 "YYYYMMDDTHHMMSS.csv"，提取“T”后面的部分作为测量时刻
        base_name = os.path.splitext(imp_file)[0]
        if 'T' in base_name:
            parts = base_name.split("T")
            if len(parts) >= 2:
                measure_time_str = parts[1]  # 例如 "131216"
            else:
                measure_time_str = "000000"
        else:
            measure_time_str = "000000"
        
        imp_file_path = os.path.join(impedance_source_dir, imp_file)
        try:
            with open(imp_file_path, "r", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                # 读取所有行，期望有 32 行（对应 CH01 到 CH32）
                imp_values = list(reader)
            if len(imp_values) != 32:
                logger.warning(f"{imp_file_path} の行数は 32 行ではありません（実際は {len(imp_values)} 行）。")
                continue
            # 对每一行生成 TSV 数据行，channel name 按顺序生成
            for idx, row in enumerate(imp_values):
                # 假设每行仅含一个阻抗值
                impedance_value = row[0] if row else ""
                channel_name = f"CH{idx+1:02d}"
                tsv_rows.append([channel_name, impedance_value, measure_time_str])
        except Exception as e:
            logger.error(f"{imp_file_path} の処理中にエラーが発生しました: {e}")
            continue

    if not tsv_rows:
        logger.info(f"日付 {date_str} の阻抗データが読み込まれませんでした。")

    # 构造输出路径：BIDS目录下的 sub-{sub_id}/ses-day{ses_day:02d} 文件夹中生成 TSV 文件
    ses_dir = os.path.join(bids_output_dir, f"sub-{sub_id}", f"ses-day{ses_day:02d}")
    os.makedirs(ses_dir, exist_ok=True)
    out_imp_name = f"sub-{sub_id}_ses-day{ses_day:02d}_impedance.tsv"
    out_imp_path = os.path.join(ses_dir, out_imp_name)
    
    try:
        with open(out_imp_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, delimiter="\t")
            writer.writerow(["channel_name", "impedance_uA", "measurement_time"])
            writer.writerows(tsv_rows)
        logger.info(f"阻抗 TSV ファイルが生成されました: {out_imp_path}")
    except Exception as e:
        logger.error(f"阻抗 TSV ファイルの生成に失敗しました: {e}")
        return

def detect_01010101_pattern(trigger_data: np.ndarray, logger: logging.Logger, threshold_count: int = 50) -> bool:
    """
    trigger データ中に "01010101" のような交互パターンが多数存在するかを検出する関数
    引数：
        trigger_data: numpy 配列（0または1を含むトリガーデータ）
        logger: ログ記録用オブジェクト
        threshold_count: 交互パターンの回数の閾値。これを超えた場合、異常と判断する
    戻り値：
        異常なパターンが検出された場合は True、そうでなければ False
    """
    count_pattern = 0
    for i in range(len(trigger_data) - 7):
        segment = trigger_data[i:i+8]
        if all(segment[j] != segment[j+1] for j in range(7)):
            count_pattern += 1
    logger.debug(f"交互パターンの検出回数: {count_pattern}")
    return count_pattern >= threshold_count
