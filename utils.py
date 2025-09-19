from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta, date
import numpy as np
import pyedflib
import warnings
import logging
import json
import csv
import os

# ログ設定の再利用
logging.basicConfig(
    level=logging.DEBUG,  # ログレベルを設定
    format='%(asctime)s - %(levelname)s - %(message)s',  # ログフォーマットの設定
    filename='process.log',  # ログファイル名をメインプログラムと同じにする
    filemode='a'  # ファイルモード（a: 追記モード）
)

# pyedflibからの警告を非表示にする
warnings.filterwarnings('ignore')

# ファイル名から日付を抽出し、datetimeオブジェクトに変換する関数
def extract_date_from_filename(filename):
    prefix = "cortec_"
    if filename.startswith(prefix):
        filename = filename[len(prefix):]
    date_str = filename.split('T')[0]

    if 'B' not in filename:
        time_str = filename.split('T')[1]
    else:
        time_str = filename.split('T')[1].split('B')[0]

    return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

############################################
### NEW PART 2: 计算ses-dayXX的辅助函数  ###
############################################
def get_post_op_day(date_str, surgery_date_str="20240501"):
    """
    date_str: 当前测量的日期（YYYYMMDD）
    surgery_date_str: 手术日期（YYYYMMDD），默认为2024/05/01，可根据实际情况改
    返回：手术后第几天(整数)
    """
    fmt = "%Y%m%d"
    measure_date = datetime.strptime(date_str, fmt)
    surgery_date = datetime.strptime(surgery_date_str, fmt)
    delta_days = (measure_date - surgery_date).days
    return delta_days

def load_signals_cortec(data_fpath, interp_method="none"):
    logging.info(f"Loading signals from {data_fpath} with interpolation method: {interp_method}")
    
    if not data_fpath.endswith((".bin", ".hdr")):
        raise ValueError("{}: invalid file extension".format(data_fpath))

    base_fpath = os.path.splitext(data_fpath)[0]
    hdr_fpath = "{}.hdr".format(base_fpath)
    bin_fpath = "{}.bin".format(base_fpath)

    # ヘッダーファイルを読み込む
    with open(hdr_fpath, "r", encoding="utf-8") as f:
        hdr_lines = f.readlines()

    if len(hdr_lines) < 4:
        raise ValueError(f"{hdr_fpath}: insufficient header data")

    # ヘッダー情報の解析
    main_header = hdr_lines[0].strip()

    # 以下三个默认值先设置为 None
    reference_channel_line = None
    amplification = None
    ground_line = None

    if len(hdr_lines) >= 2:
        reference_channel_line = hdr_lines[1].strip()  # 可能是 "27", "0", 或 ""...
    if len(hdr_lines) >= 3:
        amplification = hdr_lines[2].strip()      # 可能是 "57.5dB" 或空
    if len(hdr_lines) >= 4:
        ground_line = hdr_lines[3].strip()            # "true" / "false" / 空
    
    # 如果早期hdr只有一行, 其余值为 None
    # 后面处理 reference_channel / amplification / ground 时注意判空

    # 对 ground_line 做处理
    if ground_line and ground_line.lower() in ["true", "false"]:
        ground = ground_line.lower()
    else:
        ground = "false"

    # reference_channel = hdr_lines[1].strip()
    # amplification = hdr_lines[2].strip()
    # ground = hdr_lines[3].strip()

    C = main_header.split(";")
    if len(C) != 7:
        raise ValueError(f"{hdr_fpath}: invalid header format")

    sampling_rate = float(C[1])
    threshold_high = float(C[2])
    threshold_low = float(C[3])
    ch_num_signal = int(C[4])
    ch_num_logic = int(C[5])
    ch_num_total = ch_num_signal + ch_num_logic
    channel_names = C[6].split(":")
    if len(channel_names) != ch_num_total:
        # raise ValueError("{}: invalid number of channels".format(data_fpath))
        raise ValueError(f"{hdr_fpath}: channel number mismatch. Expect {ch_num_total}, got {len(channel_names)}")

    if reference_channel_line and reference_channel_line.isdigit():
        ref_ch_idx = int(reference_channel_line)
        if ref_ch_idx >= 0 and ref_ch_idx < ch_num_total:
            # 在 hdr 设计上, "0" => CH01, "1" => CH02, ...
            # Python中 channel_names[0] = "CH01"
            # 也就是说, if ref_ch_idx=0 => channel_names[0] => "CH01"
            reference_channel = channel_names[ref_ch_idx]
        else:
            reference_channel = None
    else:
        reference_channel = None

    with open(bin_fpath, "rb") as f:
        data_uint8 = f.read()

    byte_num_per_sample = 8 + 4 + 4 * ch_num_total
    byte_num_to_remove = len(data_uint8) % byte_num_per_sample
    if byte_num_to_remove > 0:
        data_uint8 = data_uint8[:-byte_num_to_remove]

    samples = len(data_uint8) // byte_num_per_sample
    data_uint8 = np.frombuffer(data_uint8, dtype=np.uint8).reshape(
        samples, byte_num_per_sample
    )

    unix_time_ms = np.frombuffer(data_uint8[:, 0:8].tobytes(), dtype=np.int64)
    sample_index = np.frombuffer(data_uint8[:, 8:12].tobytes(), dtype=np.uint32)
    signals = np.frombuffer(data_uint8[:, 12:].tobytes(), dtype=np.float32).reshape(
        samples, ch_num_total
    )

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
        # data_st["signals"] = interpolate_signals(
        #     data_st["signals"],
        #     data_st["sample_index"],
        #     data_st["channel_names"],
        #     method=interp_method,
        # )
        interpolated_signals, interp_mask = interpolate_signals(
            signals,
            sample_index,
            channel_names
        )
        data_st["signals"] = interpolated_signals
        data_st["interp_mask"] = interp_mask  # 新增字段

    logging.info(f"Loaded signals for {data_fpath}")
    return data_st


def interpolate_signals(
    signals, sample_indices, channel_names, method="linear+nearest"
):
    logging.info(f"Interpolating signals with method: {method}")

    if method != "linear+nearest":
        raise ValueError("Only 'linear+nearest' interpolation method is implemented.")

    if len(signals) == 0:
        return signals

    full_range = np.arange(1, sample_indices[-1] + 1)
    interpolated_signals = np.zeros((len(full_range), signals.shape[1]))

    interp_mask    = np.zeros((len(full_range), signals.shape[1]), dtype=np.int16)

    # ??? 为什么要这样做
    for ch_i, ch_name in enumerate(channel_names):
        if ch_name.startswith("CH"):
            interp_func = RegularGridInterpolator(
                (sample_indices,), signals[:, ch_i], method="linear"
            )
            interpolated_signals[:, ch_i] = interp_func(full_range)
        else:
            interp_func = RegularGridInterpolator(
                (sample_indices,), signals[:, ch_i], method="nearest"
            )
            interpolated_signals[:, ch_i] = interp_func(full_range)
        
        # 对 xq 中的每个点进行插值
        # 如果某个点本来就在 sample_index 里，我们就用原值不变
        # 其余点用插值
        for i, xq in enumerate(full_range):
            if xq in sample_index:
                # 说明是原始采样点
                original_idx = np.where(sample_index == xq)[0][0]
                # interp_mask默认0
            else:
                # 用插值
                val = interpolator([xq])[0]
                interp_mask[i, ch_i] = 1  # 标记此点为插值点

    logging.info(f"Finished interpolating signals")
    return interpolated_signals, interp_mask


def create_edf_file(data_st, edf_file_path, start_datetime, patient_name, patient_code):
    logging.info(f"Creating EDF file at {edf_file_path} for patient {patient_name}")

    if not os.path.isdir(os.path.dirname(edf_file_path)):
        os.makedirs(os.path.dirname(edf_file_path))

    signals = data_st["signals"]
    channel_names = data_st["channel_names"]
    sample_frequency = data_st["sampling_rate"]

    # 若有插值mask, 追加到 signals
    has_interp = "interp_mask" in data_st
    if has_interp:
        interp_mask = data_st["interp_mask"]
        if interp_mask.shape != signals.shape:
            logging.error("interp_mask shape mismatch! skip adding mask.")
            has_interp = False
        else:
            # 把interp_mask当作最后一个通道
            signals = np.hstack([signals, interp_mask])
            channel_names.append("CH_InterpMask")

    n_channels = len(channel_names)

    edf_writer = pyedflib.EdfWriter(
        edf_file_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
    )

    edf_writer.setPatientName(patient_name)
    edf_writer.setPatientCode(patient_code)
    edf_writer.setSex(0)  # Boss, Carolともにfemale
    edf_writer.setStartdatetime(start_datetime)

    channel_info = []
    for i, channel_name in enumerate(channel_names):
        physical_min = np.nanmin(signals[:, i])
        physical_max = np.nanmax(signals[:, i])
        if np.isnan(physical_min) or np.isnan(physical_max) or (physical_min == physical_max):
            physical_min, physical_max = -1000, 1000  # デフォルト値に設定

         # 对 "CH_InterpMask" 做特殊处理
        if ch_name == "CH_InterpMask":
            dimension = "binary"
        elif ch_name.startswith("CH"):
            dimension = "uV"
        else:
            dimension = "trigger" # ??? need furether check

        ch_dict = {
            "label": channel_name,
            "dimension": "uV" if "CH" in channel_name else "trigger",
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
    logging.info(f"EDF file {edf_file_path} created successfully.")


def extract_datetime_from_filename(filename):
    logging.info(f"Extracting datetime from filename: {filename}")
    base_filename = os.path.basename(filename)
    datetime_str = base_filename.split("T")[0] + base_filename.split("T")[1][:6]
    start_datetime = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    return start_datetime


def convert_all_bin_to_edf(original_data_dir, output_parent_dir):
    logging.info(f"Converting all .bin files in {original_data_dir} to EDF format in {output_parent_dir}")
    for root, dirs, files in os.walk(original_data_dir):
        for file in files:
            if file.endswith(".bin"):
                bin_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, original_data_dir)
                output_dir = os.path.join(output_parent_dir, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                edf_filename = os.path.splitext(file)[0] + ".edf"
                edf_file_path = os.path.join(output_dir, edf_filename)

                # PatientNameとPatientCodeの設定
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
                    logging.error(f"Error processing {bin_file_path}: {e}")

# BIDSのieeg.jsonファイル用のJSONオブジェクトを生成する関数
def create_ieeg_json_file(
    bin_file_path: str,
    out_json_path: str,
    task_name: str = "",
    task_description: str = "",
    instructions: str = "n/a",
    sampling_frequency: int = 1000,
    power_line_frequency: int = 60,
    hardware_filters: str = "n/a",
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
    ieeg_placement_scheme: str = "n/a",
    ieeg_reference: str = "Physical reference contacts located on the skull-facing side of the left/right array, CH11 and CH27 physically",
    electrode_manufacturer: str = "CorTec GmbH, Freiburg, Germany",
    electrode_manufacturers_model_name: str = "Brain Interchange ONE",
    ieeg_ground: str = "the upper back (interscapular region)",
    electrical_stimulation: bool = False,
    electrical_stimulation_parameters: str = "n/a"
):
    """
    bin_file_path       : 変換元のbinファイルのパス
    out_json_path       : 出力先のJSONファイルパス
    task_name           : BIDS記述用のタスク名
    task_description    : BIDS記述用のタスクの説明
    instructions        : 実験参加者(サル)への指示内容等があれば
    sampling_frequency  : サンプリング周波数 (Hz)
    power_line_frequency: 商用電源周波数
    hardware_filters    : ハードウェアフィルタの情報
    software_filters    : ソフトウェアフィルタの情報
    manufacturer        : デバイスメーカー
    manufacturers_model_name : デバイスモデル名
    institution_name    : 測定を実施した機関
    institution_address : 測定を実施した施設の住所
    ecog_channel_count  : ECoGチャネル数
    seeg_channel_count  : sEEGチャネル数
    eeg_channel_count   : EEGチャネル数
    eog_channel_count   : EOGチャネル数
    ecg_channel_count   : ECGチャネル数
    emg_channel_count   : EMGチャネル数
    misc_channel_count  : その他（バイポーラEMGなどの）チャネル数
    trigger_channel_count: トリガチャネル数
    recording_type      : continuous or epoched
    software_versions   : 使用ソフトウェアのバージョン
    ieeg_placement_scheme : iEEG電極の配置情報
    ieeg_reference      : リファレンス電極に関する情報
    electrode_manufacturer: 電極メーカー
    electrode_manufacturers_model_name: 電極モデル名
    ieeg_ground         : グラウンド電極の情報
    electrical_stimulation: 電気刺激を実施したか否か
    electrical_stimulation_parameters: 電気刺激のパラメータ
    """

    # binファイルを読み込み，データとメタ情報を取得
    try:
        data_st = load_signals_cortec(bin_file_path, interp_method="linear+nearest")
    except Exception as e:
        logging.error(f"Failed to load bin data from {bin_file_path}: {e}")
        return

    # data_st からデータの記録時間を計算
    signals = data_st["signals"]  # shape: (n_channels, n_samples)
    n_samples = signals.shape[1]  # サンプル数
    # サンプリング周波数とサンプル数から記録時間(秒)を計算
    recording_duration_sec = n_samples / float(sampling_frequency)

    # BIDS iEEG JSONの内容を辞書にまとめる
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
        "SessionLabelDescription": "session labeled by post-operative day"
    }
    ieeg_json_dict["Amplification"] = data_st["amplification"]  # e.g. "57.5dB"

    # 出力フォルダが存在しない場合は作成
    out_dir = os.path.dirname(out_json_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # JSONファイルとして書き出し
    try:
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(ieeg_json_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"iEEG JSON file created: {out_json_path}")
    except Exception as e:
        logging.error(f"Failed to write iEEG JSON file to {out_json_path}: {e}")


def get_default_channel_data():
    """
    デフォルトのチャネル情報を辞書で返す
    key   : チャンネル名 (例: "CH01", "TR01" など)
    value : {
        "name": <チャンネル名>,
        "type": <種類>,          # ECOG, DC など
        "units": <単位>,         # uV など
        "low_cutoff": <ハイパス>, # n/a, 2Hz など
        "high_cutoff": <ローパス>,
        "sampling_frequency": <サンプリング周波数>,
        "group": <グループ識別>,
        "status": <状態>,        # good, bad など
    }
    """

    # ECOG チャンネル（CH01～CH32）
    default_channels = {}
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

    # EXT01（1チャンネルのみ）
    default_channels["EXT01"] = {
        "name": "EXT01",
        "type": "DC",
        "units": "n/a",
        "low_cutoff": "n/a",
        "high_cutoff": "n/a",
        "sampling_frequency": 1000,
        "group": "n/a",
        "status": "good",
    }

    # TRチャンネル（TR01～TR16）
    for i in range(1, 17):
        tr_name = f"TR{i:02d}"
        default_channels[tr_name] = {
            "name": tr_name,
            "type": "DC",
            "units": "n/a",
            "low_cutoff": "n/a",
            "high_cutoff": "n/a",
            "sampling_frequency": 1000,
            "group": "n/a",
            "status": "good",
        }

    return default_channels

def create_channels_tsv_file(
    out_tsv_path: str,
    updated_channels: dict = None,
    reference_channel=None
):
    """
    デフォルトのチャネル情報をもとに、updated_channels で指定された部分を更新し、
    TSV形式で保存する。

    Parameters
    ----------
    out_tsv_path : str
        生成する TSV ファイルのフルパス
    updated_channels : dict, optional
        更新したいチャネル名: {項目: 値} の辞書の辞書 例:
        {
            "TR15": {
                "type": "DC2",
                "sampling_frequency": 10000,
            },
            "CH01": {
                "status": "bad"
            },
            ...
        }
        のような形式
    """

    # 1) デフォルトのチャネル情報を取得
    channel_data = get_default_channel_data()

     # 如果 reference_channel 存在, 我们把它标记一下
    if reference_channel in channel_data:
        channel_data[reference_channel]["status_description"] = "Used as reference"
    
    # 如果 data_st 中 channel_names 有 CH_InterpMask，代表有插值通道
    channel_names = data_st["channel_names"][:]
    has_interp = "interp_mask" in data_st
    if has_interp and "CH_InterpMask" not in channel_names:
        channel_names.append("CH_InterpMask")

    # 注意！以下数据结构的信息不完整，需要根据实际情况补充
    # 构建一个临时数据结构, key=通道名, val=字段
    channel_dict = {}
    for ch_name in channel_names:
        if ch_name == "CH_InterpMask":
            ch_type = "MISC"
            ch_units = "binary"
            ch_status = "good"
            ch_desc = "Interpolation mask channel (1=interpolated sample, 0=original)"
        elif ch_name.startswith("TR"):
            ch_type = "TRIGGER"
            ch_units = "n/a"
            ch_status = "good"
            ch_desc = ""
        else:
            ch_type = "ECOG"
            ch_units = "uV"
            ch_status = "good"
            ch_desc = ""

        channel_dict[ch_name] = {
            "name": ch_name,
            "type": ch_type,
            "units": ch_units,
            "low_cutoff": "2Hz",   # 例如
            "high_cutoff": "325Hz",# 例如
            "sampling_frequency": sampling_rate,
            "group": "n/a",
            "status": ch_status,
            "status_description": ch_desc
        }

    # 2) 更新指示があれば反映する
    if updated_channels is not None:
        for ch_name, new_info in updated_channels.items():
            if ch_name in channel_data:
                # 既存のチャネルデータに対して上書き
                channel_data[ch_name].update(new_info)
                logging.info(f"Updated channel {ch_name} with {new_info}")
            else:
                # デフォルトに存在しないチャンネル名の場合の挙動
                logging.warning(
                    f"Channel '{ch_name}' not in default config. Skip or add logic as needed."
                )

    # 3) TSVディレクトリが無ければ作成
    out_dir = os.path.dirname(out_tsv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 4) TSVファイルを書き出し
    #   BIDSのchannels.tsv仕様にあるカラム名:
    #   name  type  units  low_cutoff  high_cutoff  sampling_frequency  group  status
    with open(out_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # ヘッダー行を書き込み
        writer.writerow([
            "name", "type", "units", "low_cutoff", "high_cutoff",
            "sampling_frequency", "group", "status"
        ])

        # channel_dataのkey（例: CH01, TR15...）順にソートして書き込む
        for ch_name in sorted(channel_data.keys()):
            ch_info = channel_data[ch_name]
            row = [
                ch_info["name"],
                ch_info["type"],
                ch_info["units"],
                ch_info["low_cutoff"],
                ch_info["high_cutoff"],
                ch_info["sampling_frequency"],
                ch_info["group"],
                ch_info["status"]
            ]
            writer.writerow(row)

    logging.info(f"Channels TSV file created: {out_tsv_path}")


###################################################
### NEW PART 3: 生成 impedance TSV 的示例函数   ###
###################################################
def create_impedance_tsv(sub_id, ses_day, date_str, impedance_source_dir, bids_output_dir, logger):
    """
    假设每个测量日可能在 impedance_source_dir 下有多个阻抗文件(一日多测),
    文件名里含有时间戳, 例如:  20240501_1305_imp.csv
    这里做的是示例, 需根据你的实际数据格式实现
    """
    # 假设阻抗文件存放在 /.../Impedance/<MonkeyName>/<YYYYMMDD>/*.csv
    # 也可能是 .txt / .tsv... 你需要调整搜索方式
    day_folder = os.path.join(impedance_source_dir, date_str)
    if not os.path.isdir(day_folder):
        logger.info(f"No impedance folder found for date={date_str}, skip.")
        return  # 无此天的impedance

    # 收集当日所有阻抗文件
    impedance_files = [
        f for f in os.listdir(day_folder)
        if os.path.isfile(os.path.join(day_folder, f)) and f.endswith(".csv")
    ]
    if not impedance_files:
        logger.info(f"No impedance file found on {date_str}, skip.")
        return

    # 在BIDS内创建 ses-dayXX 目录
    ses_dir = os.path.join(bids_output_dir, f"sub-{sub_id}", f"ses-day{ses_day:02}")
    if not os.path.isdir(ses_dir):
        os.makedirs(ses_dir)

    for imp_file in impedance_files:
        # 解析文件名中的时间戳，如 20240501_1305_imp.csv
        # 具体格式需要你自己根据实际情况写解析逻辑
        # 这里只是演示把 "1305" 视为 HHMM
        base_name = os.path.splitext(imp_file)[0]
        parts = base_name.split("_")
        if len(parts) < 2:
            measure_time_str = "0000"  # 未知时刻
        else:
            measure_time_str = parts[1]  # "1305"

        # 生成 BIDS 内最终文件名
        # e.g. sub-monkeyc_ses-day03_20240501-1305_impedance.tsv
        out_imp_name = f"sub-{sub_id}_ses-day{ses_day:02}_{date_str}-{measure_time_str}_impedance.tsv"
        out_imp_path = os.path.join(ses_dir, out_imp_name)

        # 读取imp文件并写入tsv
        # 这里仅示例：假设csv里有两列: channel_name, impedance(uA)
        with open(os.path.join(day_folder, imp_file), "r", encoding="utf-8") as fin, \
             open(out_imp_path, "w", encoding="utf-8", newline="") as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout, delimiter="\t")

            # 写表头
            writer.writerow(["channel_name", "impedance_uA", "measurement_time"])
            for row in reader:
                # row: [channel_name, impedance_value]
                if len(row) < 2:
                    continue
                ch_name = row[0]
                imp_val = row[1]
                writer.writerow([ch_name, imp_val, measure_time_str])

        logger.info(f"Created impedance TSV: {out_imp_path}")

##############################################################
### NEW PART 4: 对 "01010101" 进行检测并标记bad的辅助函数  ###
##############################################################
def detect_01010101_pattern(trigger_data, logger, threshold_count=50):
    """
    简易判定：如果在 trigger_data 中出现很多类似"01010101"的重复脉冲，返回 True
    你需要根据实际采样率/信号格式定义识别逻辑
    threshold_count: 若符合“01010101”规律的采样数超过这个阈值，就判定bad
    """
    # 假设 trigger_data 为 numpy array, 里面是0或1 (或浮点)
    # 我们简单统计有多少段 连续的 [0,1,0,1,0,1,0,1]...
    # 这里只做一个演示: 判断当这个数据里1和0交替的次数是否>threshold
    # 你可以根据实际需求, 做更细化的检测

    count_pattern = 0
    for i in range(len(trigger_data)-7):
        segment = trigger_data[i:i+8]
        # 检测 segment 是否等于 [0,1,0,1,0,1,0,1] 或 [1,0,1,0,1,0,1,0]
        if all(segment[j] != segment[j+1] for j in range(7)):  # 简单交替检测
            count_pattern += 1

    logger.debug(f"detect_01010101_pattern: count={count_pattern}")
    return (count_pattern >= threshold_count)