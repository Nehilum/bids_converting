# -*- coding: utf-8 -*-

from datetime import datetime
import chardet
import logging
import shutil
import json
import csv
import os

# オリジナルモジュールの取得
from utils import load_signals_cortec, create_edf_file, extract_date_from_filename, create_ieeg_json_file, create_channels_tsv_file

# # ログの設定
# logging.basicConfig(
#     level=logging.DEBUG,  # ログレベルを設定（DEBUG, INFO, WARNING, ERROR, CRITICAL）
#     format='%(asctime)s - %(levelname)s - %(message)s',  # ログフォーマットの設定
#     filename='process.log',  # ログファイル名の指定
#     filemode='w'  # 書き込みモード（w: 書き込み/上書き, a: 追記）
# )

# コンソールにもログを出力するための設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

#######################################################
### NEW PART 1: 设置多日志输出(一个正常+一个警告/错误) ###
#######################################################
# def setup_logging(info_log='process.log', error_log='error.log'):
#     logger = logging.getLogger("bids_logger")
#     logger.setLevel(logging.DEBUG)

#     # 清空已存在的handler，防止重复写入
#     logger.handlers = []

#     # 格式
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#     # info级别(含debug,info)输出到 info_log
#     fh_info = logging.FileHandler(info_log, mode='w', encoding='utf-8')
#     fh_info.setLevel(logging.INFO)
#     fh_info.setFormatter(formatter)

#     # warning及以上输出到 error_log
#     fh_error = logging.FileHandler(error_log, mode='w', encoding='utf-8')
#     fh_error.setLevel(logging.WARNING)
#     fh_error.setFormatter(formatter)

#     # 同时在控制台输出
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)

#     logger.addHandler(fh_info)
#     logger.addHandler(fh_error)
#     logger.addHandler(ch)

#     return logger


'''
処理フロー
    1. ConditionDataディレクトリ内のサルディレクトリ名を取得する（Boss or Carol - subject名）
    2. サルディレクトリ内の日付フォルダ名を（古い順に）取得する
    3. ある日付フォルダ内のjsonファイル一覧を（古い順に）取得する
    4. あるjsonファイル内の Task Type を取得する
    5. あるjsonファイルのパスとその Task Type から edfファイル名を作成する
    6. あるedfファイルに該当する binファイル を edfファイル に変換する
    7. 3.に属するすべてのjsonファイルに対して、4・5・6を繰り返す
    8. 2.に属するすべての日付フォルダに対して、3・4・5・6・7を繰り返す
    9. 1.に属するすべてのサルディレクトリに対して、2・3・4・5・6・7を繰り返す
'''

# サル名とサルIDの対応辞書
map_sub_name_to_id = {
    "Carol": "monkeyc",
    "Boss": "monkeyb"
}

# サル名リスト
monkey_names = ["Boss", "Carol"]

# 猿の手術日
surgery_date = {
    "Boss": "20230711",
    "Carol": "20221222"
}

# タスクのマッピングデータ
# 変換するタスクの指定は、"use_flag" を True に設定することで行う
task_mapping_info = {
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
    "reaching0": {
        "use_flag": True,
        "mapped_name": "reaching0",
        "description": "Reaching without Home button",
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

# 処理する日付の範囲を指定
# TODO: 日付の範囲はサルごとに決める
limit_start_date = "20230101"
limit_end_date = "20250101"

# データディレクトリのパス
data_dir_path = r"/work/project/ECoG_Monkey/01_Data"
# debug path
# data_dir_path = r"C:\Users\ilass\osakauniv_ws\monkey-data-bids-converter\sample_data_for_BIDS\CiNet_NAS\01_Data"

impedance_dir_path = os.path.join(data_dir_path, "Impedance")

# BIDSデータディレクトリのパス
bids_data_dir_path = r"/work/project/ECoG_Monkey/BIDS"
# debug path
# bids_data_dir_path = r"C:\Users\ilass\osakauniv_ws\monkey-data-bids-converter\20241226-BIDS"

# load config file
config_file_path = r"/work/project/ECoG_Monkey/01_Data/config.json"
with open(config_file_path, "r", encoding="utf-8") as f:
        config = json.load(f)

# BIDS変換の処理をする日付の範囲を指定
range_start_date = {
    "Carol": datetime.strptime("20221222", "%Y%m%d"),
    "Boss": datetime.strptime("20240501", "%Y%m%d")
}
range_end_date = {    
    "Carol": datetime.strptime("20240501", "%Y%m%d"),
    "Boss": datetime.strptime("20240501", "%Y%m%d")
}

# BIDSデータ保存用ディレクトリが存在しない場合に作成
if not os.path.isdir(bids_data_dir_path):
    os.makedirs(bids_data_dir_path)

for monkey_name in monkey_names:
    # 初始化日志
    logger = setup_logging("process.log", "error.log")
    logging.info(f"Processing monkey: {monkey_name}")
    sub_id = map_sub_name_to_id[monkey_name]
    subject_cfg = config["subjects"][monkey_name]

    # チャンネル情報の取得
    broken_ch_list = subject_cfg.get("BrokenChannels", [])
    question_ch_list = subject_cfg.get("QuestionableChannels", [])
    remap_dict = subject_cfg.get("RemappedChannels", {})

    # 从 config.json 中解析字符串
    start_date_str = subject_cfg["start_date"]     # e.g. "20221222"
    end_date_str   = subject_cfg["end_date"]       # e.g. "20240501"
    op_day_str     = subject_cfg["OperationDay"]   # e.g. "20221222"

    # 转为 datetime
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date   = datetime.strptime(end_date_str,   "%Y%m%d")
    op_date    = datetime.strptime(op_day_str,     "%Y%m%d")

    # サルフォルダ下のConditionフォルダ内の全ての日付フォルダ名を取得
    condition_folder_path = os.path.join(data_dir_path, "Condition", monkey_name)
    dates = [name for name in os.listdir(condition_folder_path) if os.path.isdir(os.path.join(condition_folder_path, name))]
    logging.debug(f"Dates found: {dates}")

    # 日付フォルダ名を datetime オブジェクトに変換し、日付の古い順にソート
    try:
        dates_sorted = sorted(dates, key=lambda date: datetime.strptime(date, "%Y%m%d"), reverse=False)
        logging.debug(f"Dates found (sorted): {dates_sorted}")
    except ValueError as e:
        logging.error(f"Error parsing date folders: {e}")
        # 日付フォルダが無い/形式が合わないなどの場合はスキップ
        continue

    # 開始日付と終了日付を指定し、その範囲内の日付フォルダのみ処理する
    dates_sorted = [date for date in dates_sorted if start_date <= datetime.strptime(date, "%Y%m%d") <= end_date]
    logging.debug(f"Dates sorted found: {dates_sorted}")

    # 日付フォルダごとの処理
    for index, date in enumerate(dates_sorted, start=1):
        logging.info(f"Processing date: {date}")
        # ses_id = "{:02}".format(index)
        # 计算术后天数
        post_op_day = (datetime.strptime(date, "%Y%m%d") - op_date).days
        ses_id = f"day{post_op_day:02d}"
        # ある日付フォルダ内の全てのjsonファイル名を取得
        json_folder_path = os.path.join(condition_folder_path, date)
        json_file_names = [
            name for name in os.listdir(json_folder_path)
            if os.path.isfile(os.path.join(json_folder_path, name)) and name.endswith(".json")
        ]

        # 「Task Type」ごとにファイルを分類
        task_type_infos = {}
        # scans.tsv に書き込むためのリストを初期化
        scans_info = []

        for json_file_name in json_file_names:
            logging.info(f"Processing JSON file: {json_file_name}")
            json_file_path = os.path.join(json_folder_path, json_file_name)
            # JSONファイルのエンコーディングを自動検出して読み込む
            with open(json_file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']

            with open(json_file_path, 'r', encoding=encoding) as f:
                try:
                    condition_data = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}")
                    continue

            # jsonファイルから「Task Type」を取得
            task_type = condition_data.get('Task Type')
            # task_mapping_info で use_flag が True のもののみ処理する
            if task_type in task_mapping_info and task_mapping_info[task_type]["use_flag"]:
                task_type_infos.setdefault(task_type, [])
                # 拡張子を除いたファイル名
                task_type_infos[task_type].append(os.path.splitext(json_file_name)[0])

        # 「Task Type」ごとにrun番号を割りふる
        for task_type, file_name_list in task_type_infos.items():
            # 各「Task Type」ごとに、jsonファイルを日付の古い順に並べ替え
            file_name_list.sort(key=extract_date_from_filename, reverse=False)

            run_num = 0
            mapped_task_name = task_mapping_info[task_type]["mapped_name"]
            task_description = task_mapping_info[task_type]["description"]

            for file_name in file_name_list:
                run_num += 1

                # edfファイル名とパスの設定
                edf_file_name = (
                    f"sub-{sub_id}_ses-{post_operative_day}_task-{mapped_task_name}_run-{run_num:02}_ieeg.edf"
                )
                edf_file_path = os.path.join(
                    bids_data_dir_path, f"sub-{sub_id}", f"ses-{post_operative_day}", "ieeg", edf_file_name
                )

                # 元となるbinファイルパスの生成
                bin_file_name = file_name + ".bin"
                bin_file_path = os.path.join(
                    data_dir_path, "CortecData", monkey_name, date, bin_file_name
                )

                # BIDS形式のファイル一式を生成
                try:
                    data_st = load_signals_cortec(bin_file_path, interp_method="linear+nearest")

                    # ファイル名から実験開始時刻の取得
                    exp_start_datetime = extract_date_from_filename(filename=file_name)

                    # edfファイルの生成と保存
                    # --------------------------------------------------------------------------------------
                    create_edf_file(
                        data_st, edf_file_path, exp_start_datetime, monkey_name, "sub-" + str(sub_id)
                    )
                    # --------------------------------------------------------------------------------------

                    # eventsファイルのコピー
                    # --------------------------------------------------------------------------------------
                    events_file_name = file_name + "_events.tsv"
                    events_file_path = os.path.join(
                        data_dir_path, "Events", monkey_name, date, events_file_name
                    )
                    bids_events_file_name = (
                        f"sub-{sub_id}_ses-day{post_operative_day}_task-{mapped_task_name}"
                        f"_run-{run_num:02}_events.tsv"
                    )
                    bids_events_file_path = os.path.join(
                        bids_data_dir_path, f"sub-{sub_id}", f"ses-{post_operative_day}", "ieeg", bids_events_file_name
                    )
                    shutil.copy(events_file_path, bids_events_file_path)
                    # --------------------------------------------------------------------------------------

                    # channels ファイルの生成
                    # --------------------------------------------------------------------------------------
                    updates = {}
                    bids_channels_file_name = (
                        f"sub-{sub_id}_ses-{post_operative_day}_task-{mapped_task_name}"
                        f"_run-{run_num:02}_channels.tsv"
                    )
                    bids_channels_file_path = os.path.join(
                        bids_data_dir_path, f"sub-{sub_id}", f"ses-{post_operative_day}", "ieeg", bids_channels_file_name
                    )

                    # 标记坏通道
                    for ch in broken_ch_list:
                        updates[ch] = {
                            "status": "bad",
                            "status_description": "hardware permanently defective from day 0"
                        }
                        
                    create_channels_tsv_file(
                        out_tsv_path=bids_channels_file_path,
                        updated_channels=updates
                    )
                    # --------------------------------------------------------------------------------------

                    # jsonファイルの生成
                    # --------------------------------------------------------------------------------------
                    bids_json_file_name = (
                        f"sub-{sub_id}_ses-{post_operative_day}_task-{mapped_task_name}"
                        f"_run-{run_num:02}_ieeg.json"
                    )
                    bids_json_file_path = os.path.join(
                        bids_data_dir_path, f"sub-{sub_id}", f"ses-{post_operative_day}", "ieeg", bids_json_file_name
                    )

                    # 若有插值mask, misc_count = 1, 否则 misc_count = 0
                    misc_count = 0
                    has_interp = "interp_mask" in data_st
                    if has_interp:
                        misc_count = 1
                    create_ieeg_json_file(
                        bin_file_path=bin_file_path,
                        out_json_path=bids_json_file_path,
                        task_name=mapped_task_name,
                        task_description=task_description,
                        instructions="n/a",
                        sampling_frequency=int(data_st["sampling_rate"]),
                        power_line_frequency=60,
                        hardware_filters="High pass filter cut-off: ~2 Hz, Low pass filter cut-off: 325 Hz",
                        software_filters="n/a",
                        manufacturer="CorTec GmbH, Freiburg, Germany",
                        manufacturers_model_name="Brain Interchange ONE",
                        institution_name="Osaka University Graduate School of Medicine",
                        institution_address="2-2 Yamadaoka, Suita-shi, Osaka 565-0871",
                        ecog_channel_count=data_st["channel_num_signal"],
                        seeg_channel_count=0,
                        eeg_channel_count=0,
                        eog_channel_count=0,
                        ecg_channel_count=0,
                        emg_channel_count=0,
                        misc_channel_count=misc_count,
                        trigger_channel_count=data_st["channel_num_trigger"],
                        recording_type="continuous",
                        software_versions="n/a",
                        ieeg_placement_scheme=(
                            "The electrode arrays were positioned into subdural space over the "
                            "sensorimotor cortex on both hemispheres. Each of these arrays had 15 "
                            "measurement electrodes, along with a single reference electrode oriented "
                            "towards the dura mater"
                        ),
                        # .hdrファイルを参照し、gndがFalseの場合は""、Trueの場合は"the upper back (interscapular region)"を設定
                        ieeg_reference="As explained in ieeg placement scheme",
                        electrode_manufacturer="CorTec GmbH, Freiburg, Germany",
                        electrode_manufacturers_model_name="Brain Interchange ONE",
                        # .hdrファイルを参照し、gndがFalseの場合は""、Trueの場合は"the upper back (interscapular region)"を設定
                        ieeg_ground="the upper back (interscapular region)",
                        electrical_stimulation=False,
                        electrical_stimulation_parameters="n/a"
                    )
                    # --------------------------------------------------------------------------------------

                    # ---- ここで scans.tsv 用情報をリストに追加する ----------------------------------
                    # BIDS仕様上、filename はサブフォルダを含む相対パス、post_operative_day は 手術日からの経過日数を記載
                    relative_edf_path = os.path.join(
                        "ieeg", edf_file_name
                    ).replace("\\", "/")  # Windows環境対策
                    # 日付フォルダから年月日を除いた時分秒を取得
                    acquisition_time_str = exp_start_datetime.strftime("%H:%M:%S")
                    scans_info.append([relative_edf_path, acquisition_time_str])
                    # ---------------------------------------------------------------------------

                except Exception as e:
                    logging.error(f"Error processing {bin_file_path}: {e}")
                    continue

        # セッションごとの処理が終わったら scans.tsv を生成
        if scans_info:
            # ディレクトリが存在しない場合は作成
            ses_dir_path = os.path.join(bids_data_dir_path, f"sub-{sub_id}", f"ses-{post_operative_day}")
            if not os.path.isdir(ses_dir_path):
                os.makedirs(ses_dir_path)

            scans_tsv_name = f"sub-{sub_id}_ses-{post_operative_day}_scans.tsv"
            scans_tsv_path = os.path.join(ses_dir_path, scans_tsv_name)

            # タブ区切りで書き出し
            with open(scans_tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                # ヘッダ行
                writer.writerow(["filename", "time"])
                # データ行
                for row in scans_info:
                    writer.writerow(row)

            logging.info(f"Generated scans.tsv: {scans_tsv_path}")
        # 日付フォルダ(=セッション)のループここまで

    # サルディレクトリのループここまで
