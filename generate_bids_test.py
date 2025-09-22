# -*- coding: utf-8 -*-
"""
最適化された generate_bids.py
機能：
　実験データ（例：猿 "Boss" や "Carol" のデータ）を読み込み、BIDS規格に準拠したデータに変換し、
　対応する EDF、JSON、TSV ファイルなどを生成する。
"""

import logging
from datetime import datetime
import chardet
import shutil
import json
import csv
from pathlib import Path
from typing import Dict, Optional

# カスタムモジュールから必要な関数をインポート
from utils_test import (
    setup_logging,
    extract_date_from_filename,
    load_signals_cortec,
    create_edf_file,
    create_ieeg_json_file,
    create_channels_tsv_file,
    get_post_op_day,
    detect_01010101_pattern,
    create_impedance_tsv,
)

# 形如:
# {
#   "CH04": {"QuestionStartDate": "20250601", "BadAfter": "20250915"},
#   "CH19": {"QuestionStartDate": "20251201", "BadAfter": null}
# }
ProgressiveChannelsType = Dict[str, Dict[str, Optional[str]]]

# 被験者（猿）の名前と対応するBIDS IDのマッピング
MAP_SUB_NAME_TO_ID = {
    "Carol": "monkeyc",
    "Boss": "monkeyb"
}
MONKEY_NAMES = ["Boss", "Carol"]

# タスクのマッピング情報（use_flag が True のタスクのみ変換対象とする）
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

# データディレクトリとBIDS出力ディレクトリ（必要に応じてパスを調整してください）
DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/01_Data")
BIDS_DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/BIDS")
CONFIG_FILE_PATH = Path("..") / "config.json"


def main():
    """
    メイン関数：各被験者のデータを順次処理し、BIDS形式に変換する
    """
    # ログ設定：ファイルとコンソールへ同時に出力
    logger = setup_logging(info_log="process.log", error_log="error.log")
    logger.info("BIDSデータ変換プロセスを開始します")

    # 設定ファイルの読み込み
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"設定ファイル {CONFIG_FILE_PATH} の読み込みに失敗しました: {e}")
        return

    # 各被験者のデータを処理
    for monkey_name in MONKEY_NAMES:
        process_subject(monkey_name, config, logger)


def process_subject(monkey_name: str, config: dict, logger: logging.Logger) -> None:
    """
    単一被験者の全データを処理する関数
    引数：
        monkey_name: 被験者の名前（例："Boss" または "Carol"）
        config: 設定ファイルから読み込んだ情報
        logger: ログ記録用オブジェクト
    """
    logger.info(f"被験者 {monkey_name} の処理を開始します")
    sub_id = MAP_SUB_NAME_TO_ID.get(monkey_name, monkey_name.lower())
    subject_cfg = config.get("subjects", {}).get(monkey_name, {})

    # 被験者のチャネル設定および日付情報を取得
    broken_ch_list = subject_cfg.get("BrokenChannels", [])
    # 读取ProgressiveChannels
    # 格式类似: { "CH04":{"QuestionStartDate":"20250601","BadAfter":"20250915"}, ... }
    progressive_channels = subject_cfg.get("ProgressiveChannels", {})
    start_date_str = subject_cfg.get("start_date")   # 例："20221222"
    end_date_str   = subject_cfg.get("end_date")       # 例："20240501"
    op_day_str     = subject_cfg.get("OperationDay")   # 例："20221222"

    try:
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date   = datetime.strptime(end_date_str, "%Y%m%d")
        op_date    = datetime.strptime(op_day_str, "%Y%m%d")
    except Exception as e:
        logger.error(f"被験者 {monkey_name} の日付フォーマットに誤りがあります: {e}")
        return
    
    # 被験者の Condition データディレクトリの取得
    condition_folder_path = DATA_DIR_PATH / "Condition" / monkey_name
    if not condition_folder_path.is_dir():
        logger.error(f"Conditionフォルダが存在しません: {condition_folder_path}")
        return

    # 全日付フォルダを取得し、形式に合ったもののみ抽出
    dates = [d.name for d in condition_folder_path.iterdir() if d.is_dir()]
    logger.debug(f"見つかった日付フォルダ: {dates}")

    try:
        dates_sorted = sorted(dates, key=lambda d: datetime.strptime(d, "%Y%m%d"))
    except Exception as e:
        logger.error(f"日付フォルダの並べ替えに失敗しました: {e}")
        return

    # 指定された開始日～終了日の範囲内のフォルダのみ処理
    dates_filtered = [d for d in dates_sorted if start_date <= datetime.strptime(d, "%Y%m%d") <= end_date]
    logger.debug(f"指定期間内の日付フォルダ: {dates_filtered}")

    # 各日付フォルダを処理
    for date_str in dates_filtered:
        process_date(monkey_name, sub_id, date_str, op_day_str, broken_ch_list, progressive_channels, logger)


def process_date(monkey_name: str, sub_id: str, date_str: str, op_day_str: str, broken_ch_list: list, progressive_channels: ProgressiveChannelsType, logger: logging.Logger) -> None:
    """
    単一の日付フォルダ内の全JSONデータを処理し、対応するBIDSデータファイルを生成する
    引数：
        monkey_name: 被験者名
        sub_id: BIDS用の被験者ID
        date_str: 日付フォルダ名（"YYYYMMDD"形式）
        op_date: 手術日
        broken_ch_list: 不良チャネルのリスト
        logger: ログ記録用オブジェクト
    """
    logger.info(f"日付フォルダ {date_str} の処理を開始します")

    # 手術日からの経過日数を計算し、session ID（例：day03）として使用
    current_date = datetime.strptime(date_str, "%Y%m%d")
    # 使用 utils.get_post_op_day 进行统一计算
    post_op_day = get_post_op_day(date_str, op_day_str)

    ses_id = f"day{post_op_day:02d}"

    # 当該日付フォルダのJSONファイル格納ディレクトリのパス
    json_folder_path = DATA_DIR_PATH / "Condition" / monkey_name / date_str
    if not json_folder_path.is_dir():
        logger.error(f"JSONファイルフォルダが存在しません: {json_folder_path}")
        return

    # 当該フォルダ内の全JSONファイルを取得
    json_files = [f for f in json_folder_path.iterdir() if f.is_file() and f.suffix == ".json"]

    # タスクタイプごとにJSONファイルを分類（キー：タスクタイプ、値：拡張子なしファイル名のリスト）
    task_type_infos = {}
    scans_info = []  # scans.tsv 用の情報を格納するリスト

    for json_file in json_files:
        logger.info(f"JSONファイルを処理中: {json_file.name}")
        try:
            # エンコーディング自動検出後にJSONファイルを読み込む
            raw_data = json_file.read_bytes()
            encoding = chardet.detect(raw_data)["encoding"]
            with json_file.open("r", encoding=encoding) as f:
                condition_data = json.load(f)
        except Exception as e:
            logger.error(f"{json_file} の読み込みまたは解析に失敗しました: {e}")
            continue

        task_type = condition_data.get("Task Type")
        # TASK_MAPPING_INFOにあり、かつuse_flagがTrueの場合のみ処理
        if task_type in TASK_MAPPING_INFO and TASK_MAPPING_INFO[task_type]["use_flag"]:
            task_type_infos.setdefault(task_type, []).append(json_file.stem)

    # 各タスクタイプごとにBIDSファイルを生成
    for task_type, file_name_list in task_type_infos.items():
        # ファイル名から抽出した日付で並べ替え
        file_name_list.sort(key=extract_date_from_filename)

        run_num = 0
        mapped_task_name = TASK_MAPPING_INFO[task_type]["mapped_name"]
        task_description = TASK_MAPPING_INFO[task_type]["description"]

        for base_file_name in file_name_list:
            run_num += 1
            # EDFファイル名と保存パスを構築
            edf_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_ieeg.edf"
            edf_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / edf_file_name

            # 元の .bin ファイルのパスを構築（例：DATA_DIR_PATH/CortecData/<monkey_name>/<date_str>/）
            bin_file_path = DATA_DIR_PATH / "CortecData" / monkey_name / date_str / f"{base_file_name}.bin"

            try:
                # 信号データの読み込みおよび補間処理
                data_st = load_signals_cortec(str(bin_file_path), interp_method="linear+nearest")
                # ファイル名から実験開始時刻を抽出
                exp_start_datetime = extract_date_from_filename(base_file_name)
                # # 数据质量控制：对 trigger 信号进行检测
                # trigger_channel_count = data_st.get("channel_num_trigger", 0)
                # if trigger_channel_count > 0:
                #     # 假设 trigger 信号位于信号矩阵的最后 trigger_channel_count 列
                #     trigger_data = data_st["signals"][:, -trigger_channel_count:]
                #     if detect_01010101_pattern(trigger_data, logger):
                #         logger.warning(f"{bin_file_path} 的 trigger 信号中检测到大量交替模式，可能存在数据异常。")

                # EDFファイルを生成
                create_edf_file(data_st, str(edf_file_path), exp_start_datetime, monkey_name, f"sub-{sub_id}")
            except Exception as e:
                logger.error(f"{bin_file_path} の処理中にエラーが発生しました: {e}")
                continue

            # 対応する events ファイルをコピー
            events_file_name = f"{base_file_name}_events.tsv"
            events_file_path = DATA_DIR_PATH / "Events" / monkey_name / date_str / events_file_name
            bids_events_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_events.tsv"
            bids_events_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_events_file_name
            try:
                shutil.copy(str(events_file_path), str(bids_events_file_path))
            except Exception as e:
                logger.error(f"イベントファイル {events_file_path} のコピーに失敗しました: {e}")

            # channels.tsv ファイルを生成（不良チャネル情報を反映）
            bids_channels_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_channels.tsv"
            bids_channels_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_channels_file_name
            updated_channels = {}
            for ch in broken_ch_list:
                updated_channels[ch] = {
                    "status": "bad",
                    "status_description": "ハードウェアの初期段階から故障している"
                }
            # 2) 再对progressive_channels做判定
            if progressive_channels:
                for ch_name, ch_info in progressive_channels.items():
                    # 解析QuestionStartDate, BadAfter(可能为空)
                    q_start_str = ch_info.get("QuestionStartDate", None)
                    b_after_str = ch_info.get("BadAfter", None)
                    if not q_start_str:
                        # 没有QuestionStartDate, 跳过
                        continue

                    q_start_dt = datetime.strptime(q_start_str, "%Y%m%d")

                    # bad_after 可能是null或""
                    if b_after_str:
                        bad_after_dt = datetime.strptime(b_after_str, "%Y%m%d")
                    else:
                        bad_after_dt = None

                    # 判断status
                    if current_date < q_start_dt:
                        # 该日期还没到questionable的时间点 => good
                        # 但是如果它已经在 broken_channels 里被标记，则保持bad
                        # 此处可判定优先级: 如果已经bad，就不覆盖
                        if ch_name not in updated_channels:
                            updated_channels[ch_name] = {
                                "status": "good",
                                "status_description": "Stable before questionStart"
                            }
                    else:
                        # 已到questionable或更坏的阶段
                        if bad_after_dt and current_date >= bad_after_dt:
                            # bad
                            if ch_name not in updated_channels:
                                updated_channels[ch_name] = {
                                    "status": "bad",
                                    "status_description": f"Turned bad after {b_after_str}"
                                }
                            else:
                                # 如果之前标记了good, 现在要覆盖
                                if updated_channels[ch_name]["status"] == "good":
                                    updated_channels[ch_name]["status"] = "bad"
                                    updated_channels[ch_name]["status_description"] = f"Turned bad after {b_after_str}"
                        else:
                            # questionable
                            # 前提是如果还没有被标记为永久bad
                            if ch_name not in updated_channels or updated_channels[ch_name]["status"] == "good":
                                updated_channels[ch_name] = {
                                    "status": "questionable",
                                    "status_description": f"Impedance rising since {q_start_str}"
                                }
            try:
                create_channels_tsv_file(str(bids_channels_file_path), updated_channels)
            except Exception as e:
                logger.error(f"channels.tsv の生成に失敗しました: {e}")

            # iEEG JSON ファイルを生成
            bids_json_file_name = f"sub-{sub_id}_ses-{ses_id}_task-{mapped_task_name}_run-{run_num:02}_ieeg.json"
            bids_json_file_path = BIDS_DATA_DIR_PATH / f"sub-{sub_id}" / f"ses-{ses_id}" / "ieeg" / bids_json_file_name

            try:
                create_ieeg_json_file(
                    bin_file_path=str(bin_file_path),
                    out_json_path=str(bids_json_file_path),
                    task_name=mapped_task_name,
                    task_description=task_description,
                    misc_channel_count=1,
                    sampling_frequency=int(data_st["sampling_rate"]),
                    ecog_channel_count=data_st["channel_num_signal"]
                )
            except Exception as e:
                logger.error(f"iEEG JSON ファイルの生成に失敗しました: {e}")

            # scans.tsv 用の情報を記録（相対パスと取得時刻）
            relative_edf_path = str(Path("ieeg") / edf_file_name).replace("\\", "/")
            acquisition_time_str = exp_start_datetime.isoformat()
            scans_info.append([relative_edf_path, acquisition_time_str])
            # 如果有阻抗数据，则生成对应的 impedance TSV 文件
            # 参数说明：sub_id、post_op_day、date_str、阻抗源目录（例如 DATA_DIR_PATH/Impedance）、BIDS输出根目录、logger
            create_impedance_tsv(
                sub_id,
                post_op_day,  # 此处使用计算出的术后天数
                date_str,
                str(DATA_DIR_PATH / "Impedance" / monkey_name),
                str(BIDS_DATA_DIR_PATH),
                logger
            )

    # 本セッションでデータが生成されている場合、scans.tsv を作成
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
            logger.info(f"scans.tsv の生成に成功しました: {scans_tsv_path}")
        except Exception as e:
            logger.error(f"scans.tsv の生成に失敗しました: {e}")


if __name__ == "__main__":
    main()
