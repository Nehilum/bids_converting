#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os

from datetime import datetime
from pathlib import Path

def generate_readme(bids_root: str):
    """
    生成顶层 README（最小版）。
    """
    text = [
        "# Monkey ECoG Dataset",
        "",
        "This dataset follows the BIDS specification (iEEG-BIDS).",
        "",
        "## Contents",
        "- Raw iEEG recordings in sub-*/ses-*/ieeg/",
        "- Per-run events.tsv (+ events.json when available)",
        "- Global files: dataset_description.json, participants.tsv/json, CHANGES, README",
        "",
        "## Notes",
        "- Times in events.tsv are in seconds relative to run start.",
        "- Stimuli (if any) should live under /stimuli and be referenced via the `stim_file` column.",
        "",
        "## Contacts",
        "- Maintainer: <your name / email>",
        ""
    ]
    out = os.path.join(bids_root, "README")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"[INFO] Generated: {out}")

def generate_changes(bids_root: str):
    """
    生成顶层 CHANGES（初始化一条记录）。
    """
    today = datetime.today().strftime("%Y-%m-%d")
    text = [
        "Monkey ECoG Dataset — CHANGES",
        "",
        f"{today}: Initial export of raw iEEG dataset in BIDS structure.",
        "- Added dataset_description.json, participants.tsv/json, README.",
        "- Created /stimuli/ directory (placeholder).",
    ]
    out = os.path.join(bids_root, "CHANGES")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"[INFO] Generated: {out}")

def ensure_stimuli_dir(bids_root: str):
    """
    创建 /stimuli 目录与占位说明（不会影响验证）。
    """
    stim_dir = os.path.join(bids_root, "stimuli")
    os.makedirs(stim_dir, exist_ok=True)
    readme_path = os.path.join(stim_dir, "README.txt")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("Place stimulus files here. Reference via events.tsv 'stim_file' column, paths relative to /stimuli.\n")
    print(f"[INFO] Ensured: {stim_dir}")

def generate_bidsignore(bids_root: str):
    """
    生成 .bidsignore（忽略与规范无关的辅助目录/文件）。
    可按需追加 doc/ 或临时输出目录等。
    """
    lines = [
        "doc/",
        "*.log",
        "*.tmp",
        "*.DS_Store",
    ]
    out = os.path.join(bids_root, ".bidsignore")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Generated: {out}")

def generate_dataset_description(bids_root):
    """
    在BIDS根目录生成dataset_description.json
    """
    dataset_description = {
        "Name": "Monkey ECoG Dataset",
        "BIDSVersion": "1.7.0",       # 或者根据实际所用版本
        "DatasetType": "raw",         # raw / derivatives 等
        "License": " CC BY 4.0",             # 或其他如 CC-BY 4.0
        "Authors": ["Huixiang Yang", "Ryohei Fukuma", "Kotaro Okuda", "Takufumi Yanagisawa"],
        "HowToAcknowledge": "Please cite XXX if you use these data",
        "Funding": ["Grant JPMJER1801 from JST, JPMJMS2012 (TY) from Moonshot R&D, JPMJCR18A5 (TY) from CREST, JPMJCR24U2 (TY) from AIP"],
        "ReferencesAndLinks": ["https://example.com/project_page"]
    }

    out_path = os.path.join(bids_root, "dataset_description.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_description, f, indent=4)
    print(f"[INFO] Generated: {out_path}")


def generate_participants_tsv(bids_root):
    """
    在BIDS根目录生成participants.tsv
    如果需要，可根据真实实验信息动态生成，例如从数据库或配置文件读取。
    """
    # 示例数据：假设有两只猴子
    participants_data = [
        ["participant_id", "species", "sex", "age"],      # 表头
        ["sub-monkeyb", "Macaca fuscata", "F", "9"],      # Boss
        ["sub-monkeyc", "Macaca fuscata", "F", "8"]       # Carol
    ]

    out_path = os.path.join(bids_root, "participants.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in participants_data:
            writer.writerow(row)
    print(f"[INFO] Generated: {out_path}")


def generate_participants_json(bids_root):
    """
    在BIDS根目录生成participants.json，用于解释participants.tsv各列的含义。
    """
    # 这里示例地给出了四个字段的说明，可根据需要再增减。
    participants_json_content = {
        "participant_id": {
            "LongName": "Participant (monkey) ID",
            "Description": "Unique participant ID following BIDS conventions."
        },
        "species": {
            "LongName": "Species",
            "Description": "Monkey species."
        },
        "sex": {
            "LongName": "Sex",
            "Description": "Biological sex of the subject (M/F)."
        },
        "age": {
            "LongName": "Age (in years)",
            "Description": "Age of the monkey at the start of experiment."
        }
    }

    out_path = os.path.join(bids_root, "participants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(participants_json_content, f, indent=4)
    print(f"[INFO] Generated: {out_path}")


def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 上上级目录 (举例: 如果脚本在 scripts/ 下面, 那上上级就是项目根目录)
    two_levels_up_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # 然后假设我们想要在 "bids_data" 这个子文件夹下生成BIDS根目录
    # bids_root = /.../two_levels_up_dir/bids_data
    bids_root = os.path.join(two_levels_up_dir, "BIDS_test_v4")

    # 如果目录不存在就创建
    os.makedirs(bids_root, exist_ok=True)

    generate_dataset_description(bids_root)
    generate_participants_tsv(bids_root)
    generate_participants_json(bids_root)

    # 新增的全局文件/目录
    generate_readme(bids_root)
    generate_changes(bids_root)
    ensure_stimuli_dir(bids_root)
    generate_bidsignore(bids_root)
    print("[INFO] All files generated successfully.")


if __name__ == "__main__":
    main()