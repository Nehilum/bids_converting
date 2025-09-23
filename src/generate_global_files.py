#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os

from datetime import datetime
from pathlib import Path
import bids_config as config

def generate_readme(bids_root: str):
    """
    Generate top-level README.
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
    Generate top-level CHANGES (initialize with one record).
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
    Create /stimuli directory and placeholder note (does not affect validation).
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
    Generate .bidsignore (ignore auxiliary directories/files not related to the specification).
    You can append doc/ or temporary output directories as needed.
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
    Generate dataset_description.json in the BIDS root directory.
    """
    out_path = os.path.join(bids_root, "dataset_description.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config.dataset_description, f, indent=4)
    print(f"[INFO] Generated: {out_path}")

def generate_participants_tsv(bids_root):
    """
    Generate participants.tsv in the BIDS root directory.
    If needed, dynamically generate based on actual experimental information, e.g., read from a database or config file.
    """
    # Example data: Assume there are two monkeys
    out_path = os.path.join(bids_root, "participants.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in config.participants_data:
            writer.writerow(row)
    print(f"[INFO] Generated: {out_path}")

def generate_participants_json(bids_root):
    """
    Generate participants.json in the BIDS root directory, explaining the meaning of each column in participants.tsv.
    """
    out_path = os.path.join(bids_root, "participants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config.participants_json_content, f, indent=4)
    print(f"[INFO] Generated: {out_path}")

def main():
    bids_root = config.DEFAULT_BIDS_ROOT
    os.makedirs(bids_root, exist_ok=True)

    generate_dataset_description(bids_root)
    generate_participants_tsv(bids_root)
    generate_participants_json(bids_root)

    # Newly added global files/directories
    generate_readme(bids_root)
    generate_changes(bids_root)
    ensure_stimuli_dir(bids_root)
    generate_bidsignore(bids_root)
    print("[INFO] All files generated successfully.")

if __name__ == "__main__":
    main()
