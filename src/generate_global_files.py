#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json

from datetime import datetime
from pathlib import Path
import bids_config as config

def generate_readme(bids_root: Path):
    """
    Generate top-level README.
    """
    out = bids_root / "README.md"
    out.write_text("\n".join(config.READMETEXT), encoding="utf-8")
    print(f"[INFO] Generated: {out}")

def generate_changes(bids_root: Path):
    """
    Generate top-level CHANGES (initialize with one record).
    """
    today = datetime.today().strftime("%Y-%m-%d")
    text = [
        "Longitudinal Multitask Wireless ECoG Data from Two Fully Implanted Macaca fuscata — CHANGES",
        "",
        f"{today}: Initial export of the wireless subdural ECoG (iEEG) dataset from Macaca mulatta monkeys.",
        "Data curated and exported into BIDS format (iEEG-BIDS specification).",
        "Included: dataset_description.json, participants.tsv, participants.json, README, CHANGES.",
        "Provided subject/session-level iEEG data in EDF with metadata (.json), channels (.tsv), events (.tsv), electrodes.tsv, electrodes.json, coordsystem.json, and scans index (.tsv).",
        "Event files included only for curated and validated runs.",
        "Created /stimuli/ directory.",
    ]
    out = bids_root / "CHANGES.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print(f"[INFO] Generated: {out}")

def ensure_stimuli_dir(bids_root: Path):
    """
    Create /stimuli directory and placeholder note (does not affect validation).
    """
    stim_dir = bids_root / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    readme_path = stim_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Place stimulus files here. Reference via events.tsv 'stim_file' column, paths relative to /stimuli.\n",
            encoding="utf-8",
        )
    print(f"[INFO] Ensured: {stim_dir}")

def generate_dataset_description(bids_root:Path):
    """
    Generate dataset_description.json in the BIDS root directory.
    """
    out_path = bids_root / "dataset_description.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config.dataset_description, f, indent=4)
    print(f"[INFO] Generated: {out_path}")

def generate_participants_tsv(bids_root: Path):
    """
    Generate participants.tsv in the BIDS root directory.
    If needed, dynamically generate based on actual experimental information, e.g., read from a database or config file.
    """
    # Example data: Assume there are two monkeys
    out_path = bids_root / "participants.tsv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in config.participants_data:
            writer.writerow(row)
    print(f"[INFO] Generated: {out_path}")

def generate_participants_json(bids_root: Path):
    """
    Generate participants.json in the BIDS root directory, explaining the meaning of each column in participants.tsv.
    """
    out_path = bids_root / "participants.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config.participants_json_content, f, indent=4)
    print(f"[INFO] Generated: {out_path}")

def main():
    bids_root = Path(config.DEFAULT_BIDS_ROOT)
    bids_root.mkdir(parents=True, exist_ok=True)

    generate_dataset_description(bids_root)
    generate_participants_tsv(bids_root)
    generate_participants_json(bids_root)

    # Newly added global files/directories
    generate_readme(bids_root)
    generate_changes(bids_root)
    ensure_stimuli_dir(bids_root)
    print("[INFO] All files generated successfully.")

if __name__ == "__main__":
    main()
