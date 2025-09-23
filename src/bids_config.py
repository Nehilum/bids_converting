from pathlib import Path
import os
dataset_description = {
    "Name": "Monkey ECoG Dataset",
    "BIDSVersion": "1.7.0",
    "DatasetType": "raw",
    "License": " CC BY 4.0",
    "Authors": ["Huixiang Yang", "Ryohei Fukuma", "Kotaro Okuda", "Takufumi Yanagisawa"],
    "HowToAcknowledge": "Please cite XXX if you use these data",
    "Funding": ["Grant JPMJER1801 from JST, JPMJMS2012 (TY) from Moonshot R&D, JPMJCR18A5 (TY) from CREST, JPMJCR24U2 (TY) from AIP"],
    "ReferencesAndLinks": ["https://example.com/project_page"]
}
participants_data = [
    ["participant_id", "species", "sex", "age"],      # 表头
    ["sub-monkeyb", "Macaca fuscata", "F", "9"],      # Boss
    ["sub-monkeyc", "Macaca fuscata", "F", "8"]       # Carol
]

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
templates = {
    "rest": {
        "onset":        {"Description": "Event onset time relative to run start", "Units": "s"},
        "duration":     {"Description": "Event duration", "Units": "s"},
        "trial_type":   {"Description": "Type of event",
                         "Levels": {"rest_start": "Beginning of eyes-open resting state block (single row event)"}},
        "stim_file":    {"Description": "Associated stimulus file name if any, else 'n/a'"},
        "response_type":{"Description": "Response effector or button label if any, else 'n/a'"},
        "response_time":{"Description": "Latency between stimulus and response when applicable", "Units": "s"},
    },
    "sep": {
        "onset":      {"Description": "Pulse onset", "Units": "s"},
        "duration":   {"Description": "Pulse duration", "Units": "s"},
        "trial_type": {"Description": "Type of event",
                       "Levels": {"ssep_stim": "Somatosensory electrical pulse on"}},
        "stim_side":  {"Description": "Side of stimulation",
                       "Levels": {"Left": "Left wrist stimulation", "Right": "Right wrist stimulation"}},
        "amplitude_mA": {"Description": "Stimulation current amplitude", "Units": "mA"},
        "response_type":{"Description": "Response effector/button if present, else 'n/a'"},
        "response_time":{"Description": "Response latency if present", "Units": "s"},
    },
    "listening": {
        "onset":      {"Description": "Stimulus onset", "Units": "s"},
        "duration":   {"Description": "Stimulus duration", "Units": "s"},
        "trial_type": {"Description": "Type of event",
                       "Levels": {"stimulus_on": "Auditory stimulus onset"}},
        "stim_file":  {"Description": "Presented audio file name (e.g., 'DoMi.wav')"},
        "response_type":{"Description": "Participant response effector if any, else 'n/a'"},
        "response_time":{"Description": "Response latency if present", "Units": "s"},
    },
    "word": {
        "onset":      {"Description": "Stimulus onset", "Units": "s"},
        "duration":   {"Description": "Stimulus duration", "Units": "s"},
        "trial_type": {"Description": "Type of event",
                       "Levels": {"stimulus_on": "Auditory stimulus onset"}},
        "stim_file":  {"Description": "Presented audio file name"},
        "response_type":{"Description": "Participant response effector if any, else 'n/a'"},
        "response_time":{"Description": "Response latency if present", "Units": "s"},
    },
    "pressing": {
        "onset":      {"Description": "Detected movement/press onset", "Units": "s"},
        "duration":   {"Description": "Not applicable for discrete keypress events", "Units": "s"},
        "trial_type": {"Description": "Type of event",
                       "Levels": {"movement_start": "Onset of button press from start-button-referenced parsing",
                                  "movement_done":  "Response button press when there is no start-button reference"}},
        "stim_file":  {"Description": "Task has no external stimulus; set to 'n/a'"},
        "response_type":{"Description": "Which button was pressed",
                         "Levels": {"left_button": "Left-hand (or left-side) button",
                                    "right_button":"Right-hand (or right-side) button"}},
        "response_time":{"Description": "Latency if defined by paradigm; 'n/a' in current script", "Units": "s"},
    },
    "reaching": {
        "onset":      {"Description": "Detected reach movement onset", "Units": "s"},
        "duration":   {"Description": "Not applicable for discrete movement onset events", "Units": "s"},
        "trial_type": {"Description": "Type of event",
                       "Levels": {"movement_start": "Reach movement onset",
                                  "movement_done":  "Reach movement done (= response button) when no start-button"}},
        "stim_file":  {"Description": "No external stimulus; 'n/a'"},
        "response_type":{"Description": "Effector used for the reach",
                         "Levels": {"left_hand": "Left hand reach",
                                    "right_hand":"Right hand reach"}},
        "response_time":{"Description": "Latency if defined by paradigm; 'n/a' in current script", "Units": "s"},
    },
}

# Data and output directories
DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/01_Data")
BIDS_DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/BIDS_test_clean")
CONFIG_FILE_PATH = Path("..") / "config.json"
SAMPLES_FILE_DEFAULT = Path("samples.json")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
DEFAULT_BIDS_ROOT = os.path.join(project_dir, "BIDS_data")
