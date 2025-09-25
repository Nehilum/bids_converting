from pathlib import Path
from typing import Dict, Tuple

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
    ["participant_id", "species", "sex", "age"],      # table header
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
        "mapped_name": "association",
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
        "response_type":{"Description": "Which response button was pressed",
                         "Levels": {"left_button": "Left-side button",
                                    "right_button":"Right-side button"}},
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

coord_description = {
    "iEEGCoordinateSystem": "Other",
    "iEEGCoordinateSystemDescription": (
        "No spatial electrode coordinates are provided at this stage. "
        "Electrodes are listed in electrodes.tsv with non-spatial attributes only."
    ),
    "iEEGCoordinateUnits": "n/a"
}

electrodes_description = {
        "x": {
            "Description": "Electrode x-coordinate in the coordinate system defined in coordsystem.json",
            "Units": "mm"
        },
        "y": {
            "Description": "Electrode y-coordinate in the coordinate system defined in coordsystem.json",
            "Units": "mm"
        },
        "z": {
            "Description": "Electrode z-coordinate in the coordinate system defined in coordsystem.json",
            "Units": "mm"
        },
        "size": {
            "Description": "Electrode contact diameter (or characteristic size)",
            "Units": "mm"
        },
        "material": {
            "Description": "Electrode contact material (e.g., Pt/Ir 90/10)"
        },
        "type": {
            "Description": "Electrode type (e.g., ECoG, SEEG, EEG). "
                           "Note: official BIDS requires type in channels.tsv; "
                           "here we keep it as an extra column for clarity."
        },
        "group": {
            "Description": "Electrode group/array label (e.g., Left_Sheet, Right_Sheet, Left_Ref, Right_Ref)"
        },
        "hemisphere": {
            "Description": "Hemisphere label",
            "Levels": {
                "L": "Left hemisphere",
                "R": "Right hemisphere"
            }
        }
    }

COORDINATE = {
  "defaults": {
    "size": 2.0,
    "material": "Pt/Ir 90/10",
    "type": "ECoG"
  },
  "group_defaults": {
    "Left_Sheet": { "hemisphere": "L" },
    "Right_Sheet": { "hemisphere": "R" },
    "Left_Ref": { "hemisphere": "L" },
    "Right_Ref": { "hemisphere": "R" }
  },
  "electrodes": [
    {"name":"CH01","x":0.0,"y":0.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH02","x":0.0,"y":7.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH03","x":0.0,"y":14.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH04","x":0.0,"y":21.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH05","x":5.0,"y":1.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH06","x":5.0,"y":6.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH07","x":5.0,"y":11.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH08","x":5.0,"y":16.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH09","x":5.0,"y":23.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH10","x":5.0,"y":30.5,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH11","x":0.5,"y":25.0,"z":0.0,"group":"Left_Ref"},
    {"name":"CH12","x":10.0,"y":3.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH13","x":10.0,"y":8.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH14","x":10.0,"y":13.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH15","x":10.0,"y":18.0,"z":0.0,"group":"Left_Sheet"},
    {"name":"CH16","x":10.0,"y":25.0,"z":0.0,"group":"Left_Sheet"},

    {"name":"CH17","x":0.0,"y":0.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH18","x":0.0,"y":7.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH19","x":0.0,"y":14.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH20","x":0.0,"y":21.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH21","x":-5.0,"y":1.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH22","x":-5.0,"y":6.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH23","x":-5.0,"y":11.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH24","x":-5.0,"y":16.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH25","x":-5.0,"y":23.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH26","x":-5.0,"y":30.5,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH27","x":-0.5,"y":25.0,"z":0.0,"group":"Right_Ref"},
    {"name":"CH28","x":-10.0,"y":3.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH29","x":-10.0,"y":8.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH30","x":-10.0,"y":13.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH31","x":-10.0,"y":18.0,"z":0.0,"group":"Right_Sheet"},
    {"name":"CH32","x":-10.0,"y":25.0,"z":0.0,"group":"Right_Sheet"}
  ]
}

# Data and output directories
DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/01_Data")
BIDS_DATA_DIR_PATH = Path("/work/project/ECoG_Monkey/BIDS_test_clean")
# DATA_CONFIG_FILE_PATH = Path("..") / "config.json"
# SAMPLES_FILE_DEFAULT = Path("..") / "samples.json"

script_path = Path(__file__)
script_dir = script_path.absolute().parent
project_dir = script_dir.parent
DEFAULT_BIDS_ROOT = project_dir / "BIDS_data"
DATA_CONFIG_FILE_PATH = project_dir / "config.json"
SAMPLES_FILE_DEFAULT = project_dir / "samples.json"
