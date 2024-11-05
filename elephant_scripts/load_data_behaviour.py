"""
This script loads sets up the directories used to load the audio and the metadata dataframe while also reading in the metadata file and defining the label columns. 
It then associates the audio files with their respective entries in the dataframe. 
Script uses code from tutorial by https://github.com/marathomas/tutorial_repo#readme


"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Project directory. By default set to current working directory.
P_DIR = Path.cwd()

# Audio directory, contains audio (.wav) files.
AUDIO_IN = P_DIR / "audio_dir"

# Empty data directory, output files will be put here.
DATA = P_DIR / "data"

# Information about the info_file.csv
LABEL_COL = "call-type"  # -->name of column that contains labels

NA_DESCRIPTORS = [0, np.nan, "NA", "na",  # Which values indicate that this vocalisation is unlabelled?
                  "not available", "None",
                  "Unknown", "unknown", None, ""]

# Check if directories are present
if not os.path.isdir(AUDIO_IN):
    print("No audio directory found")

if not os.path.isdir(DATA):
    os.mkdir(DATA)
    print("Data directory created:", DATA)

# Read in files
info_file = os.path.join(os.path.sep, DATA, '2.elephant_rumble_behaviour_df.csv')

if os.path.isfile(info_file):
    df = pd.read_csv(info_file)
    print("Info file loaded:", info_file)
else:
    print("Input file missing:", info_file)
    print("Creating default input file without labels")
    audiofiles = os.listdir(AUDIO_IN)
    if len(audiofiles) > 0:
        df = pd.DataFrame({'filename': [os.path.basename(x) for x in audiofiles], 'label': ["unknown"] * len(audiofiles)})
        print("Default input file created")
    else:
        print("No audio files found in audio directory")

#filter on only vocalisations that contain a rumble
df = df[df['call-type']== 'rumble']

audiofiles = df['filename'].values
files_in_audio_directory = os.listdir(AUDIO_IN)

# Are there any files that are in the info_file.csv, but not in AUDIO_IN?
missing_files = list(set(audiofiles) - set(files_in_audio_directory))
if len(missing_files) > 0:
    print("Warning:", len(missing_files), "files with no matching audio in audio folder")

audio_filepaths = [os.path.join(os.path.sep, AUDIO_IN, x) for x in audiofiles]

df['audio_filepaths'] = audio_filepaths
print("Audio file paths added to DataFrame")

print("Vocalisation Dataset successfully loaded")
