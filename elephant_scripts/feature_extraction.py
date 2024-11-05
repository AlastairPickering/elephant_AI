"""
This script carries out all of the audio pre-processing steps before extracting
the acoustic features (embeddings) using the pre-trained CNN vggish and adding
in the missing duration information.
"""
# Import libraries

import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# Set path

P_DIR = Path.cwd()

# Here we define the default values for the parameters that will be used in the
# audio pre-processing steps.
DEFAULT_SAMPLERATE = 4000  # Hz

# This is the duration of audio that will be extracted from the original audio
# and passed to the vggish model. The vggish model takes 0.96s
# of audio at 16kHz as input, which is 15360 samples. Since the samplerate of
# the audio files is 4000Hz, we need to change the window duration to 4s
# in order to extract the correct number of samples.
DEFAULT_WINDOW_SIZE = 4  # seconds

# Audio of annotations will be extracted from the original audio file and
# inserted into a 4s window. This parameter defines the position of the
# annotation within the 4s window.
DEFAULT_CLIP_POSITION = "middle"

# Define functions needed to read audio files, apply frequency bandpass filter,
# extract the annotation based on timestamp, normalise amplitude, zero-pad and
# centre the annotation with the vggish-compatible 0.96s input window.


def read_audio(path):
    wav, sr = librosa.load(path, sr=None)
    return wav, sr


def apply_bandpass_filter(
    wav, low_freq, high_freq, samplerate=DEFAULT_SAMPLERATE, order=5
):
    # Define the bandpass filter
    sos = scipy.signal.butter(
        order,
        [low_freq, high_freq],
        fs=samplerate,
        btype="band",
        output="sos",
    )

    # Apply the bandpass filter to the signal
    filtered_signal = scipy.signal.sosfilt(sos, wav)
    return filtered_signal

def extract_audio(wav, annotation, samplerate=DEFAULT_SAMPLERATE):
    start_index = int(annotation.start_time * samplerate)
    end_index = int(annotation.end_time * samplerate)
    
    # Adjust start and end indices to ensure they are within the audio boundaries
    start_index = max(0, start_index)
    end_index = min(len(wav), end_index)
    
    extracted_audio = wav[start_index:end_index]
    return extracted_audio

def zero_pad(annotation, window_size, samplerate, wav):
    """
    Zero-pad and centre the input audio waveform based on the provided annotation.

    Args:
        annotation (Annotation): The annotation object containing start and end times.
        window_size (float): The duration of each window for acoustic analysis.
        samplerate (int): The sample rate of the audio.
        wav (numpy.ndarray): Input audio waveform.

    Returns:
        numpy.ndarray: Zero-padded and centred audio waveform.
    """
    annotation_duration = annotation.end_time - annotation.start_time
    num_windows = np.ceil(annotation_duration / window_size)
    return_duration = num_windows * window_size
    return_size = int(return_duration * samplerate)

    # Calculate the amount of zero-padding needed
    padding_needed = max(0, return_size - len(wav))

    # Calculate the start index for placing the waveform
    start_index = max(0, padding_needed // 2)

    # Pad the waveform with zeros and centre it
    padded_wav = np.pad(wav, (start_index, return_size - len(wav) - start_index))

    return padded_wav

def normalise_sound_file(data):
    # Calculate the peak amplitude
    peak_amplitude = np.max(np.abs(data))

    # Set the whole sound file to the peak amplitude
    normalised_data = data * (1 / peak_amplitude)

    return normalised_data

def wav_cookiecutter(
    annotation,
    window_size=DEFAULT_WINDOW_SIZE,
    position=DEFAULT_CLIP_POSITION,
    samplerate=DEFAULT_SAMPLERATE,
):
    """Extract the acoustic features of a single annotation."""
    # Get path of the audio file from the annotation info
    path = str(annotation.audio_filepaths)

    # Read the audio
    wav, sr = read_audio(path)

    # If the samplerate of the audio file does not match the samplerate
    # then the filtering and annotation extraction will produce
    # incorrect results, so we need to check this.
    assert sr == samplerate

    # Apply the bandpass filter to the signal
    wav = apply_bandpass_filter(
        wav, annotation.low_freq, annotation.high_freq, samplerate=samplerate
    )

    # Extract audio segment based on annotation times
    wav = extract_audio(wav, annotation, samplerate=samplerate)

    # Zero-pad the wav array based on the annotation
    wav = zero_pad(annotation, window_size, samplerate, wav)

    # Normalise the sound file
    normalised_clip = normalise_sound_file(wav)

    # Re-apply the bandpass filter to remove artifacts
    refiltered_wav = apply_bandpass_filter(normalised_clip, annotation.low_freq, annotation.high_freq, samplerate=samplerate)

    return refiltered_wav

# Pass the pre-processed data to the vggish model to extract the automated
# acoustic features


def feature_extraction(
    df, samplerate=DEFAULT_SAMPLERATE, window_size=DEFAULT_WINDOW_SIZE
):
    """Extracts all features from annotation data in dataframe."""
    # Load the vggish model
    model = hub.load('https://tfhub.dev/google/vggish/1')

    # Extract VGGish features
    results = []
    for _, annotation in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Apply the wav_cookiecutter function combining all of the audio
        # pre-processing steps
        wav = wav_cookiecutter(
            annotation,
            window_size=window_size,
            position="middle",
            samplerate=samplerate,
        )

        embeddings = model(wav)
        assert embeddings.shape[1] == 128  # Check the number of features per frame

        # Store info of the embeddings of each frame
        for embedding in embeddings:
            results.append(
                {
                    "recording_id": annotation.recording_id,
                    **{f"feature_{n}": feat for n, feat in enumerate(embedding)},
                }
            )

    print("Features successfully extracted")

    results = pd.DataFrame(results)
    # average vggish annotation feature vectors back into original # of annotations
    results = results.groupby("recording_id").mean()

    print("Features successfully averaged per vocalisation")

    # Add in the missing duration information as the 129th feature
    duration = df[["recording_id", "duration"]].copy()
    results = results.join(duration.set_index("recording_id"))

    # Store the embeddings in the results dataframe
    print("Duration successfully added as the 129th feature")
    return pd.DataFrame(results)  # Return the processed results


# Define function to plot spectrograms

def plot_spectrograms(steps, audios, sr=DEFAULT_SAMPLERATE, num_rows=3, num_cols=2, window_size=DEFAULT_WINDOW_SIZE, position=DEFAULT_CLIP_POSITION):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))

    # Create an additional axis for the color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # Adjust [left, bottom, width, height]

    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    
    for i, (step, audio) in enumerate(zip(steps, audios)):
        row = i // num_cols
        col = i % num_cols

        ax = axes[row, col]

        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        img = librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')

        # Adjust font sizes of x and y-axis labels and tick labels
        ax.set_xlabel('Time (s)', fontsize=20)  # Set font size for x-axis label
        ax.set_ylabel('Frequency (Hz)', fontsize=20)  # Set font size for y-axis label
        ax.tick_params(axis='both', which='both', labelsize=20)  # Set font size for tick labels

        # Add subplot label
        ax.text(-0.1, 1.0, labels[i], transform=ax.transAxes, size=20)

    # Add a single colour bar for the entire figure
    cbar = fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colour bar
    
    # Increase space between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.show()


print("Functions for Audio Pre-processing and Feature Extraction successfully loaded")
