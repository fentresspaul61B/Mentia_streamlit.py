# Installing dependencies
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
# from encoder.inference import plot_embedding_as_heatmap
import sounddevice as sd
import wavio
import datetime
import numpy as np
import pandas as pd
sample_rate = 44000

data = pd.read_csv("Mentia_Data_Updated_nov20.csv").drop("Unnamed: 0", axis=1)
matrix = pd.read_csv("Audio_Features_updated_nov20.csv").drop("Unnamed: 0", axis=1)
labels = pd.read_csv("Labels_updated_nov20.csv")["Labels"]


# Got these helper functions from:
# https://github.com/datarootsio/rootslab-streamlit-demo/blob/master/helper.py

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
        # audio_bytes = librosa.load(file)
    return audio_bytes


def record(duration=8, fs=sample_rate):
    print("Recording ...")
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    print("Complete")

    return myrecording


def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


def extract_feature(audio, mfcc=True, chroma=True, mel=True):

  audio = np.asarray(audio)
  if chroma:
    # this means we are using a short Fuorier transform to convert the
    # data into a matrix of complex numbers.
      stft=np.abs(librosa.stft(audio))
  result=np.array([])
  if mfcc:
      mfccs=np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
  result=np.hstack((result, mfccs))
  if chroma:
      chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
  result=np.hstack((result, chroma))
  if mel:
    # returns array of mel spectrogram, which is a signal that is processed to
    # in order to match the human ear
      mel=np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
  result=np.hstack((result, mel))
  return result

# fs = 48000
# duration = 5
# filename = datetime.time
# path_myrecording = f"./samples/{filename}.wav"


# my_recording = record(duration, fs)
# save_record(path_myrecording, my_recording, fs)
import noisereduce as nr

"""
@software{tim_sainburg_2019_3243139,
  author       = {Tim Sainburg},
  title        = {timsainb/noisereduce: v1.0},
  month        = jun,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {db94fe2},
  doi          = {10.5281/zenodo.3243139},
  url          = {https://doi.org/10.5281/zenodo.3243139}
}


@article{sainburg2020finding,
  title={Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires},
  author={Sainburg, Tim and Thielk, Marvin and Gentner, Timothy Q},
  journal={PLoS computational biology},
  volume={16},
  number={10},
  pages={e1008228},
  year={2020},
  publisher={Public Library of Science}
}
"""

def noise_reducer(file_path):
  y, sr = librosa.load(file_path, sample_rate)
  reduced_noise = nr.reduce_noise(y=y, sr=sr)
  return reduced_noise


def load_audio(path, sr=sample_rate, noise_reduce_on=True):
    y, sr = librosa.load(path, sr=sr)
    if not noise_reduce_on:
        return y
    audio = nr.reduce_noise(y=y, sr=sr)
    return audio



"""
This function bellow displays audio to play.
It takes in a table, column_name which is for the
path_name, it takes in the row index "sample_num"
and it takes in the labels column name, which is by
default "labels"

"""

from IPython.display import Audio
from IPython.core.display import display as audio_display
from librosa import display

def play_audio(file_path):
  plt.figure(figsize=(12,4))
  test_audio = librosa.load(path=file_path, sr=8000)[0]
  librosa.display.waveplot(test_audio);
  # PNN stands for positive negative neutral
  return audio_display(Audio(file_path))





from IPython.core.display import display as audio_display

# This function is to normalize an audio input
def norm_input(file_path, min_max):
  # audio, sr = librosa.load(file_path, sr=8000)
  audio = load_audio(file_path, noise_reduce_on=True)
  extract_features = extract_feature(audio)
  # print("My Features are: " + str(extract_features))
  new_col = []
  # scaled_value = (value - min) / (max - min)
  for idx, MM in enumerate(min_max):
    val = (extract_features[idx] - MM[0]) / (MM[1] - MM[0])
    new_col.append(val)
  # min_val = min(extract_features)
  # max_val = max(extract_features)
  # minmax = [min_val, max_val]
  # for i in range(len(extract_features)):
  #   val = (extract_features[i] - minmax[0]) / (minmax[1] - minmax[0])
  #   new_col.append(val)
  return pd.DataFrame(new_col).T


# Helper Functions for normalzing data
# This computs the min and max values by column in a data set.
# I decided to do normalization manually because I was having issues
# When trying to use new data, and it wasnt being normalized properly
# using sci kit learn and in return predicting everything as negative. To fix this
# I decided to pull up my bootstraps and just use functions that do it manually.
# The input of this function is a dataset, the output is an list of lists.
# Each item in the list is the column wise min and max's
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset.copy()[0])):
		col_values = [row[i] for row in dataset.copy()]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
  for row in dataset:
    for i in range(len(row)):
      row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
  return dataset
