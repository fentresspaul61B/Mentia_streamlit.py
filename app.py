


"""
I created the app.py file which has everything from
UI.py, Model.py, and Helper.py all together in one file
becaues there were issues with streamlit when using the
"import Helper" call from file to file



START OF HELPER FUNCTIONS

"""



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
# https://github.com/datarootsio/rootslab-streamlit-demo/blob/master/py

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




"""
START OF MODEL

"""


import glob
import seaborn as sns
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import plotly.graph_objs as go

# Importing the Data
data = pd.read_csv("Mentia_Data_Updated_nov20.csv").drop("Unnamed: 0", axis=1)
matrix = pd.read_csv("Audio_Features_updated_nov20.csv").drop("Unnamed: 0", axis=1)
labels = pd.read_csv("Labels_updated_nov20.csv")["Labels"]



# print(data.shape, matrix.shape, labels.shape)


# Importing ML Dependencies
import sklearn as sk
from sklearn import preprocessing
from tensorflow.keras.layers import Conv2D,MaxPool2D, Flatten, LSTM
from keras.layers import Dropout,Dense,TimeDistributed
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





  # MM represents the min max values. This is used to normalize the data.


"""
This is the MLP model. We have tried using with and without normalization,
and are getting similar results. Might need to train on more data, to get a more
representative sample.
"""

data = pd.read_csv("Mentia_Data_Updated_nov20.csv").drop("Unnamed: 0", axis=1)
matrix = pd.read_csv("Audio_Features_updated_nov20.csv").drop("Unnamed: 0", axis=1)
labels = pd.read_csv("Labels_updated_nov20.csv")["Labels"]

seed = 9

# Here is the design matrix with all of the audio features.
# MM represents the min max values. This is used to normalize the data.
MM = (dataset_minmax(matrix.values))

# X = normalize_dataset(matrix.values, MM)
X = matrix
# Labels: 1, 2, 3. Integers
Y = labels

# Shuffling the data so no patters get picked up in the algorithm about how the data is organized in the table.
sk.utils.shuffle(X, random_state=seed)

# Creating training, Testing tables for the design matrix and the corresponding labels
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .1, random_state = seed)

# Instantiating the Model
model=MLPClassifier(alpha=0.0001, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, momentum=0.7, random_state=seed)
model.fit(X_train,Y_train)

# Making Predictions on the Test Set
y_pred = model.predict(X_test)

# Get the classification accuracy on the test set
KNN_Y_test = Y_test.astype('int')
print("MLP Neural Network")
print(classification_report(KNN_Y_test, y_pred))

print(confusion_matrix(KNN_Y_test, y_pred))

"""
Trying different models.
"""

MM = (dataset_minmax(matrix.values))
X = normalize_dataset(matrix.values, MM)
seed = 9
Y = labels

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .1, random_state = seed)

from sklearn.decomposition import PCA
pca = PCA(n_components=131)

# Fitting the PCA model to the training data
X_train = pca.fit_transform(X_train)

# transofmring the test data
X_test = pca.transform(X_test)

# Using KNN Clustering to make predictions
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import *

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("Random Forrest")
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


"""
KNN Classifier
"""
# k = np.sqrt(data.shape[0])
k = 11
print("k " + str(k))

from sklearn.neighbors import KNeighborsClassifier as knn
clf = knn(n_neighbors=k)
KNeighborsClassifier = clf.fit(X_train, Y_train)

y_pred = KNeighborsClassifier.predict(X_test)

# Get the classification accuracy on the test data
KNN_Y_test = Y_test.astype('int')
print("KNN")
print(classification_report(KNN_Y_test, y_pred))
print(confusion_matrix(KNN_Y_test, y_pred))







"""
Below are helper functions that are connected to the record buton.

"""

def extract_feature(audio, mfcc=True, chroma=True, mel=True):
    audio = np.asarray(audio)
    sample_rate = 8000
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
    return np.array(result)


emotion_dict = {1: "Positive", 2: "Negative", 3: "Neutral"}



def make_prediction(file_path):
    # X = norm_input(file_path, MM)
    # X = pca.transform(X)
    # y_pred = KNeighborsClassifier.predict(X)


    X = pd.DataFrame(extract_feature(load_audio(file_path, noise_reduce_on=True))).T
    y_pred = model.predict(X.values)
    return "This Audio is classified as speech with " + emotion_dict[y_pred[0]] + " emotion."




"""
START OF UI

"""


import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import os
# from Helper import read_audio, record, save_record, extract_feature, normalize_dataset
# from Model import make_prediction
import Helper
import Model
import time
import librosa
from IPython.core.display import display as audio_display





paths = ([
    "Audio_Features_updated_nov20.csv",
    "Labels_updated_nov20.csv",
    "Mentia_Data_Updated_nov20.csv"
    ])

# .drop("Unnamed:0", axis=1)
data = pd.read_csv(paths[2]).drop("Unnamed: 0", axis=1)
matrix = pd.read_csv(paths[0]).drop("Unnamed: 0", axis=1)
labels = pd.read_csv(paths[1])["Labels"]


MM = Helper.dataset_minmax(matrix.values)


emotion_dict = {1: "Positive", 2: "Negative", 3: "Neutral"}


# giving each recorded file a unique hashcode to save as
# The purpose of doing this is because issues arose when
# naming two different files the same thing, because
# they were not re writing over eachother.
file_name = hash(time.time())


st.set_page_config(page_title="SER Mentia", page_icon = '🎙️')

# st.write(
# """
# # Fall 2021 UC Berkeley Data Discovery Project: Building a Speech Emotion Recognition Algorithm For People With Memory Loss
# Authors: Paul Fentress, Chi Hoang
# ## Define the problem:
# Can we predict positive, negative, and neutral emotion through speech for people with memory loss?
# """
# )




st.write("""# Mentia SER Algorithm
## Use the record button below to record an audio clip.
""")


import noisereduce as nr
# perform noise reduction
# reduced_noise = nr.reduce_noise(y=data, sr=rate)


# from Model import extract_feature
# from Helper import norm_input, load_audio, play_audio
if st.button("Record"):
    with st.spinner("Recording Your Audio..."):
    # st.write("Recording...")
        duration = 8
        # time.sleep(duration)

        fs = 44000
        path_myrecording = f"./samples/{file_name}.wav"
        my_recording = Helper.record(duration, fs)
        # my_recording = nr.reduce_noise(my_recording, sr=fs)
        Helper.save_record(path_myrecording, my_recording, fs)
        features = pd.DataFrame(Model.extract_feature(Helper.load_audio(path_myrecording), MM))
        normalized_features = pd.DataFrame(Helper.norm_input(path_myrecording, MM))
    st.success("Done!")
    st.write(Model.make_prediction(path_myrecording))
    st.write(features.T)
    # st.write(normalized_features)


    st.audio(Helper.read_audio(path_myrecording))


print("Complete")




# st.write((plt.pyplot.hist(matrix["0"])))
# st.write(extract_feature(load_audio("samples/2024038730407998383.wav")))
# st.write(extract_feature(load_audio("/Users/paulfentress/Desktop/Mentia/Joyce_Audio/Joyce_audio-04.wav")))




#
#
# st.write(normalize_dataset(matrix.values, MM))
