# Installing General Dependencies



import pandas as pd
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


import Helper


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
MM = (Helper.dataset_minmax(matrix.values))

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

MM = (Helper.dataset_minmax(matrix.values))
X = Helper.normalize_dataset(matrix.values, MM)
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
k = 55
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

from Helper import extract_feature, load_audio, noise_reducer

def make_prediction(file_path):
  X = Helper.norm_input(file_path, MM)
  X = pca.transform(X)
  y_pred = KNeighborsClassifier.predict(X)


  # X = pd.DataFrame(extract_feature(load_audio(file_path, noise_reduce_on=True)))
  # y_pred = model.predict(X.values)
  return "This Audio is classified as speech with " + emotion_dict[y_pred[0]] + " emotion."
