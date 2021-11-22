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




# st.write((plt.pyplot.hist(matrix["0"])))
# st.write(extract_feature(load_audio("samples/2024038730407998383.wav")))
# st.write(extract_feature(load_audio("/Users/paulfentress/Desktop/Mentia/Joyce_Audio/Joyce_audio-04.wav")))




#
#
# st.write(normalize_dataset(matrix.values, MM))
