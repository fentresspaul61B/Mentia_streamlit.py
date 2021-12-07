import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# from Helper import read_audio, record, save_record, extract_feature, normalize_dataset
# from Model import make_prediction
import Helper
import Model
import time
import librosa
import seaborn as sns
from IPython.core.display import display as audio_display





paths = ([
        "Audio_Features_updated_nov20.csv",
        "Labels_updated_nov20.csv",
        "Mentia_Data_Updated_nov20.csv",
        "G_matrix.csv"
        ])

# .drop("Unnamed:0", axis=1)
data = pd.read_csv(paths[2]).drop("Unnamed: 0", axis=1)
matrix = pd.read_csv(paths[0]).drop("Unnamed: 0", axis=1)
labels = pd.read_csv(paths[1])["Labels"]

def cleaner(string):
    if string == "Carol":
        return "Anon_1"
    elif string == "joyce":
        return "Anon_2"
    else:
        return string


data["Data_set"] = data["Data_set"].map(cleaner)

g_matrix = pd.read_csv(paths[3]).drop("Unnamed: 0", axis=1).drop("Labels", axis=1)
g_labels = pd.read_csv(paths[3])["Labels"]


MM = Helper.dataset_minmax(matrix.values)

labels = labels.append(g_labels)
matrix = matrix.append(g_matrix)

emotion_dict = {1: "Positive", 2: "Negative", 3: "Neutral"}


# giving each recorded file a unique hashcode to save as
# The purpose of doing this is because issues arose when
# naming two different files the same thing, because
# they were not re writing over eachother.
file_name = hash(time.time())


st.set_page_config(page_title="SER Mentia", page_icon = '🎙️')


# Setting font:

st.write(
"""
# Fall 2021 UC Berkeley Data Discovery Project: Building a Speech Emotion Recognition Algorithm For People With Memory Loss

Authors: Paul Fentress, Chi Hoang"""
)


# t = st.radio("Toggle to see font change", [True, False])
t = False

if t:
    st.write(
        """
        <style>
@font-face {
  font-family: 'Times';
  font-style: normal;
  font-weight: 50;
  src: url(https://fonts.gstatic.com/s/tangerine/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}

    html, body, [class*="css"]  {
    font-family: 'Tangerine';
    font-size: 14px;
    }
    </style>

    """,
        unsafe_allow_html=True,
    )













# st.write(g_matrix.shape)
# st.write(matrix.append(g_matrix))
# st.write(labels.apend(g_labels))


st.write("""
# Introduction:

Speech is a vital way to express one's emotions, needs and thoughts. The ability to speak coherently becomes increasingly challenging for people that are experiencing this heartbreaking cognitive impairment. As a result, memory loss can cause stress to not only the patient but also to their family and their caregivers.

"Since 1900, the percentage of Americans age 65 and older nearly quadrupled (from 4.1% in 1900 to 16% in 2019), and the number increased more than 17 times (from 3.1 million to 54.1 million)." [https://acl.gov/aging-and-disability-in-america/data-and-research/projected-future-growth-older-population](https://acl.gov/aging-and-disability-in-america/data-and-research/projected-future-growth-older-population) Dementia appears commonly in between the ages of 65 and onward and is usually chronic, dysfunctional, and secondary to neurodegenerative processes for which there is currently no cure. ([https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6195406/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6195406/)). Speech impairment is one of the most common struggles that patients with dementia, unfortunately, have to face. ([https://www.sciencedirect.com/science/article/abs/pii/S1214021X15300387](https://www.sciencedirect.com/science/article/abs/pii/S1214021X15300387)).

In a world where those with cognitive impairments are often misunderstood or ignored, technology including Speech Emotion Recognition (S.E.R.) is a powerful tool for filling in this gap and elevating therapeutic digital experiences.

## What is DevaWorld?

Mentia's signature project DevaWorld is currently the leading technology of digital therapy for those with Dementia, Memory loss, or who are Cognitively Impaired.  DevaWorld is an interactive virtual world that is accessed through an app and played on an Ipad with the assistance of a caregiver. Inside DevaWorld, the user is guided through different options such as playing the piano, watering plants, or eating virtual chocolates with the avatar Julie.

"""
)
from PIL import Image
julie = Image.open("Julie.png")
st.image(julie,width=100, caption = "In Game Avatar Julie")


st.write(
"""
One of the most frustrating parts of memory loss is losing one's sense of autonomy and accomplishment. DevaWorld is a source of therapeutic entertainment that restores these things.  Currently, DevaWorld is always played with a caregiver; however, by implementing AI into Julie, DevaWorld will be playable without the assistance of a caregiver so they can focus on more important tasks.

"""

)

st.write(
"""
## Project Goal

Currently, Julie prompts the user with questions such as "would you like to play the piano?" and other suggestions in a predictable manner which depend on where the user touches on the screen. Mentia's plan for DevaWorld 3.0 is to incorporate AI into Julie so the user's experience is more engaging while making DevaWorld playable without caregiver assistance.

Speech to text, Computer Vision, and S.E.R. are the forms of intelligence Mentia plans to incorporate into Julie. Chi and I were tasked with creating the S.E.R. algorithm piece of Julies Intelligence. Specifically, we were tasked to build a model that accurately classifies the user's emotions as positive, negative, or neutral.

Our success metric was model accuracy, which we wanted to achieve 85% model accuracy.

The diagram below shows a high-level overview of how our S.E.R. will be used, and which specific part of the project Chi and I worked on which is inside the teal highlighted box.
"""

)

st.write(
"""
**Step 1: Input Signal**

First, a user speaks out loud while playing DevaWorld:
"""
)

diagram_1 = Image.open("Diagram_1.png")
st.image(diagram_1)

st.write(
"""
**Step 2: Audio Data Collection (Microphone)**

Second, the microphone on the IPad will pick up the Audio Data while the User is playing DevaWorld.
"""
)

diagram_2 = Image.open("Diagram_2.png")
st.image(diagram_2)

st.write(
"""
**Step 3: Predictive Model (Our Project)**

Process the audio data. Make a classification of Positive, Negative, or Neutral.
"""
)

diagram_3 = Image.open("Diagram_3.png")
st.image(diagram_3)

st.write(
"""
**Step 4: Julies Action**

Julie will make a more intelligent prompt or no prompt in the game based on the prediction from our S.E.R. model.
"""
)

diagram_4 = Image.open("Diagram_4.png")
st.image(diagram_4)

st.write(
"""
# About the Data

Collected data came from multiple sources:
- 281 Samples of 2 speakers came from an interview that was conducted at Elder Ashram
- 18 samples of a speaker came from an interview that was conducted by Mentia experts
- 178 samples of more than speakers: 5+)
- Tess: SER data set from Kaggle (1527 Samples, Speakers: 2)
- Ravdness: SER data set from Kaggle (763 Samples, Speakers: 24)
- Savee: SER data set from Kaggle (286 Samples, Speakers: 4)
- There are 37 total speakers in the training data set and only 13 different speakers for people elderly people or people with Memory Loss
- 50% of the Training data comes from the Tess dataset (1527 samples)
- Only one language included: English

"""
)

fetures_with_lables = pd.read_csv(paths[0]).drop("Unnamed: 0", axis=1).join(labels).join(data)




# st.cache
st.write("Number of Samples By Emotion")
color_pallete = "viridis"

dist_of_categories_simp = data.groupby("Emotion_string", as_index=False).count().sort_values("Data_set")

x = dist_of_categories_simp["Emotion_string"]
y = dist_of_categories_simp["name"]
df = pd.DataFrame()
df["Emotion_string"] = x
df["Number of Samples"] = y
df = df.sort_values("Number of Samples", ascending=True)
import altair as alt
c = (alt.Chart(df)
     .mark_bar()
     .encode(
     x='Emotion_string',
     y='Number of Samples',
     tooltip=['Emotion_string', 'Number of Samples'],
     color=alt.Color('Emotion_string', scale=alt.Scale(scheme=color_pallete)))
     )
st.altair_chart(c, use_container_width=True)



st.write("Number of Samples By Data Set",)


dist_of_data_sets = data.groupby("Data_set", as_index=False).count().sort_values("Data_set")
x = dist_of_data_sets["Data_set"]
y = dist_of_data_sets["name"]
df = pd.DataFrame()
df["Data_set"] = x
df["Number of Samples"] = y
df = df.sort_values("Number of Samples", ascending=True)
import altair as alt
c = (alt.Chart(df)
     .mark_bar()
     .encode(
     x='Data_set',
     y='Number of Samples',
     tooltip=['Data_set', 'Number of Samples'],
     color=alt.Color('Data_set', scale=alt.Scale(scheme=color_pallete)))
     )
st.altair_chart(c, use_container_width=True)

st.write(
"""
## Data Labeling

We labeled the data positive, negative or neutral by listening to audio clips and using our judgment. In order to verify what the team agreed on for labeling, we sent out a survey of audio examples to the Mentia team to classify as positive, negative, or neutral. After the team took the survey, we had a better understanding of how to label the data. Each sample is numbered based on a labeling system: 1 is for a positive response, 2 is for a negative response and 3 is for a neutral response.

With that being said, this labeling process is inherently a very subjective task. Moreover, the context of the video could introduce bias, so we tried to focus on listening to the audio clip as an individual clip out of context while labeling the data.

## Issues With The data and Our Solutions

One of the issues we had was collecting a sufficient amount of representative data to train on. We were given 172 videos to use, which were recordings of People with Memory Loss playing DevaWorld, and them speaking their opinion on changes that were going to be added to the game. Mentia's lead scientist was conducting the interview, and guiding the people with Memory Loss through the DevaWorld. Due to the videos being two or more people, isolate the samples where the person with Memory loss was speaking, not the scientist performing the interview. The process of converting the video to audio, splitting the audio into samples to isolate the correct speaker, then labeling the samples was cumbersome.

In order to overcome this, we sought out other data sets that were pre-labeled. A limitation of doing this is that we were including audio samples of people without memory loss, and therefore our dataset, was not the ideal representation for this problem. With more time cold emailing other researchers, we could create a more representative data set with more elderly people, and people with memory loss to create the ideal training data set. Furthermore, adding a wide range of different people, at different stages of memory loss, with multiple languages would
create a more representative data set.

We have a lot of noise in the Devaworld samples, for example, there are often multiple voices in the samples, or music playing from the Devaworld app. Additionally, Julie, the avatar in Devaworld, is speaking at the same time as the people playing Devaworld. The video quality of the Mentia Videos is relatively low compared to the other samples, therefore there is a lot of white noise in the samples from Mentia. In order to deal with the white noise, we used a noise reduction python package; however, we have not yet dealt with isolating a single voice from a group, which means the model could give poor results when trying to classify a sample where two people are talking over each other.

In version 3.0 of Devaworld, the goal is to have the user play DevaWorld without caregiver assistance, and therefore the problem of overlapping voices will not be an issue in this scenario.
"""
)


# string = ["A_97_year_old_video", "AlzheimerDisease_video", "Anon_1", "Ravdness", "Savee", "elderly_advice_video", "ep_18"]


st.write(

"""
# Methods

We used Machine Learning to address this question. We collected audio data, extracted signals from the audio, and built a predictive model.

## Feature Selection

In order to extract audio features, we used the Librosa Python Package. Librosa has many useful tools for processing, listening, and visualizing audio in Python. For our model, we extracted three features from our audio data which were: Chroma, MFS, and MFCC. [https://librosa.org/doc/](https://librosa.org/doc/)

""")

st.write(

"""
## Chroma

"In Western music, the term chroma feature or Chromagram closely relates to the twelve different pitch classes. Chroma-based features, which are also referred to as "pitch class profiles", are a powerful tool for analyzing music whose pitches can be meaningfully categorized (often into twelve categories) and whose tuning approximates to the equal-tempered scale. One main property of chroma features is that they capture harmonic and melodic characteristics of music, while being robust to changes in timbre and instrumentation." [https://en.wikipedia.org/wiki/Chroma_feature](https://en.wikipedia.org/wiki/Chroma_feature)

Chroma is a mathematical way of sorting audio frequency signals into bins that correspond to the Western musical scale: {C, C♯, D, D♯, E , F, F♯, G, G♯, A, A♯, B}. There are 12 musical notes in the western musical scale and the Librosa chroma method detects the presence of these 12 notes from an audio sample and returns a matrix with 12 columns. We flatten this matrix by taking the mean value of the 12 columns corresponding to each note, which returns a (12,1) flattened array of numerical values.

"""

)

with st.echo():
    example_path = "samples/11520134717607883.wav"
    fig = Helper.plot_wav_file(example_path)
    st.pyplot(fig = plt)
    # This is creating the audio player
    st.audio(Helper.read_audio(example_path))

    fig = Helper.plot_features(example_path, n_mfcc=7, mel=False, chroma=True )
    st.pyplot(fig = plt)


st.write(
"""

## Mel Frequency Spectrogram:

[https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

[https://www.kaggle.com/shivamburnwal/speech-emotion-recognition](https://www.kaggle.com/shivamburnwal/speech-emotion-recognition)

The Mel Scale is a logarithmic transformation of audio in order to fit the human hearing range. Audio signals can be quantified into frequencies, and the human hearing range is from 20 - 20,000 hertz. The sound of low sub-bass that you hear at a concert, or the rumble of seismic plates being shifted is around the 0 - 30 hertz range, while white noise and hissing sounds are around the 20,000-hertz range. The interesting thing about human hearing is that our perception of audio is not linear. For example, a human can easily tell the difference between 100 and 200 hertz (kick drum vs a snare), while the difference between 7000 and 7100 hertz is barely audible.  If we kept our data on a linear scale, our analysis and models would consider a 5000 and 5200 frequency as equally different than a 200 and 400-hertz noise, which as stated above is not the case for our own experience, which is why we squish our audio into the Mel Scale so the difference between 200 and 400 is substantial while the difference between 5000 and 5200 is less substantial, therefore creating a better representation of human perception.

Now that we understand the Mel Scale, what is a Spectrogram? "A spectrogram is a detailed view of audio, able to represent time, frequency, and amplitude all on one graph. A spectrogram can visually reveal broadband, electrical, or intermittent noise in the audio, and can allow you to easily isolate those audio problems by sight." [https://www.izotope.com/en/learn/understanding-spectrograms.html](https://www.izotope.com/en/learn/understanding-spectrograms.html)

(Octave is twice the frequency on the musical scale, 3 times the frequency is the dominant note)

In our case we are not so much interested in the visualization of the spectrogram, rather we are interested in the numerical values of it of the most prominent frequencies for a given time slice. A spectrogram is created by taking the Fast Fourier Transform (FFT) of multiple time slices of an audio sample. The Mel Frequency spectrogram is a logarithmically scaled numerical description of audio frequencies over time. [https://www.youtube.com/watch?v=spUNpyF58BY](https://www.youtube.com/watch?v=spUNpyF58BY)

For our feature extraction function, our audio is sliced into 128 beats per minute, which in turn is a matrix with 128 columns. The number of rows ranges from 0 to the highest present frequency. We then take the mean of all 128-time slices which gives us an array of 128 values each representing the average frequency at that point in time for the given sample.

"""
)

with st.echo():
    fig = Helper.plot_features(example_path)
    st.pyplot(fig = plt)

st.write(
"""
## MFCC: Mel-Frequency Cepstral Coefficients

MFCC's represent distinct units of sounds which are the individual parts that compose the Mel-frequency cepstrum. The Mel-Frequency Cepstrum is a way to model the human vocal tract. "Any sound generated by humans is determined by the shape of their vocal tract (including tongue, teeth, etc). If this shape can be determined correctly, any sound produced can be accurately represented." [https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b](https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b)

These coefficients are used to isolate the most relevant audio signals to human hearing, which is a relatively small range of frequencies compared to the total range of sound frequencies. The first 12-13 coefficients are the most relevant to human hearing, while any after that are less important. The reason why we have 40 MFCC features is that Librosa takes the first and second derivative of the 13 coefficients which ends up being 13 + 13 + 13, and then one extra coefficient is taken to make an even number of 40.
"""
)

with st.echo():
    fig = Helper.plot_features(example_path, n_mfcc=40, mel=False)
    st.pyplot(fig = plt)


st.write(
"""
## Features Summary

- 12 from CHROMA. These are the 12 musical notes on a western musical scale and their corresponding frequency bins.
- 128 from the Mel Frequency Spectrogram, where are the means of the MFS for 128-time slices. This feature shows how prominent different frequencies are in a given time slice, measured by volume.
- 40 from MFCC: Mel-Frequency Cepstral Coefficients, where each value is the mean of the MFCC for to top 40 most relevant coefficients to model the human vocal tract.

128(MFS) + 40(MFCC) + 12(Chroma) = 180 total features
"""
)

st.write(
"""
## Loading Data in Batches Using Pytorch DataLoader

Now that we understood the features and extracted them from our audio samples, we added them to a table. After attempting to load all the data at once, we ran into dead kernels and extended run times, so we switched to using a Pytorch DataLoader. In order to extract the features from the data, we Librosa's audio loading method.

"""
)



with st.echo():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    all_file_paths = np.array(data["file_path"])

    loaded_data = []

    def librosa_in_batches(data_set, saple_rate=8000):
      # this means we will only upload 128 samples at a time
      # whic makes it much easier on our cpu
      BATCH_SIZE = 128


      file_loader = DataLoader(data_set, batch_size=BATCH_SIZE)
      batches = []
      for i, batch in enumerate(file_loader):
        batches.append(batch)
          # print(i, batch)
      batches = np.array(batches)
      number_of_batches = batches.shape[0]
      batch_num = 0
      file_num = 0
      print("Total number of batches to load: " + str(number_of_batches))
      for each_batch in batches:
        batch_num +=1
        print("Loading batch number: " + str(batch_num))
        for each_file in each_batch:
          loaded_data.append(librosa.load(each_file, sr=saple_rate))
          file_num +=1
          if (file_num % 5 == 0):
            print(".")
            print("Loaded data current length: " + str(len(loaded_data)))

      print("Loading Completed!")


st.write(
"""
## Feature Extraction
We adopted our feature extraction function from the project linked here:
https://www.kaggle.com/shivamburnwal/speech-emotion-recognition
"""
)

with st.echo():
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
      return result

st.write(
"""
## Features Table
"""
)

st.write(matrix)


st.write(
"""
## **Data Pre-processing**

**How did you Pre-process the Data?**

- We used Normalization for our MLP Neural Network Model
- We used PCA for our KNN and Random Forrest Models

**What is the purpose of Pre-processing for SER?**

The reason we chose to use normalization is because our audio data does not follow a normal distribution and because we used a gradient-based prediction algorithm: MLP Neural Network. In order to verify that our data was indeed not normally distributed we used a QQ (Quantile Quantile) plot to verify how our data was distributed. A QQ plot is a way of visually checking the distribution of data.

A QQ plot that has the points scatter straight along the line means that the sample data does follow the distribution you have specified (In our case we wanted to see if our data follows a normal distribution.). If your data diverges from the straight line that means your data does not follow the distribution you have specified. A QQ plot compares the quantiles of a distribution to our sample data and sees how correlated they are for all data points. If our sample data is highly correlated with the target distribution, the data will fit the target distribution line, otherwise, our sample data will diverge from the target line.

"""
)

# Resseting the figure
plt.figure()
with st.echo():
    import scipy

    measurements = matrix.values.reshape(matrix.shape[0]*matrix.shape[1])

    fig = scipy.stats.probplot(measurements, dist="norm", plot=plt)

    st.pyplot(fig = plt)


st.write("""
We found that our extracted feature data diverged from the target distubtion around the tails, which meant that our features do not follow a normal distrubtion.

After using a helper function to normalize our features, our data is ready for predictions:

### Features Table (After Normalization)
""")

with st.echo():
    MM = (Helper.dataset_minmax(matrix.values))
    X = Helper.normalize_dataset(matrix.values, MM)
    st.write(X)


# data_dict = {"a": 1, "b": 2, "c": 3}
# data_items = data_dict. items()
# data_list = list(data_items)
# df = pd. DataFrame(data_list) create DataFrame from `data_list`
# print(df)



st.write(
"""
# Model Selection
We iterated through 3 different models which are ideal for data which do not follow a normal distribution and obtained the following accuracy for each model:
- KNN
- PCA + Random Forest
- MLP Neural Network

Using PCA, we can reduce the dimension of the dataset while minimizing information loss.  First, we instantiated a PCA function with a baseline of 97 components.
"""

)
plt.figure()
# num_vars = 180
# features_mean = np.mean(matrix, axis=0)
# features_std = np.std(matrix, axis=0)
# features_centered = (matrix - features_mean) / features_std
#
# u, s, vt = np.linalg.svd(features_centered, full_matrices=False)
#
# eigvals = s**2 / sum(s**2)  # NOTE (@amoeba): These are not PCA eigenvalues.
#                                # This question is about SVD.
#
# sing_vals = np.arange(num_vars) + 1
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue')
# plt.xticks(np.arange(0,26, 2))
# scree = plt.plot(np.arange(25), eigvals[:25]);

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn

# Normalzing the data:
MM = (Helper.dataset_minmax(matrix.values))
X = Helper.normalize_dataset(matrix.values, MM)
seed = 9
Y = labels

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .1, random_state = seed)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train = np.array(Y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

#instantiate random forest model and fit the scaled data to the model
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)


pca_test = PCA(n_components=97)
pca_test.fit(X_train_scaled)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)


st.pyplot(fig = plt)


evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
st.write(pca_df.head(90))


st.write(
"""
The above graph shows that even when PCA reduces our predicting variables to 87 components, our dataset still covers more than 95% of the variance. Therefore, there is no need to keep all 180 features. We then fit our training test set and testing test set into this PCA model.
	To check if there’s any improvement we can get, we now fit our training data and testing data into a Random Forest model. We first tried to see which parameters are best fit for our model by using Randomized Search CV. This gave us the following results:

"""



)


# st.cache
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn

# Normalzing the data:
MM = (Helper.dataset_minmax(matrix.values))
X = Helper.normalize_dataset(matrix.values, MM)
seed = 9
Y = labels

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .1, random_state = seed)

from sklearn.decomposition import PCA

# Using scree plot analysis we found 87 principle components do
# retain 99% of variance.
pca = PCA(n_components=87)

# Fitting the PCA model to the training data
X_train = pca.fit_transform(X_train)

# transofmring the test data
X_test = pca.transform(X_test)

# Choosing number of neighbors after iterating through
# different values for k, 31 gave the best results.
k = 51

# Fitting the model:
clf = knn(n_neighbors=k)
KNeighborsClassifier = clf.fit(X_train, Y_train)

# Making Predictions:
y_pred = KNeighborsClassifier.predict(X_test)


st.write(

"""
## Initial Model Accuracy
""")

dct = ({
        "KNN": 0.8355263157894737,
        "Random Forrest": .81
})

accuracy_table = pd.DataFrame(list(dct.items()))
accuracy_table = accuracy_table.rename(columns={0: "Model", 1: "Accuracy"})

st.write(accuracy_table)

# Accuracy:
# st.write(sum(Y_test ==  y_pred) / len(y_pred))

# st.write("KNN Model Accuracy: 0.8585526315789473")


#@title
# import plotly.figure_factory as ff
# # st.cache
# z = confusion_matrix(Y_test, y_pred)[::-1]
# # z = confusion_matrix( y_pred, KNN_Y_test)
#
# x = ['Positive', 'Negative', 'Neutral']
# y = x[::-1].copy()
#
# # change each element of z to type string for annotations
# z_text = [[str(y) for y in x] for x in z]
#
# # set up figure
# fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale=color_pallete)
#
# # add title
# fig.update_layout(title_text='<b>Confusion matrix</b>',
#                   #xaxis = dict(title='x'),
#                   #yaxis = dict(title='x')
#                  )
#
# # add custom xaxis title
# fig.add_annotation(dict(font=dict(color="black",size=14),
#                         x=0.5,
#                         y=-0.15,
#                         showarrow=False,
#                         text="Predicted value",
#                         xref="paper",
#                         yref="paper"))
#
# # add custom yaxis title
# fig.add_annotation(dict(font=dict(color="black",size=14),
#                         x=-0.35,
#                         y=0.5,
#                         showarrow=False,
#                         text="Real value",
#                         textangle=-90,
#                         xref="paper",
#                         yref="paper"))
#
# # adjust margins to make room for yaxis title
# fig.update_layout(margin=dict(t=50, l=200))
#
# # add colorbar
#
#
# # fig.update_layout(height=500, width=700)
# fig['data'][0]['showscale'] = True
# st.plotly_chart(fig)




st.write("""

## Final Model: MLP Neural Network

A multi-layer perceptron is a classification algorithm that consists of multiple layers of perceptrons. In addition to an input layer, an output layer, MLP can have more than one hidden layer. Each hidden layer consists of neurons that use non-linear activation functions. These are known as nodes. Each node in one layer connects with a certain weight to every node in the following layer (https://en.wikipedia.org/wiki/Multilayer_perceptron). The output layer acts as a logical net that chooses an index to send to the output on the basis of inputs it receives from the hidden layer so that the classification error is minimized. (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.329.3346&rep=rep1&type=pdf).

""")

MLP_image = Image.open("MLP.png")
st.image(MLP_image)

st.write(

r"""
source: [https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)

Consider the network in Figure 1. Its input layer consists of
a set of input features $X = \left \{ x_i | i \in {\displaystyle \mathbb {N} } \right \}$.

Each $x_i$ is a neuron. Each neuron in the network performs
a weighted linear summation of its input,

$$\sum_{i = 1}^{N} w_{i}x_{i} + \theta$$

 followed by a nonlinear activation function g (https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)


$$g(.) = R \rightarrow R$$
"""

)




st.write("""
([https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)).

This gives us an ability to separate our data which is non-linearly separable. (this is why we choose the MLP model). However, one of the disadvantages of MLP is that it starts with a different set of random weights. This can result in different validation accuracy ([https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)). MLP also requires users to tune some hyperparameters such as the number of hidden neurons, layers, and iterations. Too many hidden neurons can cause overfitting, but too little hidden neurons can result in under-fitting the data ([https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.329.3346&rep=rep1&type=pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.329.3346&rep=rep1&type=pdf)).


## Our MLP Model Using Scikit Learn:

"""
)

data = pd.read_csv("Mentia_Data_Updated_nov20.csv").drop("Unnamed: 0", axis=1)
matrix = pd.read_csv("Audio_Features_updated_nov20.csv").drop("Unnamed: 0", axis=1)
labels = pd.read_csv("Labels_updated_nov20.csv")["Labels"]



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
with st.echo():
    # st.cache
    import sklearn as sk
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    seed = 5

    # Here is the design matrix with all of the audio features.
    # MM represents the min max values. This is used to normalize the data.
    MM = (Helper.dataset_minmax(matrix.values))

    X = Helper.normalize_dataset(matrix.values, MM)
    # X = matrix
    # Labels: 1, 2, 3. Integers
    Y = labels

    # Shuffling the data so no patters get picked up in the algorithm about how the data is organized in the table.
    sk.utils.shuffle(X, random_state=seed)

    # Creating training, Testing tables for the design matrix and the corresponding labels
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .1, random_state = seed)

    # Instantiating the Model
    MLP_model=MLPClassifier(alpha=0.0001, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500, momentum=0.7, random_state=seed)
    MLP_model.fit(X_train,Y_train)

    # Making Predictions on the Test Set
    y_pred = MLP_model.predict(X_test)

    # Get the classification accuracy on the test set

    st.write(""" ## Model Results: """)
    st.write(sum(Y_test ==  y_pred) / len(y_pred))

# Creating a confusion matrix for MLP Classifier

import plotly.figure_factory as ff
# st.cache
z = confusion_matrix(Y_test, y_pred)[::-1]
# z = confusion_matrix( y_pred, KNN_Y_test)

x = ['Positive', 'Negative', 'Neutral']
y = x[::-1].copy()

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in z]

# set up figure
fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale=color_pallete)

# add title
fig.update_layout(title_text='<b>Confusion matrix</b>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))

# add colorbar


# fig.update_layout(height=500, width=700)
fig['data'][0]['showscale'] = True
st.plotly_chart(fig)






# Create scree plot:
# https://www.datasklr.com/principal-component-analysis-and-factor-analysis/principal-component-analysis



st.write("""
## Next Steps

We have a model that performs relatively well on the test set; however, there is more to improve the model. Some of the ways the model could be improved are:

- Add more representative samples of those with Memory Loss to the data set
- Use Convolutional Neural Network Rather than MLP to possibly improve Model accuracy
- Create a data pipeline for uploading new Audio data into the model quickly
- Shuffle videos during the labeling process to remove possible bias

Mentia is planning on using our S.E.R. algorithm as part of their dialog model to be implemented into Julie. It is Mentia's plan to incorporate speech to text, computer vision, and our S.E.R. algorithm into Julie.

## Conclusion

During this project we got gained valuable experience implementing the full Machine Learning Life Cycle. We started with a problem: can we predict positive, negative and neutral emotions for patients with memory loss. We collected, labeled audio samples and extracted MFCC, Chroma, and MFS features from these samples. We iterated through multiple Random Forrest, KNN and MLP models until we achieved our best model the MLP Neural Network with 87.5% accuracy.

With the model we built, Mentia can implement S.E.R. into their dialog model for Julie inside DevaWorld, creating a more engaging digital experience.


"""
)

st.write(
"""
## References:
- [1] https://acl.gov/aging-and-disability-in-america/data-and-research/projected-future-growth-older-population
- [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6195406/
- [3] https://www.sciencedirect.com/science/article/abs/pii/S1214021X15300387
- [4] https://librosa.org/doc/
- [5] https://en.wikipedia.org/wiki/Chroma_feature
- [6] https://www.kaggle.com/shivamburnwal/speech-emotion-recognition

"""


)


# st.write(matrix.head())

# st.write("""# Mentia SER Algorithm
# ## Use the button below to record an audio clip.
# """)



# https://pretagteam.com/question/how-do-you-record-users-audio-in-streamlit-shearing
# from IPython.display import Audio
# from ipywebrtc import CameraStream, AudioRecorder
#
#
#
# if st.button("Record"):
#     with st.spinner("Recording Your Audio..."):
#
#         camera = CameraStream(constraints = {
#               'audio': True,
#               'video': False
#            },
#            mimeType = 'audio/wav')
#         recorder = AudioRecorder(stream = camera)
#         recorder.recording = True
#
#
#
#         duration = 8
#
#
#         fs = 44000
#         path_myrecording = f"./samples/{file_name}.wav"
#         my_recording = Helper.record(duration, fs)
#         Helper.save_record(path_myrecording, my_recording, fs)
#         features = pd.DataFrame(Model.extract_feature(Helper.load_audio(path_myrecording), MM))
#         normalized_features = pd.DataFrame(Helper.norm_input(path_myrecording, MM))
#     recorder.recording = False
#     recorder.save('test.wav')
#     st.success("Done!")
#     st.write(Model.make_prediction(path_myrecording))
#     st.write(features.T)
#     # st.write(normalized_features)
#
#
#     st.audio(Helper.read_audio(path_myrecording))










#
# import noisereduce as nr
#
# if st.button("Record"):
#     with st.spinner("Recording Your Audio..."):
#
#         duration = 4
#
#
#         fs = 44000
#         path_myrecording = f"./samples/{file_name}.wav"
#         my_recording = Helper.record(duration, fs)
#         # my_recording = Helper.record_JS(duration, fs)
#
#         Helper.save_record(path_myrecording, my_recording, fs)
#
#         features = pd.DataFrame(Model.extract_feature(Helper.load_audio(path_myrecording), MM))
#         normalized_features = pd.DataFrame(Helper.norm_input(path_myrecording, MM))
#     st.success("Done!")
#
#     # This is printing out the prediction
#     st.write(Model.make_prediction(path_myrecording))
#
#     # Showing the feature values extracted:
#     st.write(Helper.norm_input(path_myrecording, MM))
#
#     # plotting the wave file
#     fig = Helper.plot_wav_file(path_myrecording)
#     st.pyplot(fig = plt)
#     # This is creating the audio player
#     st.audio(Helper.read_audio(path_myrecording))
#
#     # This is showing the audio plot
#     fig = Helper.plot_features(path_myrecording)
#     st.pyplot(fig = plt)











# st.write((plt.pyplot.hist(matrix["0"])))
# st.write(extract_feature(load_audio("samples/2024038730407998383.wav")))
# st.write(extract_feature(load_audio("/Users/paulfentress/Desktop/Mentia/Joyce_Audio/Joyce_audio-04.wav")))


# left = matrix.loc[:, '0':'13']
# right = matrix.loc['13':]
# tbl = left.join(right)
# st.write(
# tbl.head()




# )




#
#
# st.write(normalize_dataset(matrix.values, MM))
