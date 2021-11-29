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

st.write(
"""
# Fall 2021 UC Berkeley Data Discovery Project: Building a Speech Emotion Recognition Algorithm For People With Memory Loss
Authors: Paul Fentress, Chi Hoang
## What is the question your project is attempting to answer?
Can we accurately classify the user emotion for people with Memory Loss as positive negative or neutral, In order to see how engaged the user is while playing Devaworld.

## Where does your project fit within the broader conversation/controversy surrounding your topic?
Our project fits into the bigger picture of Memory Loss and assisted living,
because one of the main goals of Devaworld, is to take some of the
workloads off of the overworked and underpaid caregivers while
providing a fun experience for the user, which is more a stimulating
and positive experience than switching on the TV and zoning out.

## What success would look like:

** What are you trying to accomplish? **
We are trying to build a SER prediction algorithm that can accuractely
predict the emotions of people with Memory Loss, and later can be integrated
into the Devaworld app.

** What is the outcome you hope to achieve? **
Acheive 85% + accuracy for our SER model, while creating a useful for Mentia.

## The Data
** Where does it come from? **
We have multiple different data sources:

- Interviews Chi and I conducted at assisted living home (281 Samples, Speakers: 2)
- Various Youtube videos of people with memory loss / older people (178 Samples, Speakers: 5+)
- Tess: SER data set from Kaggle (1527 Samples, Speakers: 2)
- Ravdness: SER data set from Kaggle (763 Samples, Speakers: 24)
- Savee: SER data set from Kaggle (286 Samples, Speakers: 4)

"""
)
# dff = data.groupby("Data_set").count()
# st.write(dff)
# st.write(dff["name"] / sum(dff["name"]) )

fetures_with_lables = pd.read_csv(paths[0]).drop("Unnamed: 0", axis=1).join(labels).join(data)

# hist = plt.hist(fetures_with_lables["0"], bins=50)
# st.pyplot(fig = plt)

st.write(data)

# st.write(fetures_with_lables.groupby(["Data_set", "Labels"], as_index=False).mean()[["Data_set","Labels", "0"]])





st.write("Number of Samples By Emotion")
color_pallete = "Viridis"

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
** What bias might be present in the data? **

- There are 37 total speakers in the training data set, and only 13 different speakers for people eldlerly people or people with Memory Loss.
In order order to improve the model more representative samples should be added to the data set in order to
balance the data.
- 50% of the data comes from the Tess dataset (1527 samples). Therefore there is not a wide range different voices, and there could be
possible overfitting towards the Tess dataset.
- Only one language included: English

** What were some of the other issues with the dataset (missing values, limitations, etc.)? How did you deal with those issues? **

One of the issues we had was collecting a sufficient amount of data to train on. Although we were given many videos to use,
the process of converting the video to audio, splitting the audio into samples, then labeling the samples was cumbersome.
In order to overcome this, we sought out other data sets that were pre-labeled. A limitation of doing this
is that we were including audio samples of people without memory loss, and therefore our dataset, was not perfectly tailored for
this problem. With more time and cold emailing, we could create a more representative data set with more elderly people, and
people with memory loss. Furthermore, a wide range of different people, at different stages of memory loss, with multiple languages would
create the ideal data set.We have a lot of noise in the Devaworld samples, for example, there are often multiple voices in the samples,
or music playing from the Devaworld app. Additionally Julie, the avatar in Devaworld, is speaking at the same time as the people playing Devaworld.
The video quality of the Mentia Videos is relatively low compared to the other samples, therefore there is a lot of white noise in the samples from Mentia.
In order to deal with the white noise, we used a denoiser python package; however, we have not yet dealt with isolating a single voice from a group,
 which means the model could be confused when trying to classify a sample where two people are talking over each other.

## Solution/Model

""")

st.write(

"""** What statistical model did you use? How does it work? **

For pre processing, we used normalization because our audio data does not follow
a normal distrubtion. In order to verify that our data was indeed not normally distrubuted we used a QQ (Quantile Quantile)
plot to verify how our data was distributed. A QQ plot is a way of visually checking the
distrubtion of data.

A QQ plot that has the points scatter straight along the line means that
the sample data does follow the distrubtion you have specified. If your data diverges from the straight line
that means your data does not follow the distribtion you have specified. The math reasoning behind this
is that the QQ plot compares the quantiles of a distrubtion (in our case we are checking for normal) to
our sample data, and sees how correlated they are for all data points.

First you sort the data in ascending order, then compare the quantiles of your sample data
to the normal distrubtion. 
"""


)

import scipy
measurements = matrix.values.reshape(matrix.shape[0]*matrix.shape[1])

fig = scipy.stats.probplot(measurements, dist="norm", plot=plt)

st.pyplot(fig = plt)


st.write("""

We used a Multi Layer Perceptron as our Model which is a type of Neural Network.

""")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
with st.echo():
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
    st.write(sum(Y_test ==  y_pred) / len(y_pred))

# Creating a confusion matrix for MLP Classifier

import plotly.figure_factory as ff

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


st.write(

"""
** Did you try any other models before settling on your final one? **

We Tried 2 other models Including:
- K - Nearest Neighbors
- Random Forrest
""")

st.write("Initial Model: K-Nearest Neighbors")

with st.echo():

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

    # Accuracy:
    st.write(sum(Y_test ==  y_pred) / len(y_pred))

# st.write("KNN Model Accuracy: 0.8585526315789473")


#@title
import plotly.figure_factory as ff

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

st.write(

"""
## How does K-Nearest Neighbors Work?

"""

)




# Create scree plot:
# https://www.datasklr.com/principal-component-analysis-and-factor-analysis/principal-component-analysis

num_vars = 180
features_mean = np.mean(matrix, axis=0)
features_std = np.std(matrix, axis=0)
features_centered = (matrix - features_mean) / features_std

u, s, vt = np.linalg.svd(features_centered, full_matrices=False)

eigvals = s**2 / sum(s**2)  # NOTE (@amoeba): These are not PCA eigenvalues.
                               # This question is about SVD.

sing_vals = np.arange(num_vars) + 1
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xticks(np.arange(0,26, 2))
scree = plt.plot(np.arange(25), eigvals[:25]);
# st.pyplot(fig = plt)


st.write("""
## Impact/Next Steps
** What were your results / what results do you expect? **

Our KNN and MLP models predict the data in the test set with 87% accuracy.

** What decisions will be made as a result of your work? **

Hopefuly the model will be used within Devaworld at some point.
Currently there is further work to be done on the App before implementing the model.

** What work is left to be done? **

Collecting more represtnative data, further testing, and deploying as web app to test from anywhere.

** Will this work be relevant in short/medium/long term? **

If / When the model is in production this will become more clear. The model could
definately be improved by the step above.



"""
)


st.write(matrix.head())

st.write("""# Mentia SER Algorithm
## Use the button below to record an audio clip.
""")



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


import noisereduce as nr

if st.button("Record"):
    with st.spinner("Recording Your Audio..."):

        duration = 4


        fs = 44000
        path_myrecording = f"./samples/{file_name}.wav"
        my_recording = Helper.record(duration, fs)
        # my_recording = Helper.record_JS(duration, fs)

        Helper.save_record(path_myrecording, my_recording, fs)

        features = pd.DataFrame(Model.extract_feature(Helper.load_audio(path_myrecording), MM))
        normalized_features = pd.DataFrame(Helper.norm_input(path_myrecording, MM))
    st.success("Done!")

    # This is printing out the prediction
    st.write(Model.make_prediction(path_myrecording))

    # Showing the feature values extracted:
    st.write(Helper.norm_input(path_myrecording, MM))

    # plotting the wave file
    fig = Helper.plot_wav_file(path_myrecording)
    st.pyplot(fig = plt)
    # This is creating the audio player
    st.audio(Helper.read_audio(path_myrecording))

    # This is showing the audio plot
    fig = Helper.plot_features(path_myrecording)
    st.pyplot(fig = plt)



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
