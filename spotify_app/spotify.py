import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from recommendation import *

st.set_page_config(layout="centered")
st.title("Song Recommendation with Spotify")
st.subheader("An App Created by Donghyun Suh")

# importing data
song_data = pd.read_csv("spotify_app/data.csv")
year_data = pd.read_csv("spotify_app/data_by_year.csv")


@st.cache
def df_to_csv(df):
    return df.to_csv().encode("utf-8")


st.download_button(label="Download the dataset as CSV",
                   data=df_to_csv(song_data),
                   file_name="song.csv")

st.write("Try visualizing different metadata features of songs to your"
         " taste! For example, if you choose 'tempo' as your feature, you can"
         " observe that music has gotten significantly faster over the past"
         " years.")

# Create a Line plot based on user-input
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'liveness', 'valence', 'tempo', 'loudness']

with st.form('user_input_1'):
    selected_feature = st.selectbox('Please select one of the features:',
                                    options=sound_features)
    st.form_submit_button()

line_title = selected_feature[0].upper() + selected_feature[1:]
fig1 = px.line(year_data, x="year", y=selected_feature,
               title="{} trend of songs by year".format(line_title))
st.plotly_chart(fig1)

# K-Means Clustering for Songs

st.write("Below is the visualization of over 170k songs categorized by"
         " using K-means algorithm, where you can select K to visualize"
         " how well the songs are categorized based on different k values."
         " Note that I have used a dimensionality reduction technique known"
         " as PCA to visualize the clusters in 2-dimension.")

with st.form('user_input_2'):
    k = st.selectbox("Please select the desired K for K-Means Clustering",
                     options=[i for i in range(15, 30)])
    st.form_submit_button()


@st.cache()
def perform_pca(num_cluster):
    song_pipeline = Pipeline([('scaler', StandardScaler()),
                              ('kmeans', KMeans(n_clusters=k))])
    X = song_data.select_dtypes(np.number)
    y = song_pipeline.fit_predict(X)
    song_data['cluster'] = y

    pca_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = song_data['name']
    projection['cluster'] = song_data['cluster']

    return projection


fig2 = px.scatter(perform_pca(k), x="x", y="y", color="cluster",
                  hover_data=["x", "y", "title"],
                  title="K-Means Clustering of Songs with k = {}".format(k))
st.plotly_chart(fig2)

# Recommend Songs by playlist
cid = "18e1aae7d28d45f0ac9a16a3355fd23f"
secret = "0e460b0897a44e08b9684dee2575cc9a"


@st.cache(allow_output_mutation=True)
def access_api(client_key, secret_key):
    manager = SpotifyClientCredentials \
        (client_id=client_key, client_secret=secret_key)
    stf = spotipy.Spotify(client_credentials_manager=manager)
    return stf


sp = access_api(cid, secret)

# Song Recommendation
st.write("Now the final step! Enter the url for your spotify playlist and press"
         " submit for recommendations! You can get recommended as many songs"
         " as you like up to 20 songs. Here is a sample link if you don't"
         " have a spotify account: https://open.spotify.com/playlist/0zWlLYUrChgnDRC6DEsWBl?si=e41bf9b1fc53406f")

with st.form('user_input_3'):
    spotify_uri = st.text_input("Please enter your sharable\
                                 link to spotify playlist")
    n = st.number_input("Please type the number of songs you would like to be\
                       recommended", min_value=1, max_value=15)
    st.form_submit_button()

try:
    result = recommend_songs(sp, spotify_uri, song_data, n)
    st.write(result)
except:
    st.write("Playlist link not inserted.")
