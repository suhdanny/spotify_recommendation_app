import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def find_song(track, artist, sp_api):

    song_data = defaultdict()

    results = sp_api.search(q='track: {} artist: {}'.format(track,
                                                            artist), limit=1)
    if not results['tracks']['items']:
        return None

    results = results['tracks']['items'][0]

    track_id = results['id']
    audio_features = sp_api.audio_features(track_id)[0]
    song_data['name'] = [track]
    song_data['artist'] = [artist]
    song_data['year'] = int(results["album"]["release_date"].split('-')[0])
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_playlist_info(uri, sp_api):

    playlist_URI = uri.split("/")[-1].split("?")[0]

    info = []

    for track in sp_api.playlist_tracks(playlist_URI)["items"]:
        track_name = track["track"]["name"]
        artist_name = track["track"]["artists"][0]["name"]
        info.append((track_name, artist_name))

    return info


features = ['valence', 'acousticness', 'year', 'danceability', 'duration_ms',
            'energy', 'explicit','instrumentalness', 'key', 'liveness',
            'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_mean_vector(uri, sp_api):
    playlist_data = get_playlist_info(uri, sp_api)

    song_vectors = []

    for track, artist in playlist_data:
        song_data = find_song(track, artist, sp_api)

        if song_data is None:
            continue

        song_vector = song_data[features].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def get_titles_in_playlist(uri, sp_api):
    tracks = []
    temp = get_playlist_info(uri, sp_api)
    for track, artist in temp:
        tracks.append(track)
    return tracks


song_data = pd.read_csv("spotify_app/data.csv")


def recommend_songs(sp_api, uri, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    tracks = get_titles_in_playlist(uri, sp_api)
    song_center = get_mean_vector(uri, sp_api)
    X = song_data.select_dtypes(np.number)
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(spotify_data[features])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(tracks)]
    return rec_songs[metadata_cols].to_dict(orient='records')
