import pandas as pd
import numpy as np
import matplotlib
import os
import spotipy

def read_file_into_df(file_path):
    df = pd.read_csv(file_path)
    return df

def drop_table_from_df(df, list_of_column_to_remove):
    df = df.drop(list_of_column_to_remove, axis = 1)
    return df

charts = read_file_into_df(r"C:\Users\lior_\Desktop\Academic Shit\SEMINAR - SOFTWARE ENGINEERING\Datasets Spotify\charts.csv")
tracks_dataset = read_file_into_df(r"C:\Users\lior_\Desktop\Academic Shit\SEMINAR - SOFTWARE ENGINEERING\Datasets Spotify\dataset.csv")

tracks_dataset = drop_table_from_df(tracks_dataset, ["artists", "explicit", "duration_ms", "Unnamed: 0", "album_name", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "time_signature", "track_genre"])

