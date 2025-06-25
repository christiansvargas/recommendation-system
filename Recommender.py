# ML libraries
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore", FutureWarning)

# Loads dataset of songs with song IDs, artists, album names, titles, popularity, length, etc.
songs = pd.read_csv("songs.csv")

# Finds a song in the dataset
def find_song(title, artist, album):
    # Normalize all inputs
    title = title.lower()
    artist = artist.lower()
    album = album.lower()

    # Find matching title and album rows in the dataset
    song = songs[(songs["track_name"].str.lower() == title) &
                 (songs["album_name"].str.lower() == album)]
    if song.empty: # Check if not found
        return ""
    
    # Finds matching artist row in the dataset
    song = song[song["artists"].apply(lambda x: any(a.strip().lower() == artist for a in x.split(";")))]
    if song.empty: # Check if not found
        return ""
    
    # Returns the song ID from the dataset
    return song.iloc[0]["track_id"]

# Recommends songs based on user input
def recommend(track_ids):
    return

# Initial welcome interface
def welcome():
    track_ids = [] # Stores user inputted songs
    # Repeatedly get user input
    title = input("Welcome to your music recommender system! You may press 'Enter' once you're done. Please enter the title of a song you like: ")
    while title != "":
        artist = input("Please enter the artist for the song: ")
        album = input("Please enter the album that the song is on: ")

        # Finds the song with the matching title, artist and album
        track_id = find_song(title, artist, album)
        if track_id != "":
            track_ids.append(track_id)
            print(f"Added {title.title()} by {artist.title()} off {album.title()}")
        else:
            print("Error: song not found")

        title = input("\nYou may press 'Enter' once you're done. Please enter the title of a song you like: ")

    # Recommend if the user entered at least 1 song
    if len(track_ids) > 0:
        recommend(track_ids)

# Start interface
welcome()