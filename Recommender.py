# ML/CSV libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter("ignore", FutureWarning)

# Loads dataset of songs with song IDs, artists, album names, titles, popularity, length, etc.
songs = pd.read_csv("songs.csv")

# Stores user inputted songs
track_ids = []

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
    # Extract numerical features of interest to find similarities with other songs
    numerics = ["popularity", "danceability", "duration_ms", "explicit", "key", "mode", "time_signature", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo"]
    X_numeric = songs[numerics].astype(float)

    # Scale the data to a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Builds the user profile
    liked = songs[songs.track_id.isin(track_ids)]
    up_numeric = liked[numerics].astype(float)
    up_scaled  = scaler.transform(up_numeric)
    user_profile = up_scaled.mean(axis=0).reshape(1, -1)

    # Compute similarities
    sims = cosine_similarity(X_scaled, user_profile).ravel()

    # Pick the top 5 similar songs
    songs["sim"] = sims
    recs = (songs.loc[~songs.track_id.isin(track_ids)].nlargest(5, "sim"))
    top_5 = recs[["track_name", "artists", "album_name", "sim"]]

    # Show top 5
    print("Here are 5 songs you might like:\n")
    for i, row in top_5.reset_index(drop=True).iterrows():
        print(f"{i + 1}. {row['track_name']} by {row['artists']} off {row['album_name']}")

    # Delete the similarity score column
    del songs["sim"]

    # Check to continue inputting and recommending
    check = input("\nWould you like to continue entering songs? (y/n): ").strip().lower()
    while check != "y" and check != "n": # Continue prompting if anything else is typed
        check = input("Error: (y/n): ").strip().lower()

    return (check == "y")

# Main interface
def main():
    # Repeatedly get user input
    title = input("You may press 'Enter' once you're done. Please enter the title of a song you like: ")
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
    if not track_ids:
        return

    # Continue recommending if needed
    if recommend(track_ids):
        main()

# Welcome message
print("Welcome to your music recommender system!")

# Start interface
main()

# Closing message
print("Thanks for using your music recommender system!")