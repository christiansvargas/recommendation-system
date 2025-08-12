# ML/CSV libraries
import re, difflib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Loads dataset of songs with song IDs, artists, album names, titles, popularity, length, etc.
songs = pd.read_csv("songs.csv").copy()
if "Unnamed: 0" in songs.columns: # Preps data
    songs = songs.drop(columns=["Unnamed: 0"])

# Content features
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "time_signature",
            "duration_ms", "explicit"]

# Ensure numeric dtypes
if songs["explicit"].dtype == bool:
    songs["explicit"] = songs["explicit"].astype(int)
X = songs[FEATURES].astype(float)

# Scale once
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizes text
def _norm(s: str) -> str:
    s = str(s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

# Precompute a pair key for deduping (track_name + artists)
def _pair(row):
    return f"{_norm(row['track_name'])}__{_norm(row['artists'])}"
songs["_pair"] = songs.apply(_pair, axis=1)

# Finds a song in the dataset
def find_song(title, artist=None, album=None, topn=3, threshold=0.45):
    # Normalizes all inputs
    title_n  = _norm(title)
    artist_n = _norm(artist) if artist else ""
    album_n  = _norm(album)  if album  else ""

    # Prefilter by first word of title if possible
    if title_n:
        first = title_n.split()[0]
        candidates = songs[songs["track_name"].str.lower().str.contains(first, na=False)].copy()
        if candidates.empty:
            candidates = songs.copy()
    else:
        candidates = songs.copy()

    def score_row(r):
        t = _norm(r["track_name"])
        a = _norm(r["artists"])
        al= _norm(r["album_name"])
        t_s = difflib.SequenceMatcher(None, title_n,  t).ratio() if title_n  else 0
        a_s = difflib.SequenceMatcher(None, artist_n, a).ratio() if artist_n else 0
        al_s= difflib.SequenceMatcher(None, album_n,  al).ratio() if album_n  else 0
        # Title matters most, then artist, then album
        return 0.6*t_s + 0.3*a_s + 0.1*al_s

    candidates["match_score"] = candidates.apply(score_row, axis=1)
    picks = candidates.sort_values("match_score", ascending=False).head(topn)

    best = picks.iloc[0]
    return best["track_id"] if best["match_score"] >= threshold else ""

# Recommends songs based on user input
def recommend(track_ids, k=5, same_genre=True, popularity_weight=0.15):
    # Stores songs they like
    liked = songs[songs.track_id.isin(track_ids)]
    if liked.empty:
        print("No liked tracks found in dataset.")
        return pd.DataFrame()

    # Builds the user profile
    up_scaled = scaler.transform(liked[FEATURES].astype(float))
    user_profile = up_scaled.mean(axis=0, keepdims=True)

    # Compute similarities
    sims = cosine_similarity(X_scaled, user_profile).ravel()

    # Encourage popularity slightly
    if "popularity" in songs.columns and popularity_weight > 0:
        pop_norm = songs["popularity"].astype(float) / 100.0
        sims = (1 - popularity_weight) * sims + popularity_weight * pop_norm.values

    # Stay in the same genres
    mask = np.ones(len(songs), dtype=bool)
    if same_genre and "track_genre" in songs.columns:
        liked_genres = set(liked["track_genre"].head(10).tolist())
        if liked_genres:
            mask = songs["track_genre"].isin(liked_genres).values

    # Exclude liked and dedupe (track, artist)
    liked_pairs = set(liked["_pair"])
    order = np.argsort(-sims)
    recs = []
    seen = set(liked_pairs)
    for idx in order:
        if not mask[idx]:
            continue
        row = songs.iloc[idx]
        p = row["_pair"]
        if row["track_id"] in track_ids or p in seen:
            continue
        seen.add(p)
        recs.append({"track_name": row["track_name"], "artists": row["artists"], "album_name": row["album_name"], "similarity": float(sims[idx]),
                     "track_id": row["track_id"]})
        if len(recs) == k:
            break

    recs_df = pd.DataFrame(recs)
    if recs_df.empty:
        print("No recommendations found.")

    return recs_df

# Main interface
def main():
    print("Welcome to your music recommender system!") # Welcome message
    track_ids = [] # Stores user inputted songs
    # Repeatedly get user input
    while True:
        title = input("You may press 'Enter' once you're done. Please enter the title of a song you like: ").strip()
        if title == "":
            break
        artist = input("You may press 'Enter' to skip. Otherwise, please enter the artist for the song: ").strip()
        album  = input("You may press 'Enter' to skip. Otherwise, please enter the album that the song is on: ").strip()

        # Finds the song with the matching title, artist and album
        track_id = find_song(title, artist or None, album or None)
        if track_id:
            row = songs.loc[songs.track_id == track_id].iloc[0]
            print(f"Added: {row['track_name']} by {row['artists']} off {row['album_name']}")
            track_ids.append(track_id)
        else:
            print("Error: song not found")

    # Check for no songs
    if not track_ids:
        print("Thanks for using your music recommender system!") # Closing message
        return
    
    # Recommend
    while True:
        recs = recommend(track_ids, k=5, same_genre=True, popularity_weight=0.15)
        if not recs.empty:
            print("\nHere are 5 songs you might like:\n")
            for i, r in recs.iterrows():
                print(f"{i+1}. {r['track_name']} by {r['artists']} off {r['album_name']}")

        check = input("\nWould you like to continue entering songs? (y/n): ").strip().lower()
        if check == "y":
            # loop back to add more likes
            pass
        elif check == "n":
            break
        else:
            print("Error: (y/n): ")
            continue

        # gather more inputs
        while True:
            title = input("You may press 'Enter' once you're done. Please enter the title of a song you like: ").strip()
            if title == "":
                break
            artist = input("You may press 'Enter' to skip. Otherwise, please enter the artist for the song: ").strip()
            album  = input("You may press 'Enter' to skip. Otherwise, please enter the album that the song is on: ").strip()
            track_id = find_song(title, artist or None, album or None)
            if track_id:
                row = songs.loc[songs.track_id == track_id].iloc[0]
                print(f"Added: {row['track_name']} by {row['artists']} off {row['album_name']}")
                track_ids.append(track_id)
            else:
                print("Error: song not found")

    print("Thanks for using your music recommender system!") # Closing message

# Start interface
if __name__ == "__main__":
    main()