# Recommendation System

Hi, my name is **Christian**, and this project is for a personal music recommendation system. It works by using machine learning (ML) to figure out what to recommend to the user based on their current interests in music. The reason for this project was that I felt like the current music recommendations that apps like Spotify use often give inaccurate results for me, so I wanted to make a personalized system to have more music to enjoy. There were many different choices for what recommendation system I would make, but I chose music because I listen to it all the time while working, cleaning and doing many other tasks. Also, I wanted to strengthen my understanding in Python and AI principles since I'll be taking an intro to AI course in the future.

## Frameworks and Technologies
- **Programming Languages:** Python (3.9+)
- **Libraries:** NumPy, pandas, scikit-learn, Matplotlib and Seaborn
- **ML Technique:** Content-based filtering with cosine similarity
- **Data Prep:** StandardScaler feature scaling (boolean to int for explicit)
- **Interface:** Command Line

## Usage
- Use **python Recommender.py** to run
- **Optional:** You may use the script as a library using **from Recommender import find_song, recommend,** for instance.

## Troubleshooting
- "song not found" → Check spelling/diacritics; try without artist/album; lower threshold.
- "liked songs not found" → Your input tracks weren’t in songs.csv.
- "recommendations not found" → Increase k, set same_genre=False, or raise popularity_weight a bit.

## Contact and GitHub
- **Email:** Christian.S.Vargas4@gmail.com
- **GitHub:** [github.com/christiansvargas/recommendation-system](https://github.com/christiansvargas/recommendation-system)

## Other Information
- **Initial Spark:** ["Evaluation Metrics for Recommendation Systems – An Overview"](https://towardsdatascience.com/evaluation-metrics-for-recommendation-systems-an-overview-71290690ecba/)
- **Training Data:** ["Spotify Tracks Dataset"](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download)

## Roadmap
- Add diversity/novelty constraints (avoid over-recommending the same artist/genre)
- Web app UI
- Offline evaluation (Precision@K, MAP, Coverage)
- Spotify API integration to fetch live audio features

## Fun Fact
- The recommender gives popularity only a small nudge (15% by default), so it can still surface lesser-known tracks when they match your taste.