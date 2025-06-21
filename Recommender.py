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

