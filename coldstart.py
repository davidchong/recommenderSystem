## https://github.com/topspinj/recommender-tutorial/blob/master/part-2-cold-start-problem.ipynb

import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))

genres_counts = Counter(g for genres in movies['genres'] for g in genres)
print(f"There are {len(genres_counts)} genre labels.")

movies = movies[movies['genres']!='(no genres listed)']
del genres_counts['(no genres listed)']

print("The 5 most common genres: \n", genres_counts.most_common(5))

genres_counts_df = pd.DataFrame([genres_counts]).T.reset_index()
genres_counts_df.columns = ['genres', 'count']
genres_counts_df = genres_counts_df.sort_values(by='count', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='genres', y='count', data=genres_counts_df, palette='viridis')
plt.xticks(rotation=90)
plt.show()

def extract_year_from_title(title):
    t = title.split(' ')
    year = None
    if re.search(r'\(\d+\)', t[-1]):
        year = t[-1].strip('()')
        year = int(year)
    return year

movies['year'] = movies['title'].apply(extract_year_from_title)
print(f"Original number of movies: {movies['movieId'].nunique()}")

movies = movies[~movies['year'].isnull()]
print(f"Number of movies after removing null years: {movies['movieId'].nunique()}")

"""
def get_decade(year):
    year = str(year)
    decade_prefix = year[0:3] # get first 3 digits of year
    decade = f'{decade_prefix}0' # append 0 at the end
    return int(decade)
"""

def round_down(year):
    return year - (year%10)

movies['decade'] = movies['year'].apply(round_down)

plt.figure(figsize=(10,6))
sns.countplot(movies['decade'], palette='Blues')
plt.xticks(rotation=90)

genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))

movie_decades = pd.get_dummies(movies['decade'])

movie_features = pd.concat([movies[genres], movie_decades], axis=1)

cosine_sim = cosine_similarity(movie_features, movie_features)
print(f"Dimensions of our movie features cosine similarity matrix: {cosine_sim.shape}")

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

title = movie_finder('juminji')

movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]
"""
n_recommendations=10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
similar_movies = [i[0] for i in sim_scores]

print(f"Because you watched {title}:")
movies['title'].iloc[similar_movies]
"""
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Recommendations for {title}:")
    print(movies['title'].iloc[similar_movies])

get_content_based_recommendations('aladin', 5)