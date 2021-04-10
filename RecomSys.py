import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import KFold

cosine_sim = pd.read_csv('data\\cosine_sim.csv')

cosine_sim_map = pd.read_csv('data\\cosine_sim_map.csv', header=None)
cosine_sim_map = cosine_sim_map.set_index(0)
cosine_sim_map = cosine_sim_map[1]

reader = Reader()
ratings = pd.read_csv('data\\ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)
kf = KFold(n_splits=5)
kf.split(data)
svd = SVD()
train_set = data.build_full_trainset()
svd.fit(train_set)

id_map = pd.read_csv('data\\movie_ids.csv')
id_to_title = id_map.set_index('id')
title_to_id = id_map.set_index('title')

smd = pd.read_csv('data\\metadata_small.csv')

def hybrid(user_id, title):
    idx = cosine_sim_map[title]
    sim_scores = list(enumerate(cosine_sim[str(int(idx))]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    print(movie_indices)
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(user_id, id_to_title.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return print(movies.head(10))

hybrid(1, 'The Godfather')