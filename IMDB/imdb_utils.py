from .imdb_env import IMDBEnvironment

def get_env(n_actions, spec, seed=None):
    '''
    Returns an environment of n movies and constraint based on the specification
    '''
    f_reviews = 'IMDB/imdb/user_reviews_uncensored.csv'
    f_genres = 'IMDB/imdb/movie_genres.csv'
    env = IMDBEnvironment(f_reviews, f_genres, n_actions=n_actions, std_mul=1., seed=seed)
    A, c = env.get_constraints(spec)
    return env, A, c

