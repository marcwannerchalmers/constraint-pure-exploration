import numpy as np
import pandas as pd

import os
import pickle as pkl


class Environment(object):
    """
    Abstract class specifying which methods environments have and what they return
    """

    def __init__(self, **kwargs):
        """
        Initializes the environment
        """
        pass

    def reset(self):
        """
        Resets the environment and returns an observation
        Returns:
            observation (object)
        """
        raise Exception('Reset not implemented')

    def step(self, action):
        """
        Plays an action, returns a reward and updates or terminates the environment
        Args:
            action: Played action
        Returns:
            observation (object): The observation following the action (None if terminal state reached).
            reward (float): The reward of the submitted action
            done (boolean): True if the environment has reached a terminal state
            info (dict): Returns e.g., the reward distribution of all actions so that regret can be computed
        """
        raise Exception('Stepping/acting not implemented')

    def save(self, path, overwrite=False, make_dirs=False):
        """
        Stores the environment to a binary
        """

        if not overwrite and os.path.isfile(path):
            raise Exception(
                'File at location %s already exists and overwrite is set to False' % path)

        if make_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        pkl.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        """
        Loads the environment from a binary
        """

        return pkl.load(open(path, 'rb'))



class IMDBEnvironment(Environment):
    def __init__(self, review_path=None, genre_path=None, n_actions=10, std_mul=1., seed=None, **kwargs):

        D_reviews = pd.read_csv(review_path, index_col='Unnamed: 0').drop(columns=['User'])
        D_genres = pd.read_csv(genre_path, index_col='Unnamed: 0')

        
        
        d = D_reviews.shape[1]
        env_state = np.random.RandomState(0) # Make sure we always get the same movies and mean ratings
        I = sorted(env_state.choice(d,d,replace=False))
        D_reviews = D_reviews.iloc[:,I]
        
        self.means = D_reviews.mean()[:n_actions]
        self.std = D_reviews.std()[:n_actions]

        self.std_mul = std_mul
        self.random_state = np.random.RandomState(seed)
        
        self.movies = D_reviews.columns[:n_actions].tolist()
        
        D_genres = D_genres[D_genres['movie_title'].isin(self.movies)]
        D_genres = D_genres.rename(lambda x : x.replace('genre_',''), axis='columns')
        self.D_genres = D_genres
        
        self.genres = D_genres.columns[1:].tolist()
        
    def get_means(self):
        return self.means
    
    def get_std(self):
        return np.max(self.std)
        
    def reset(self):
        '''
        Reset environment.
        '''

        return None
    
    
    def step(self, action):
        """
        Plays an action, returns a reward and updates or terminates the environment
        Args:
            action: Played action
        Returns:
            observation (object): The observation following the action (None if terminal state reached).
            reward (float): The reward of the submitted action
            done (boolean): True if the environment has reached a terminal state
            info (dict): Returns e.g., the reward distribution of all actions so that regret can be computed
        """

        regret = self.simple_regret(action)

        reward = self.random_state.normal(self.means[action], self.std_mul*self.std[action])

        info = {'regret': regret,
                'expected_reward': self.means[action]}

        return reward, reward, False, info


    def simple_regret(self, action):
        return np.max(self.means) - self.means[action]
    
    def get_constraints(self, spec):
        ''' 
        Returns a constraint set based on requested fractions of genres
        '''

        k = len(self.movies)

        B = np.zeros((len(spec), k))
        c = np.zeros(len(spec))
        j=0
        for g, v in spec.items():
            d,t = v

            I = np.where(self.D_genres[g]>0)[0]
            B[j,I] = 1

            if(len(I)<1 and d=='>' and t>0):
                raise Exception('Requested a non-zero fraction of \'%s\', but there are no included movies of that genre.' % g)


            if d=='<':
                c[j] = t
            elif d=='>':
                B[j,I] = -B[j,I]
                c[j] = -t
            else:
                raise Exception('Unknown constraint type')
            j += 1
        return B, c