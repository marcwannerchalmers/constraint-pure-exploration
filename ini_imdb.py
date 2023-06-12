import numpy as np
from constraint_explore import TnS, CGE, UniformExplorer, GaussianBandit, get_policy, solve_game, compute_neighbors, run_imdb_exp
from IMDB.imdb_utils import get_env
import json
import lzma
import dill as pickle

import time
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

if __name__ == '__main__':
    '''
    Create and run experiments on slurm-cluster. Define run details in python_script.sh
    '''

    restricted = True # Controlls if Scenario 1 or 2. 
    n_movies = 12
    n_arms = n_movies
    exp_name = f'imdb_{n_movies}_{restricted}/'
    os.makedirs(exp_name, exist_ok=True)

    # Setup exp
    delta = 0.1
    spec = {
        'action': ('<', 0.3),
        'drama': ('>', 0.3),
        'family': ('>', 0.3)
    }
    n_seeds = 1000

    imdb, A, b = get_env(n_movies, spec=spec)
    mu = imdb.get_means().values
    sigma = imdb.get_std()
    optimal_policy,aux  = get_policy(mu, A, b)
    neighbors = compute_neighbors(optimal_policy, aux['A'], aux['b'], slack=aux['slack'])

    # get lower bound and optimal allocation
    if restricted:
        # Compute allocation constraint
        simplex = np.ones_like(mu).reshape(1, -1)
        eye = np.eye(len(mu))
        one = np.array([1])
        allocation_A = np.concatenate([A, - eye, simplex, -simplex], axis=0)
        allocation_b = np.concatenate([b,  np.zeros(n_arms), one, -one], axis=0)
    else: 
        allocation_A = None
        allocation_b = None
    optimal_policy, aux = get_policy(mu, A, b)
    neighbors = compute_neighbors(optimal_policy, aux['A'], aux['b'], slack=aux['slack'])
    optimal_allocation, game_value = solve_game(mu=mu, vertex=optimal_policy, neighbors=neighbors,
                                                 allocation_A=allocation_A, allocation_b=allocation_b, tol=1e-32)
    
    kl_delta = delta * np.log(delta / (1 - delta)) + (1 - delta) * np.log((1 - delta) / delta)
    lower_bound  = 1 / game_value * kl_delta
    
    # Save env
    meta_data = {}
    meta_data['delta'] = delta
    meta_data['n_seeds'] = n_seeds
    meta_data['mu'] = mu
    meta_data['A'] = A
    meta_data['b'] = b
    meta_data['optimal_policy'] = optimal_policy
    meta_data['optimal_allocation'] = optimal_allocation
    meta_data['game_value'] = game_value
    meta_data['lower_bound'] = lower_bound
    meta_data['n_arms'] = n_movies

    sleep = 3 # seconds to sleep between submitting jobs


    with open(exp_name + 'meta_data.json', 'w') as fp:
        json.dump(meta_data, fp, cls=NumpyEncoder, indent=4)
    
    # Run experiments

    # Uniform exp 
    uniform_path = exp_name + 'uniform/'
    os.makedirs(uniform_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f'Running uniform {seed}')
        env, _, _ = get_env(n_movies, spec=spec, seed=seed)
        explorer = UniformExplorer(len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted)
        exp = {
            'bandit': env,
            'explorer': explorer,
            'A': A,
            'b': b
        }
        path = uniform_path + f'seed_{seed}.xz'
        pickle.dump(exp, lzma.open(path, 'wb'))

        os.system(f"sbatch python_script.sh run_exp.py {path}")
    
    time.sleep(sleep) # avoid submitting too many jobs at once

    # Exp with optimal allocation
    optimal_path = exp_name + 'optimal/'
    os.makedirs(optimal_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f'Running optimal {seed}')
        env, _, _ = get_env(n_movies, spec=spec, seed=seed)
        explorer = UniformExplorer(len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted, allocation=optimal_allocation)
        exp = {
            'bandit': env,
            'explorer': explorer,
            'A': A,
            'b': b
        }
        path = optimal_path + f'seed_{seed}.xz'
        pickle.dump(exp, lzma.open(path, 'wb'))

        os.system(f"sbatch python_script.sh run_exp.py {path}")
    
    time.sleep(sleep) # avoid submitting too many jobs at once


    # Exp with TnS
    tns_path = exp_name + 'tns/'
    os.makedirs(tns_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f'Running tns {seed}')
        env, _, _ = get_env(n_movies, spec=spec, seed=seed)
        explorer = TnS(len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted)
        exp = {
            'bandit': env,
            'explorer': explorer,
            'A': A,
            'b': b
        }
        path = tns_path + f'seed_{seed}.xz'
        pickle.dump(exp, lzma.open(path, 'wb'))

        os.system(f"sbatch python_script.sh run_exp.py {path}")
    


    # Exp with CGE
    cge_path = exp_name + 'cge/'
    os.makedirs(cge_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f'Running cge {seed}')
        env, _, _ = get_env(n_movies, spec=spec, seed=seed)
        explorer = CGE(len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted)
        exp = {
            'bandit': env,
            'explorer': explorer,
            'A': A,
            'b': b
        }
        path = cge_path + f'seed_{seed}.xz'
        pickle.dump(exp, lzma.open(path, 'wb'))

        os.system(f"sbatch python_script.sh run_exp.py {path}")
