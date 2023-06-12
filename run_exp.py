import dill as pickle
import lzma
from constraint_explore import run_exploration_experiment

import argparse

'''
Run wrapper for standard experiments on cluster
'''

def get_args():
    parser =argparse.ArgumentParser()
    parser.add_argument('exp', help='path to exp')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with lzma.open(args.exp, 'rb') as f:
        exp = pickle.load(f)

    t, correct, _, _, avg_time = run_exploration_experiment(bandit=exp['bandit'], explorer=exp['explorer'], A=exp['A'], b=exp['b'])

    results = {}
    results['stopping time'] = t
    results['correct'] = correct
    results['average time per sample'] = avg_time
    print(results['stopping time'])
    pickle.dump(results, lzma.open(args.exp, 'wb'))