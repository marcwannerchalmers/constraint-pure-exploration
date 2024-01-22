import numpy as np
from constraint_explore import (
    TnS,
    CGE,
    UniformExplorer,
    GaussianBandit,
    get_policy,
    solve_game,
    compute_neighbors,
)
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


if __name__ == "__main__":
    """
    Create and run experiments on slurm-cluster. Define run details in python_script.sh
    """

    restricted = True  # Scenario 2 (True) or Scenario 1 (False)
    exp_name = f"experiment_{restricted}/"
    os.makedirs(exp_name, exist_ok=True)

    # Setup exp
    delta = 0.1
    A = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]])
    b = np.array([0.5, 0.5])
    mu = np.array([1, 0.5, 0.4, 0.95, 0.8])
    n_arms = len(mu)
    n_seeds = 1000

    # get lower bound and optimal allocation
    if restricted:
        # Compute allocation constraint
        simplex = np.ones_like(mu).reshape(1, -1)
        eye = np.eye(len(mu))
        one = np.array([1])
        allocation_A = np.concatenate([A, -eye, simplex, -simplex], axis=0)
        allocation_b = np.concatenate([b, np.zeros(n_arms), one, -one], axis=0)
    else:
        allocation_A = None
        allocation_b = None
    optimal_policy, aux = get_policy(mu, A, b)
    neighbors = compute_neighbors(
        optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
    )
    optimal_allocation, game_value = solve_game(
        mu=mu,
        vertex=optimal_policy,
        neighbors=neighbors,
        allocation_A=allocation_A,
        allocation_b=allocation_b,
        tol=1e-32,
    )

    kl_delta = delta * np.log(delta / (1 - delta)) + (1 - delta) * np.log(
        (1 - delta) / delta
    )
    lower_bound = 1 / game_value * kl_delta

    # Treat as best-arm identification problem
    bai, aux = get_policy(mu, None, None)
    neighbors = compute_neighbors(bai, aux["A"], aux["b"], slack=aux["slack"])
    bai_allocation, bai_game_value = solve_game(mu=mu, vertex=bai, neighbors=neighbors)
    bai_lower_bound = 1 / bai_game_value * kl_delta

    # Save env
    meta_data = {}
    meta_data["delta"] = delta
    meta_data["n_seeds"] = n_seeds
    meta_data["mu"] = mu
    meta_data["A"] = A
    meta_data["b"] = b
    meta_data["optimal_policy"] = optimal_policy
    meta_data["optimal_allocation"] = optimal_allocation
    meta_data["game_value"] = game_value
    meta_data["lower_bound"] = lower_bound
    meta_data["n_arms"] = n_arms
    meta_data["bai"] = bai
    meta_data["bai_allocation"] = bai_allocation
    meta_data["bai_game_value"] = bai_game_value
    meta_data["bai_lower_bound"] = bai_lower_bound
    meta_data["restricted"] = restricted

    sleep = 3  # seconds to sleep between submitting jobs

    with open(exp_name + "meta_data.json", "w") as fp:
        json.dump(meta_data, fp, cls=NumpyEncoder, indent=4)

    # Run experiments

    # Uniform exp
    uniform_path = exp_name + "uniform/"
    os.makedirs(uniform_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f"Running uniform {seed}")
        env = GaussianBandit(mu, seed=seed)
        explorer = UniformExplorer(
            len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted
        )
        exp = {"bandit": env, "explorer": explorer, "A": A, "b": b}
        path = uniform_path + f"seed_{seed}.xz"
        pickle.dump(exp, lzma.open(path, "wb"))

        os.system(f"sbatch python_script.sh run_exp.py {path}")

    time.sleep(sleep)  # avoid submitting too many jobs at once

    # Exp with optimal allocation
    optimal_path = exp_name + "optimal/"
    os.makedirs(optimal_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f"Running optimal {seed}")
        env = GaussianBandit(mu, seed=seed)
        explorer = UniformExplorer(
            len(mu),
            A=A,
            b=b,
            delta=delta,
            restricted_exploration=restricted,
            allocation=optimal_allocation,
        )
        exp = {"bandit": env, "explorer": explorer, "A": A, "b": b}
        path = optimal_path + f"seed_{seed}.xz"
        pickle.dump(exp, lzma.open(path, "wb"))

        os.system(f"sbatch python_script.sh run_exp.py {path}")

    time.sleep(sleep)  # avoid submitting too many jobs at once

    # Exp with TnS
    tns_path = exp_name + "tns/"
    os.makedirs(tns_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f"Running tns {seed}")
        env = GaussianBandit(mu, seed=seed)
        explorer = TnS(
            len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted
        )
        exp = {"bandit": env, "explorer": explorer, "A": A, "b": b}
        path = tns_path + f"seed_{seed}.xz"
        pickle.dump(exp, lzma.open(path, "wb"))

        os.system(f"sbatch python_script.sh run_exp.py {path}")

    # Exp with CGE
    cge_path = exp_name + "cge/"
    os.makedirs(cge_path, exist_ok=True)
    for seed in range(n_seeds):
        print(f"Running cge {seed}")
        env = GaussianBandit(mu, seed=seed)
        explorer = CGE(
            len(mu), A=A, b=b, delta=delta, restricted_exploration=restricted
        )
        exp = {"bandit": env, "explorer": explorer, "A": A, "b": b}
        path = cge_path + f"seed_{seed}.xz"
        pickle.dump(exp, lzma.open(path, "wb"))

        os.system(f"sbatch python_script.sh run_exp.py {path}")
