import numpy as np
import matplotlib.pyplot as plt
import os
from constraint_explore import BernoulliBandit, CGE, run_exploration_experiment

def f(x):
    return (np.sin(13*x)*np.sin(27*x) + 1)/2
    

def lip(x1, x2):
    return 14*np.abs(x1 - x2)

def constraint_matrices(x, lip_fun):
    m = len(x)
    n = m*(m - 1)
    A = np.zeros((n,m))
    b = np.zeros((n,))
    k = 0
    lambda_mat = np.zeros((m,m))
    for i,j in np.ndindex((m, m)):
        if i==j:
            continue
        A[k,i] = 1
        A[k,j] = -1
        b[k] = lip_fun(x[i], x[j])
        lambda_mat[i,j] = lip_fun(x[i], x[j])
        k += 1

    return A, b, lambda_mat

def kl_bernoulli(mu, lam):
    return mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))

def main():
    n_arms = 10
    x = np.linspace(0, 1, n_arms)
    exp_arms = f(x)
    bandit = BernoulliBandit(expected_rewards=exp_arms)
    A, b, lambda_mat = constraint_matrices(x, lip)
    explorer = CGE(n_arms, A, b, 0.01, dist_type="Bernoulli", lambda_mat=lambda_mat)
    result = run_exploration_experiment(bandit, explorer)
    print(result)
    best_arm = 1
    plt.plot(x, f(x))
    plt.axvline(x[best_arm])
    plt.show()


if __name__=="__main__":
    main()