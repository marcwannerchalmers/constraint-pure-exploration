import numpy as np
import matplotlib.pyplot as plt
import os
from constraint_explore import BernoulliBandit

def f(x):
    return (np.sin(13*x)*np.sin(27*x) + 1)/2
    
def randf(x):
    return np.random.binomial(1, f(x))


def randomize_nojax(f):
    def randf(x, c=0):
        return np.random.binomial(1, f(x) + c)
    return randf

def delta0(h: int):
    return 0.5**h
    
def delta1(h: int):
    return 14*0.5**h

def delta2(h: int):
    return 222*0.5**(2*h)

def lip(dx: float):
    return 14*dx

def get_lip(L:float):
    return lambda dx: L*dx

def main():
    n_arms = 10
    x = np.linspace(0, 1, n_arms)
    exp_arms = f(x)
    bandit = BernoulliBandit(expected_rewards=exp_arms)


    best_arm = 1

    plt.plot(x, f(x))
    plt.axvline(x[best_arm])
    plt.show()


if __name__=="__main__":
    main()