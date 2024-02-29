import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    x = np.linspace(0, 1, 1000)
    #plt.plot(x, np.mean(randf(np.stack([x for _ in range(100000)])), axis=0))
    #plt.show()
    #return
    fig, ax = plt.subplots()
    #ax.plot(x, DOO_example1(x), c="r", linewidth=1)




    plt.show()


if __name__=="__main__":
    main()