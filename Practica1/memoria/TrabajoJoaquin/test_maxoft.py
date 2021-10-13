import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import kstest


data = np.loadtxt("randomseq.txt")

for t in range(1, 4):
    U = data[:(len(data)//t)*t]
    C = np.split(U, t)
    V = np.amax(C, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(V,bins=40,density=True)
    plt.title(f'Función de densidad con t={t}')
    plt.savefig(f'densidadt-{t}.png')
    plt.close()

    plt.hist(V, density=True, cumulative=True, label='CDF', bins=40)
    plt.title(f'CDF con t={t}')
    plt.savefig(f'cdf-{t}.png')
    plt.close()

    print(kstest(V, lambda x: x**t))

def congruential_gen():
    current = 559079219
    while True:
        current = (16807 * current) % (2**31 - 1)
        yield current / (2**31 - 1)


data = np.array(list(itertools.islice(congruential_gen(), 0, len(U))))
for t in range(1, 4):
    U = data[:(len(data)//t)*t]
    C = np.split(U, t)
    V = np.amax(C, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(V,bins=40,density=True)
    plt.title(f'Función de densidad con t={t}')
    plt.savefig(f'cong_densidadt-{t}.png')
    plt.close()

    plt.hist(V, density=True, cumulative=True, label='CDF', bins=40)
    plt.title(f'CDF con t={t}')
    plt.savefig(f'cong_cdf-{t}.png')
    plt.close()

    print(kstest(V, lambda x: x**t))