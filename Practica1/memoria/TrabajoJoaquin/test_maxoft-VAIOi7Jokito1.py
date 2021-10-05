import numpy as np
import scipy.stats as scist

t = 4

U = np.loadtxt('randomseq.txt')
U = U[:(len(U) // t)*t]  # Guarantee size of U is multiple of t
clusters = np.split(U, t)
V = np.amax(clusters, axis=0)

print(scist.kstest(V, lambda x: x ** t))
print(scist.kstest(U, lambda x: x))