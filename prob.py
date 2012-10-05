import random
from collections import Counter

INF = float('inf')

def mult_sample(vals):
    vals = list(vals)
    if len(vals) == 1: return vals[0][0]
    x = random.random() * sum(v for _, v in vals)
    for k, v in vals:
        if x < v: return k
        x -= v

def remove_random(assignments):
    i = random.randrange(0, len(assignments))
    assignment = assignments[i]
    del assignments[i]
    return assignment

class DirichletMultinomial:
    def __init__(self, K, alpha):
        self.K = K
        self.alpha = alpha
        self.count = Counter()
        self.N = 0

    def increment(self, k):
        assert (0 <= k < self.K)
        self.count[k] += 1
        self.N += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.count[k] -= 1
        self.N -= 1

    def prob(self, k):
        assert k >= 0
        if k > self.K: return 0
        return (self.alpha + self.count[k])/(self.K * self.alpha + self.N)

    def __str__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ Dir({self.alpha})'.format(self=self)

class Uniform:
    def __init__(self, N):
        self.N = N
        self.p = 1/float(N)

    def increment(self, k): pass

    def decrement(self, k): pass

    def prob(self, k):
        if k > self.N: return 0
        return self.p

    def __str__(self):
        return 'Uniform(N={self.N})'.format(self=self)
