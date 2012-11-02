import math
import random

INF = float('inf')

def mult_sample(vals):
    vals = list(vals)
    if len(vals) == 1: return vals[0][0]
    x = random.random() * sum(v for _, v in vals)
    for k, v in vals:
        if x < v: return k
        x -= v
    return k

def remove_random(assignments):
    i = random.randrange(0, len(assignments))
    assignment = assignments[i]
    del assignments[i]
    return assignment

class DirichletMultinomial(object):
    def __init__(self, K, alpha_prior):
        self.K = K
        self.alpha_prior = alpha_prior
        alpha_prior.tie(self)
        self.count = [0]*K
        self.N = 0

    @property
    def alpha(self):
        return self.alpha_prior.x

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

    def log_likelihood(self):
        ll = (math.lgamma(self.K * self.alpha) - math.lgamma(self.K * self.alpha + self.N)
                + sum(math.lgamma(self.alpha + self.count[k]) for k in xrange(self.K))
                - self.K * math.lgamma(self.alpha))
        return ll

    def __str__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ Dir({self.alpha})'.format(self=self)

class Uniform(object):
    def __init__(self, K):
        self.K = K
        self.p = 1./K
        self.count = 0

    def increment(self, k):
        self.count += 1

    def decrement(self, k):
        self.count -= 1

    def prob(self, k):
        if k > self.K: return 0
        return self.p

    def log_likelihood(self):
        return self.count * math.log(self.p)

    def __str__(self):
        return 'Uniform(K={self.K}, count={self.count})'.format(self=self)
