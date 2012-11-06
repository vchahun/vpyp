import math
import random
try:
    import numpypy
except ImportError:
    pass
import numpy

# Utility functions

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

# Distributions with priors 

class DirichletMultinomial(object):
    def __init__(self, K, alpha_prior):
        self.K = K
        self.alpha_prior = alpha_prior
        alpha_prior.tie(self)
        self.count = numpy.zeros(K)
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
        if k >= self.K: return 0
        return (self.alpha + self.count[k])/(self.K * self.alpha + self.N)

    def log_likelihood(self):
        return (math.lgamma(self.K * self.alpha) - math.lgamma(self.K * self.alpha + self.N)
                + sum(math.lgamma(self.alpha + self.count[k]) for k in xrange(self.K))
                - self.K * math.lgamma(self.alpha))

    def __repr__(self):
        return 'Multinomial(K={self.K}, N={self.N}) ~ Dir({self.alpha})'.format(self=self)

class Uniform(object):
    def __init__(self, K):
        self.K = K
        self.count = 0

    def increment(self, k):
        self.count += 1

    def decrement(self, k):
        self.count -= 1

    def prob(self, k):
        if k >= self.K: return 0
        return 1./self.K

    def log_likelihood(self):
        return - self.count * math.log(self.K)

    def __repr__(self):
        return 'Uniform(K={self.K}, count={self.count})'.format(self=self)

class BetaBernouilli(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.positive = 0.
        self.total = 0.

    @property
    def p(self):
        return (self.alpha + self.positive)/(self.alpha + self.beta + self.total)

    def increment(self, k):
        self.total += 1
        self.positive += k

    def decrement(self, k):
        self.total -= 1
        self.positive -= k

    def prob(self, k):
        return self.p if k else (1 - self.p)

    def log_likelihood(self):
        return (math.lgamma(self.alpha + self.beta)
                - math.lgamma(self.alpha) - math.lgamma(self.beta)
                + math.lgamma(self.positive + self.alpha)
                + math.lgamma(self.total - self.positive + self.beta)
                - math.lgamma(self.alpha + self.beta + self.total))

    def __repr__(self):
        return ('Bernouilli(positive={self.positive}, total={self.total}) '
                '~ Beta({self.alpha}, {self.beta})').format(self=self)
