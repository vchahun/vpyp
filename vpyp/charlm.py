try:
    import numpypy
except ImportError:
    pass
import numpy
import math

try:
    import kenlm
except ImportError:
    pass

class CharLM:
    def __init__(self, path, vocabulary):
        self.lm = kenlm.LanguageModel(path)
        self.vocabulary = vocabulary
        self.K = len(vocabulary)
        self.count = numpy.zeros(self.K)
        self.probs = numpy.zeros(self.K)
        for k in xrange(self.K):
            self.probs[k] = self.get_prob(k)

    def increment(self, k):
        assert (0 <= k < self.K)
        self.count[k] += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.count[k] -= 1

    def get_prob(self, k):
        chars = ' '.join(self.vocabulary[k])
        return 10**self.lm.score(chars)

    def prob(self, k):
        assert (k >= 0)
        if k >= self.K: return self.get_prob(k)
        return self.probs[k]

    def log_likelihood(self, full=False):
        return numpy.log(self.probs).dot(self.count)

    def resample_hyperparemeters(self, n_iter):
        return (0, 0)

    def __getstate__(self):
        return (self.lm.path, self.vocabulary)

    def __setstate__(self, state):
        path, self.vocabulary = state
        self.lm = kenlm.LanguageModel(path)
        self.K = 0

    def __repr__(self):
        return 'CharLM(n={self.lm.order})'.format(self=self)

class PoissonUniformCharLM:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.K = len(vocabulary)
        L = sum(map(len, vocabulary))
        self.length = L/float(self.K) - 1 # Poisson MLE
        self.n_char = len(set(c for w in vocabulary for c in w))
        self.count = numpy.zeros(self.K)
        self.probs = numpy.zeros(self.K)
        for k in xrange(self.K):
            self.probs[k] = self.get_prob(k)

    def increment(self, k):
        assert (0 <= k < self.K)
        self.count[k] += 1

    def decrement(self, k):
        assert (0 <= k < self.K)
        self.count[k] -= 1

    def get_prob(self, k):
        word_length = len(self.vocabulary[k]) # ~ 1 + Poisson(length)
        return math.exp((word_length - 1) * math.log(self.length) # length^w
                - math.lgamma(word_length) - self.length # exp(-length) / w!
                - self.length * math.log(self.n_char)) # (1/nc)^w

    def prob(self, k):
        assert (k >= 0)
        if k >= self.K: return self.get_prob(k)
        return self.probs[k]

    def log_likelihood(self, full=False):
        return numpy.log(self.probs).dot(self.count)

    def resample_hyperparemeters(self, n_iter):
        return (0, 0)

    def __getstate__(self):
        return (self.length, self.n_char, self.vocabulary)

    def __setstate__(self, state):
        self.length, self.n_char, self.vocabulary = state
        self.K = 0

    def __repr__(self):
        return ('PoissonUniformCharLM(length={self.length}, '
                'n_char={self.n_char})').format(self=self)
