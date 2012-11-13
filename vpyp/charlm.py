try:
    import numpypy
except ImportError:
    pass
import numpy

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
