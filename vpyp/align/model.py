import logging
from itertools import izip
try:
    import numpypy
except ImportError:
    pass
import numpy, math
from ..prob import mult_sample, Uniform, BetaBernouilli
from ..pyp import PYP
from ..prior import PYPPrior

def alignment_matrix(flen, elen, scale, p_null):
    mx = numpy.array([[math.exp(-scale * abs(j/float(elen)-i/float(flen)))
                      for j in xrange(elen)]
                        for i in xrange(flen)])
    null_row = p_null * numpy.ones((1, elen))
    mx *= (1 - p_null) / mx.sum(axis=0) # normalize columns
    return numpy.concatenate((null_row, mx))

class AlignmentModel:
    def __init__(self, n_source, n_target):
        self.null = BetaBernouilli(1, 1)
        self.scale = 4.0 # scale_prior = GammaPrior(1, 1, 4.0)
        self.t_base = Uniform(n_target)
        self.t_prior = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0) # d, theta = 0.1, 1
        self.t_table = [PYP(self.t_base, self.t_prior) for _ in xrange(n_source)]

    def increment(self, f, e):
        a_prob = alignment_matrix(len(f)-1, len(e), self.scale, self.null.prob(1))
        for j, ej in enumerate(e):
            i = mult_sample((i, self.t_table[fi].prob(ej) * a_prob[i, j])
                    for i, fi in enumerate(f))
            self.null.increment(i==0)
            self.t_table[f[i]].increment(ej)
            yield i

    def decrement(self, f, e, a):
        for ej, i in izip(e, a):
            self.null.decrement(i==0)
            self.t_table[f[i]].decrement(ej)

    def log_likelihood(self):
        return (sum(t_word.log_likelihood() for t_word in self.t_table)
                + self.t_base.log_likelihood() + self.t_prior.log_likelihood()
                + self.null.log_likelihood()) # + self.scale_prior.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling t-table PYP hyperparameters')
        a1, r1 = self.t_prior.resample(n_iter)
        """
        logging.info('Resampling alignment distribution scale parameter')
        a2, r2 = self.scale_prior.resample(n_iter)
        return (a1+a2, r1+r2)
        """
        return (a1, r1)

    def map_estimate(self):
        t_table = [dict((w, t_word.prob(w)) for w in t_word.tables) for t_word in self.t_table]
        return (self.null.prob(1), self.scale, t_table)

    @staticmethod
    def combine(samples):
        t_tables = [t_table for _, _, t_table in samples]
        def align(f, e):
            a_probs = [alignment_matrix(len(f)-1, len(e), scale, p_null) 
                    for p_null, scale, _ in samples]
            for j, ej in enumerate(e):
                _, i = max((sum(t_table[fi].get(ej, 0) * a_prob[i, j]
                                for t_table, a_prob in izip(t_tables, a_probs)), i)
                        for i, fi in enumerate(f))
                yield i
        return align

    def __repr__(self):
        return ('AlignmentModel(#source words={t_table_size} '
                '| t-table ~ PYP(base={self.t_base}, prior={self.t_prior}) '
                '| p(NULL) ~ {self.null}; scale = {self.scale})'
                #'scale ~ {self.scale_prior})'
                ).format(self=self, t_table_size=len(self.t_table))
