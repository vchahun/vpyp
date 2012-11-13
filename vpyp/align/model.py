import logging
from itertools import izip
from collections import defaultdict
try:
    import numpypy
except ImportError:
    pass
import numpy, math
from ..prob import mult_sample, BetaBernouilli
from ..pyp import PYP
from ..prior import PYPPrior, GammaPrior, stuple

def diagonal_matrix(flen, elen, scale):
    diag = numpy.array([[math.exp(-scale * abs(j/float(elen)-i/float(flen)))
                      for j in xrange(elen)]
                        for i in xrange(flen)])
    return diag / diag.sum(axis=0) # normalize columns

def alignment_matrix(diag, p_null):
    null_row = p_null * numpy.ones((1, diag.shape[1]))
    diag *= (1 - p_null)
    return numpy.concatenate((null_row, diag))

class AlignmentDistribution:
    def __init__(self, scale_prior):
        self.scale_prior = scale_prior
        scale_prior.tie(self)
        self.assignments = defaultdict(list)

    @property
    def scale(self):
        return self.scale_prior.x

    def prob(self, flen, elen):
        return diagonal_matrix(flen, elen, self.scale)

    def increment(self, flen, elen, i, j):
        self.assignments[flen, elen].append((i, j))

    def decrement(self, flen, elen, i, j):
        self.assignments[flen, elen].remove((i, j))

    def log_likelihood(self):
        ll = 0
        for (flen, elen), points in self.assignments.iteritems():
            a_prob = self.prob(flen, elen)
            ll += sum(math.log(a_prob[i-1, j]) for i, j in points if i > 0)
        return ll

    def resample_hyperparemeters(self, n_iter):
        return self.scale_prior.resample(n_iter)

    def __repr__(self):
        return 'AlignmentDistribution(scale ~ {self.scale_prior})'.format(self=self)

class AlignmentModel(object):
    def __init__(self, n_source, t_base):
        """AlignmentModel(n_source, t_base) -> alignment model
        n_source: size of the source vocabulary
        t_base: shared base of the t-table PYPs"""
        self.null = BetaBernouilli(1.0, 1.0) # p(NULL) ~ Beta(1, 1)
        self.a_table = AlignmentDistribution(GammaPrior(1.0, 1.0, 4.0))
        self.t_base = t_base
        self.t_table = [PYP(self.t_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0))
                for _ in xrange(n_source)]

    @property
    def p_null(self):
        return self.null.prob(1)

    def increment(self, f, e):
        a_prob = alignment_matrix(self.a_table.prob(len(f)-1, len(e)), self.p_null)
        for j, ej in enumerate(e):
            i = mult_sample((i, self.t_table[fi].prob(ej) * a_prob[i, j])
                    for i, fi in enumerate(f))
            self.null.increment(i==0)
            self.a_table.increment(len(f)-1, len(e), i, j)
            self.t_table[f[i]].increment(ej)
            yield i

    def decrement(self, f, e, a):
        for j, (ej, i) in enumerate(izip(e, a)):
            self.null.decrement(i==0)
            self.a_table.decrement(len(f)-1, len(e), i, j)
            self.t_table[f[i]].decrement(ej)

    def log_likelihood(self):
        return (sum(t_word.log_likelihood() + t_word.prior.log_likelihood()
                    for t_word in self.t_table)
                + self.t_base.log_likelihood(full=True)
                + self.null.log_likelihood()
                + self.a_table.log_likelihood() + self.a_table.scale_prior.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        ar = stuple((0, 0))
        logging.info('Resampling t-table PYP base hyperparameters')
        ar += self.t_base.resample_hyperparemeters(n_iter)
        logging.info('Resampling t-table PYP hyperparameters')
        for t_word in self.t_table:
            ar += t_word.resample_hyperparemeters(n_iter)
        logging.info('Resampling alignment distribution scale parameter')
        ar += self.a_table.resample_hyperparemeters(n_iter)
        return ar

    def map_estimate(self):
        t_table = [dict((w, t_word.prob(w)) for w in t_word.tables) for t_word in self.t_table]
        return (self.p_null, self.a_table.scale, t_table)

    @staticmethod
    def combine(samples):
        t_tables = [t_table for _, _, t_table in samples]
        def align(f, e):
            a_probs = [alignment_matrix(diagonal_matrix(len(f)-1, len(e), scale), p_null)
                    for p_null, scale, _ in samples]
            for j, ej in enumerate(e):
                _, i = max((sum(t_table[fi].get(ej, 0) * a_prob[i, j]
                                for t_table, a_prob in izip(t_tables, a_probs)), i)
                        for i, fi in enumerate(f))
                yield i
        return align

    def __repr__(self):
        return ('AlignmentModel(#source words={n_source} '
                '| t-table[f] ~ PYP(base={self.t_base})'
                '| a-table ~ {self.a_table} + p(NULL)={self.p_null} ~ {self.null}'
                ).format(self=self, n_source=len(self.t_table))
