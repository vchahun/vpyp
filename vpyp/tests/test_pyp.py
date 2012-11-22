import math
import logging
try:
    import numpypy
except ImportError:
    pass
import numpy
from nose.tools import assert_almost_equals as aeq_
from ..prob import mult_sample, Uniform
from ..pyp import CRP, PYP
from ..prior import PYPPrior

class PYPGenerator(CRP):
    def __init__(self, discount, strength, base, K):
        super(PYPGenerator, self).__init__()
        self.discount = discount
        self.strength = strength
        self.base = base
        self.K = K

    def _dish_prob(self, k):
        c = (self.strength + self.discount * self.ntables) * self.base.prob(k)
        if k in self.tables:
            c += sum(self.tables[k]) - self.discount * len(self.tables[k])
        return c/float(self.total_customers + self.strength)

    def _table_probs(self, k):
        yield -1, (self.strength + self.discount * self.ntables) * self.base.prob(k)
        for i, n in enumerate(self.tables.get(k, [])):
            yield i, n - self.discount

    def observation(self):
        dish_p = [self._dish_prob(k) for k in xrange(K)]
        aeq_(sum(dish_p), 1)
        k = mult_sample(enumerate(dish_p))
        table_p = list(self._table_probs(k))
        i = mult_sample(table_p)
        if self._seat_to(k, i):
            self.base.increment(k)
        return k

    def __repr__(self):
        return ('PYP(d={self.discount}, theta={self.strength}, '
                '#customers={self.total_customers}, #tables={self.ntables}, '
                '#dishes={self.K}, base={self.base})').format(self=self)

def run_sampler(data, model, n_iter, mh_iter):
    for it in xrange(n_iter):
        for k in data:
            if it > 0: model.decrement(k)
            model.increment(k)
        if it % 30 == 29:
            logging.info('Iteration %d/%d', it+1, n_iter)
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / len(data))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
            logging.info('Resampling hyperparameters...')
            model.resample_hyperparemeters(mh_iter)

K = 100
N = 10000
d, theta = 0.9, 0.1
n_iter = 1000
mh_iter = 100

dist_str = lambda v: '|'.join(str(int(math.log(p))) for p in v)

def test_pyp():
    gen = PYPGenerator(d, theta, Uniform(K), K)
    logging.info('Generating %d PYP observations', N)
    observations = numpy.array([gen.observation() for _ in range(N)])
    logging.info('Final generator configuration: %s', gen)
    odist = numpy.array([gen._dish_prob(k) for k in xrange(K)])
    logging.info('Original distribution: %s', dist_str(odist))
    logging.info('Learning PYP model')
    prior = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 100.0)
    model = PYP(Uniform(K), prior)
    run_sampler(observations, model, n_iter, mh_iter)
    fdist = numpy.array([model.prob(k) for k in xrange(K)])
    logging.info('Learned distribution: %s', dist_str(fdist))
    logging.info('L1 diff: %s', numpy.abs(odist-fdist).sum())
    logging.info('KL div: %s', (odist*numpy.log(odist/fdist)).sum())

if __name__ == '__main__':
    import random
    random.seed(4498234908329048320948203984)
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    logging.info('Test PYP')
    test_pyp()
    logging.info('-> OK')
