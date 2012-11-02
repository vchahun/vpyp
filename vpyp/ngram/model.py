import logging
from ..pyp import PYP
from ..prior import BetaGammaPrior

class BackoffBase:
    def __init__(self, backoff, ctx):
        self.backoff = backoff
        self.ctx = ctx

    def increment(self, k):
        return self.backoff.increment(self.ctx, k)

    def decrement(self, k):
        return self.backoff.decrement(self.ctx, k)

    def prob(self, k):
        return self.backoff.prob(self.ctx, k)

class PYPLM:
    def __init__(self, order, initial_base):
        self.alpha = BetaGammaPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.8) # d, theta = 0.8, 1
        self.order = order
        self.backoff = initial_base if order == 1 else PYPLM(order-1, initial_base)
        self.models = {}

    def __getitem__(self, ctx):
        """ create a new PYP if the context has not been seen """
        if ctx not in self.models:
            base = (self.backoff if self.order == 1 else BackoffBase(self.backoff, ctx[1:]))
            return PYP(base, self.alpha)
        return self.models[ctx]

    def increment(self, ctx, w):
        if ctx not in self.models:
            self.models[ctx] = self[ctx]
        self.models[ctx].increment(w)

    def decrement(self, ctx, w):
        self.models[ctx].decrement(w)

    def prob(self, ctx, w):
        return self[ctx].prob(w)

    def log_likelihood(self):
        return (sum(m.log_likelihood() for m in self.models.itervalues())
                + self.alpha.log_likelihood()
                + self.backoff.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling level %d hyperparameters', self.order)
        a1, r1 = self.alpha.resample(n_iter)
        if self.order == 1:
            return (a1, r1)
        else:
            a2, r2 = self.backoff.resample_hyperparemeters(n_iter)
            return (a1+a2, r1+r2)

    def __repr__(self):
        return ('PYPLM(order={self.order}, #ctx={C}, alpha={self.alpha}, '
                'backoff={self.backoff})').format(self=self, C=len(self.models))
