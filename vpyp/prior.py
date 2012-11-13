import math
import random

# Probability density functions

def beta_pdf(alpha, beta, x):
    return (math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
            + (alpha - 1)  * math.log(x) + (beta - 1) * math.log(1 - x))

def gamma_pdf(shape, scale, x):
    return (- shape * math.log(scale) - math.lgamma(shape) 
            + (shape - 1) * math.log(x) - x/scale)

# Hyperparameter Priors

class SampledPrior(object):
    def __init__(self):
        self.tied_distributions = []

    def tie(self, distribution):
        self.tied_distributions.append(distribution)

    def full_log_likelihood(self):
        return sum(d.log_likelihood() for d in self.tied_distributions) + self.log_likelihood()

    def resample(self, n_iter):
        stats = [0, 0]
        old_ll = self.full_log_likelihood() # p(x)
        for _ in xrange(n_iter):
            old_parameters = self.parameters # x
            self.sample_parameters() # x -> x*
            new_parameters = self.parameters # x*
            new_ll = self.full_log_likelihood() # p(x*)
            old_lq = self.proposal_log_likelihood(new_parameters, old_parameters) # q(x* -> x)
            new_lq = self.proposal_log_likelihood(old_parameters, new_parameters) # q(x -> x*)
            log_acc = new_ll + old_lq - old_ll - new_lq # p(x*) q(x* -> x) / p(x) q(x -> x*)
            if log_acc > 0 or random.random() < math.exp(log_acc): # accept
                stats[0] += 1
                old_ll = new_ll # update likelihood
            else: # reject
                stats[1] += 1
                self.parameters = old_parameters # revert parameters
        return stats

class GammaPrior(SampledPrior):
    """Prior for parameters with [0, +inf[ range"""
    def __init__(self, shape, scale, x):
        super(GammaPrior, self).__init__()
        self.shape = shape
        self.scale = scale
        self.x = x

    def log_likelihood(self):
        return gamma_pdf(self.shape, self.scale, self.x)
    
    def get_parameters(self):
        return (self.x,)

    def set_parameters(self, params):
        self.x, = params

    parameters = property(get_parameters, set_parameters)

    def sample_parameters(self):
        self.x = random.gammavariate(1, self.x) # Mean: x
        if self.x <= 0:
            self.x = 1e-12

    def proposal_log_likelihood(self, x_from, x_to):
        return gamma_pdf(1, x_from[0], x_to[0])

    def __repr__(self):
        return ('GammaPrior(x={self.x} ~ Gamma({self.shape}, {self.scale}) | '
                'nties={nties})').format(self=self, nties=len(self.tied_distributions))

class BetaPrior(SampledPrior):
    """Prior for parameters with [0, 1] range"""
    def __init__(self, alpha, beta, x):
        super(BetaPrior, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.x = x

    def log_likelihood(self):
        return beta_pdf(self.alpha, self.beta, self.x)
    
    def get_parameters(self):
        return (self.x,)

    def set_parameters(self, params):
        self.x, = params

    parameters = property(get_parameters, set_parameters)

    def sample_parameters(self):
        self.x = random.betavariate(10, 10*(1-self.x)/self.x) # Mean: x
        if self.x <= 0 or self.x >= 1:
            self.x = 0.5

    def proposal_log_likelihood(self, x_from, x_to):
        return beta_pdf(10, 10*(1-x_from[0])/x_from[0], x_to[0])

    def __repr__(self):
        return ('BetaPrior(x={self.x} ~ Beta({self.alpha}, {self.beta}) | '
                'nties={nties})').format(self=self, nties=len(self.tied_distributions))

class PYPPrior(SampledPrior):
    """Prior for PYP parameters (discount: ]0, 1]; strength: [-discount, +inf[)"""
    def __init__(self, x_alpha, x_beta, y_shape, y_scale, discount, strength):
        """x = d; y = theta + d"""
        super(PYPPrior, self).__init__()
        self.x_prior = BetaPrior(x_alpha, x_beta, discount)
        self.y_prior = GammaPrior(y_shape, y_scale, discount + strength)

    @property
    def discount(self):
        return self.x_prior.x

    @property
    def strength(self):
        return self.y_prior.x - self.x_prior.x

    def log_likelihood(self):
        return self.x_prior.log_likelihood() + self.y_prior.log_likelihood()    

    def get_parameters(self):
        return (self.x_prior.x, self.y_prior.x)

    def set_parameters(self, params):
        self.x_prior.x, self.y_prior.x = params

    parameters = property(get_parameters, set_parameters)

    def sample_parameters(self):
        self.x_prior.sample_parameters()
        self.y_prior.sample_parameters()

    def proposal_log_likelihood(self, xy_from, xy_to):
        return (self.x_prior.proposal_log_likelihood((xy_from[0],), (xy_to[0],))
                + self.y_prior.proposal_log_likelihood((xy_from[1],), (xy_to[1],)))

    def __repr__(self):
        return ('PYPPrior(discount={self.discount}, strength={self.strength} | '
                'discount ~ Beta({self.x_prior.alpha}, {self.x_prior.beta}); '
                'strength + discount ~ Gamma({self.y_prior.shape}, {self.y_prior.scale}) | '
                'nties={nties})').format(self=self, nties=len(self.tied_distributions))

import operator

class stuple(tuple):
    def __add__(self, other):
        return self.__class__(map(operator.add, self, other))
