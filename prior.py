import math
import random

def beta_pdf(alpha, beta, x):
    return (math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
            + (alpha - 1)  * math.log(x) + (beta - 1) * math.log(1 - x))

def gamma_pdf(shape, scale, x):
    return (- shape * math.log(scale) - math.lgamma(shape) 
            + (shape - 1) * math.log(x) - x/scale)

class SampledPrior(object):
    def tie(self, distributions):
        self.tied_distributions = distributions

    def full_log_likelihood(self):
        return sum(d.log_likelihood() for d in self.tied_distributions) + self.log_likelihood()

    def resample(self, niter):
        stats = [0, 0]
        old_ll = self.full_log_likelihood() # p(x)
        for _ in xrange(niter):
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
    def __init__(self, shape, scale, x):
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

    def proposal_log_likelihood(self, x_from, x_to):
        return gamma_pdf(1, x_from[0], x_to[0])

    def __repr__(self):
        return 'Prior(x = {self.x} ~ Gamma({self.shape}, {self.scale}))'.format(self=self)

class BetaGammaPrior(SampledPrior):
    def __init__(self, x_alpha, x_beta, y_shape, y_scale, x, y):
        self.x_alpha, self.x_beta = x_alpha, x_beta
        self.y_shape, self.y_scale = y_shape, y_scale
        self.x = x
        self.y = y

    def log_likelihood(self):
        return (beta_pdf(self.x_alpha, self.x_beta, self.x) 
                + gamma_pdf(self.y_shape, self.y_scale, self.y))
    
    def get_parameters(self):
        return (self.x, self.y)

    def set_parameters(self, params):
        self.x, self.y = params

    parameters = property(get_parameters, set_parameters)

    def sample_parameters(self):
        self.x = random.betavariate(10, 10*(1-self.x)/self.x) # Mean: x
        self.y = random.gammavariate(1, self.y) # Mean: y

    def proposal_log_likelihood(self, xy_from, xy_to):
        return (beta_pdf(10, 10*(1-xy_from[0])/xy_from[0], xy_to[0])
                + gamma_pdf(1, xy_from[1], xy_to[1]))

    def __repr__(self):
        return ('Prior(x = {self.x} ~ Beta({self.x_alpha}, {self.x_beta}); '
                'y = {self.y} ~ Gamma({self.y_shape}, {self.y_scale}))').format(self=self)
