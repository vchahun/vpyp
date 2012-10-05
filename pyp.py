import math
import random
from prob import mult_sample

class CRP(object):
    def __init__(self):
        self.tables = {}
        self.ntables = 0
        self.ncustomers = {}
        self.total_customers = 0

    def _seat_to(self, k, i):
        if not k in self.tables: # add new dish
            self.tables[k] = []
            self.ncustomers[k] = 0
        self.ncustomers[k] += 1
        self.total_customers += 1
        tables = self.tables[k]
        if i == -1: # add new table
            self.ntables += 1
            tables.append(1)
        else: # existing table
            tables[i] += 1
        return (i == -1)

    def _unseat_from(self, k, i):
        self.ncustomers[k] -= 1
        self.total_customers -= 1
        tables = self.tables[k]
        for t in xrange(len(tables)):
            if i < tables[t]:
                tables[t] -= 1
                if tables[t] == 0: # cleanup empty table
                    del tables[t]
                    self.ntables -= 1
                    if len(tables) == 0: # cleanup dish
                        del self.tables[k]
                        del self.ncustomers[k]
                    return True
                return False
            i -= tables[t]

class PYP(CRP):
    def __init__(self, theta, d, base):
        super(PYP, self).__init__()
        self.theta = theta
        self.d = d
        self.base = base

    def _dish_tables(self, k): # all the tables labeled with dish k
        if k in self.tables:
            # existing tables
            for i, n in enumerate(self.tables[k]):
                yield i, n-self.d
            # new table
            yield -1, (self.theta + self.d * self.ntables) * self.base.prob(k)
        else:
            yield -1, 1

    def increment(self, k):
        i = mult_sample(self._dish_tables(k))
        if self._seat_to(k, i):
            self.base.increment(k)

    def decrement(self, k):
        i = random.randint(0, self.ncustomers[k]-1)
        if self._unseat_from(k, i):
            self.base.decrement(k)
    
    def prob(self, k): # total prob for dish k
        # new table
        w = (self.theta + self.d * self.ntables) * self.base.prob(k)
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return w / (self.theta + self.total_customers)

    def log_likelihood(self):
        # XXX this is the leave-one-out log likelihood, not log p(X|d,t) 
        return sum(count * math.log(self.prob(k))
                for k, count in self.ncustomers.iteritems())

    def __str__(self):
        return 'PYP(d={self.d}, theta={self.theta}, #customers={self.total_customers}, #tables={self.ntables}, #dishes={V}, Base={self.base})'.format(self=self, V=len(self.tables))
