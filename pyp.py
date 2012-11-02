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
        tables[i] -= 1
        if tables[i] == 0: # cleanup empty table
            del tables[i]
            self.ntables -= 1
            if len(tables) == 0: # cleanup dish
                del self.tables[k]
                del self.ncustomers[k]
            return True
        return False

class PYP(CRP):
    def __init__(self, base, d_theta_prior):
        super(PYP, self).__init__()
        self.base = base
        self.d_theta_prior = d_theta_prior
        #self._ll = 0

    @property
    def d(self):
        # x = d
        return self.d_theta_prior.x

    @property
    def theta(self):
        # y = theta + d
        return self.d_theta_prior.y - self.d_theta_prior.x

    def _dish_tables(self, k): # all the tables labeled with dish k
        if k in self.tables:
            # existing tables
            for i, n in enumerate(self.tables[k]):
                yield i, n-self.d
            # new table
            yield -1, (self.theta + self.d * self.ntables) * self.base.prob(k)
        else:
            yield -1, 1

    def _customer_table(self, k, n): # find table index of nth customer with dish k
        tables = self.tables[k]
        for i in xrange(len(tables)):
            if n < tables[i]: return i
            n -= tables[i]

    def increment(self, k):
        i = mult_sample(self._dish_tables(k))
        """
        if i == -1:
            self._ll += math.log((self.theta + self.d * self.ntables) 
                    / (self.theta + self.total_customers) * self.base.prob(k))
        else:
            self._ll += math.log((self.tables[k][i] - self.d) / (self.theta + self.total_customers))
        """
        if self._seat_to(k, i):
            self.base.increment(k)

    def decrement(self, k):
        i = self._customer_table(k, random.randint(0, self.ncustomers[k]-1))
        if self._unseat_from(k, i):
            self.base.decrement(k)
        """
            self._ll -= math.log((self.theta + self.d * self.ntables) 
                    / (self.theta + self.total_customers) * self.base.prob(k))
        else:
            self._ll -= math.log((self.tables[k][i] - self.d)
                    / (self.theta + self.total_customers))
        """
    
    def prob(self, k): # total prob for dish k
        # new table
        w = (self.theta + self.d * self.ntables) * self.base.prob(k)
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return w / (self.theta + self.total_customers)

    def pseudo_log_likelihood(self):
        return sum(count * math.log(self.prob(k)) for k, count in self.ncustomers.iteritems())

    def log_likelihood(self):
        ll = (math.lgamma(self.theta) - math.lgamma(self.theta + self.total_customers)
                + math.lgamma(self.theta / self.d + self.ntables)
                - math.lgamma(self.theta / self.d)
                + self.ntables * (math.log(self.d) - math.lgamma(1 - self.d))
                + sum(math.lgamma(n - self.d) for tables in self.tables.itervalues() for n in tables)
                + self.base.log_likelihood()) # XXX base might be shared!
        #print self._ll, ll, self.pseudo_log_likelihood()
        return ll

    def __str__(self):
        return 'PYP(d={self.d}, theta={self.theta}, #customers={self.total_customers}, #tables={self.ntables}, #dishes={V}, Base={self.base})'.format(self=self, V=len(self.tables))
