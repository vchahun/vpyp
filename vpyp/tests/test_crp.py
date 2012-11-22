from nose.tools import eq_
from ..pyp import CRP

def test_crp():
    crp = CRP()
    # add customers
    eq_(crp._seat_to(0, -1), True) # 0:[1]
    eq_(crp._seat_to(1, -1), True) # 0:[1] 1:[1]
    eq_(crp._seat_to(0, 0), False) # 0:[2] 1:[1]
    eq_(crp._seat_to(1, -1), True) # 0:[2], 1:[1, 1]
    eq_(crp._seat_to(1, 1), False) # 0:[2], 1:[1, 2]
    eq_(crp._seat_to(9, -1), True) # 0:[2], 1:[1, 2], 9:[1]
    # check configuration
    eq_(crp.tables, {0:[2], 1:[1, 2], 9:[1]})
    eq_(crp.ntables, 4)
    eq_(crp.ncustomers, {0:2, 1:3, 9:1})
    eq_(crp.total_customers, 6)
    # remove customers
    eq_(crp._unseat_from(1, 0), True)  # 0:[2], 1:[2], 9:[1]
    eq_(crp._unseat_from(1, 0), False) # 0:[2], 1:[1], 9:[1]
    eq_(crp._unseat_from(9, 0), True)  # 0:[2], 1:[1]
    eq_(crp._unseat_from(0, 0), False) # 0:[1], 1:[1]
    eq_(crp._unseat_from(0, 0), True)  # 1:[1]
    eq_(crp._unseat_from(1, 0), True)
    # check configuration
    eq_(crp.tables, {})
    eq_(crp.ntables, 0)
    eq_(crp.ncustomers, {})
    eq_(crp.total_customers, 0)
