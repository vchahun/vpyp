vpyp
====

An implementation of Pitman-Yor processes in Python, inspired by [cpyp](http://github.com/redpony/cpyp).

Implemented models:
- an n-gram model ([Teh, 2006](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf))
- a topic model (LDA with a Pitman-Yor prior instead of a Dirichlet prior)
- an alignment model (variant of the IBM 2 model)

Requirements:
- Python 2.7 ([pypy](http://www.pypy.org) is highly recommended for speed)
- the [kenlm](http://github.com/vchahun/kenlm) module, if you want to use character language models as base distributions
