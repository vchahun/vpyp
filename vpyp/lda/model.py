import logging
from collections import defaultdict
from ..prob import mult_sample, remove_random, DirichletMultinomial, Uniform
from ..prior import GammaPrior, PYPPrior
from ..pyp import PYP

class TopicModel(object):
    def __init__(self, n_topics):
        self.n_topics = n_topics
        self.assignments = defaultdict(list)

    def increment(self, doc, word):
        z = mult_sample((k, self.topic_prob(doc, word, k)) for k in xrange(self.n_topics))
        self.assignments[doc, word].append(z)
        self.document_topic[doc].increment(z)
        self.topic_word[z].increment(word)

    def decrement(self, doc, word):
        z = remove_random(self.assignments[doc, word])
        self.document_topic[doc].decrement(z)
        self.topic_word[z].decrement(word)

    def topic_prob(self, doc, word, k):
        return self.document_topic[doc].prob(k) * self.topic_word[k].prob(word)

    def prob(self, doc, word):
        return sum([self.topic_prob(doc, word, k) for k in xrange(self.n_topics)])

    def log_likelihood(self):
        return (sum(t.log_likelihood() for t in self.topic_word)
                + self.alpha.log_likelihood()
                + sum(d.log_likelihood() for d in self.document_topic)
                + self.beta.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling doc-topic hyperparameters')
        a1, r1 = self.alpha.resample(n_iter)
        logging.info('Resampling topic-word hyperparameters')
        a2, r2 = self.beta.resample(n_iter)
        return (a1+a2, r1+r2)

    def map_estimate(self, n_words):
        for topic in self.topic_word:
            yield [topic.prob(word) for word in range(n_words)]

    def __repr__(self):
        return ('TopicModel(#topics={self.n_topics} '
                '| alpha={self.alpha}, beta={self.beta})').format(self=self)

class LDA(TopicModel):
    def __init__(self, n_topics, n_docs, n_words):
        super(LDA, self).__init__(n_topics)
        self.alpha = GammaPrior(1.0, 1.0, 1.0) # alpha = 1
        self.beta = GammaPrior(1.0, 1.0, 1.0) # alpha = 1
        self.document_topic = [DirichletMultinomial(n_topics, self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [DirichletMultinomial(n_words, self.beta) for _ in xrange(n_topics)]

class LPYA(TopicModel):
    def __init__(self, n_topics, n_docs, n_words):
        super(LPYA, self).__init__(n_topics)
        self.alpha = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.0) # d, theta = 0.1, 1
        self.beta = PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0) # d, theta = 0.8, 1
        self.document_base = Uniform(n_topics)
        self.topic_base = Uniform(n_words)
        self.document_topic = [PYP(self.document_base, self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [PYP(self.topic_base, self.beta) for _ in xrange(n_topics)]

    def log_likelihood(self):
        return (super(LPYA, self).log_likelihood()
                + self.document_base.log_likelihood()
                + self.topic_base.log_likelihood())
