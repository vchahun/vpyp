import logging
from ..prob import mult_sample, DirichletMultinomial
from ..prior import GammaPrior, PYPPrior, stuple
from ..pyp import PYP

class TopicModel(object):
    def __init__(self, n_topics):
        self.n_topics = n_topics

    def increment(self, doc, word):
        z = mult_sample((k, self.topic_prob(doc, word, k)) for k in xrange(self.n_topics))
        self.document_topic[doc].increment(z)
        self.topic_word[z].increment(word)
        return z

    def decrement(self, doc, word, z):
        self.document_topic[doc].decrement(z)
        self.topic_word[z].decrement(word)

    def topic_prob(self, doc, word, k):
        return self.document_topic[doc].prob(k) * self.topic_word[k].prob(word)

    def prob(self, doc, word):
        return sum(self.topic_prob(doc, word, k) for k in xrange(self.n_topics))

    def map_estimate(self, n_words):
        for topic in self.topic_word:
            yield [topic.prob(word) for word in range(n_words)]

class LDA(TopicModel):
    def __init__(self, n_topics, n_docs, n_words):
        super(LDA, self).__init__(n_topics)
        self.alpha = GammaPrior(1.0, 1.0, 1.0) # alpha = 1
        self.beta = GammaPrior(1.0, 1.0, 1.0) # alpha = 1
        self.document_topic = [DirichletMultinomial(n_topics, self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [DirichletMultinomial(n_words, self.beta) for _ in xrange(n_topics)]

    def log_likelihood(self):
        return (sum(d.log_likelihood() for d in self.document_topic)
                + self.alpha.log_likelihood()
                + sum(t.log_likelihood() for t in self.topic_word)
                + self.beta.log_likelihood())

    def resample_hyperparemeters(self, n_iter):
        logging.info('Resampling doc-topic hyperparameters')
        a1, r1 = self.alpha.resample(n_iter)
        logging.info('Resampling topic-word hyperparameters')
        a2, r2 = self.beta.resample(n_iter)
        return (a1+a2, r1+r2)

    def __repr__(self):
        return ('LDA(#topics={self.n_topics} '
                '| alpha={self.alpha}, beta={self.beta})').format(self=self)

class LPYA(TopicModel):
    def __init__(self, n_topics, n_docs, topic_base):
        super(LPYA, self).__init__(n_topics)
        self.alpha = GammaPrior(1.0, 1.0, 1.0) # alpha = 1
        self.topic_base = topic_base
        self.document_topic = [DirichletMultinomial(n_topics, self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [PYP(self.topic_base, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0)) 
                for _ in xrange(n_topics)]

    def log_likelihood(self):
        return (sum(d.log_likelihood() for d in self.document_topic)
                + self.alpha.log_likelihood()
                + sum(t.log_likelihood() + t.prior.log_likelihood() for t in self.topic_word)
                + self.topic_base.log_likelihood(full=True))

    def resample_hyperparemeters(self, n_iter):
        ar = stuple((0, 0))
        logging.info('Resampling topic-word PYP base hyperparameters')
        ar += self.topic_base.resample_hyperparemeters(n_iter) # G_w^0
        logging.info('Resampling doc-topic hyperparameters')
        ar += self.alpha.resample(n_iter) # alpha
        logging.info('Resampling all topic-word PYP hyperparameters')
        for topic in self.topic_word:
            ar += topic.resample_hyperparemeters(n_iter) # d_w, T_w
        return ar

    def __repr__(self):
        return ('LPYA(#topics={self.n_topics} '
                '| alpha={self.alpha}, beta=PYP(base={self.topic_base}))').format(self=self)
