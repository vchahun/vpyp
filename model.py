import logging
from collections import defaultdict
from prob import mult_sample, remove_random, DirichletMultinomial, Uniform, GammaPrior, BetaGammaPrior
from pyp import PYP

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

    def resample_hyperparemeters(self, niter):
        acceptance, rejection = 0, 0
        logging.info('Resampling doc-topic hyperparameters')
        a, r = self.alpha.resample(niter)
        acceptance += a
        rejection += r
        logging.info('Resampling topic-word hyperparameters')
        a, r = self.beta.resample(niter)
        acceptance += a
        rejection += r
        return (acceptance, rejection)

    def map_estimate(self, n_words):
        for topic in self.topic_word:
            yield [topic.prob(word) for word in range(n_words)]

    def __str__(self):
        #dts = '\n'.join(map(str, self.document_topic))
        #tws = '\n'.join(map(str, self.topic_word))
        return 'TopicModel(#topics={self.n_topics} | alpha={self.alpha}, beta={self.beta}):'.format(self=self)
                #'\n-> Document-topic processes:\n{dts}'
                #'\n-> Topic-word processes:\n{tws}').format(self=self, dts=dts, tws=tws)

class LDA(TopicModel):
    def __init__(self, n_topics, n_docs, n_words):
        super(LDA, self).__init__(n_topics)
        self.alpha = GammaPrior(1, 1, 1) # alpha = 1
        self.beta = GammaPrior(1, 1, 1) # alpha = 1
        self.document_topic = [DirichletMultinomial(n_topics, self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [DirichletMultinomial(n_words, self.beta) for _ in xrange(n_topics)]
        self.alpha.tie(self.document_topic)
        self.beta.tie(self.topic_word)

class LPYA(TopicModel):
    def __init__(self, n_topics, n_docs, n_words):
        super(LPYA, self).__init__(n_topics)
        self.alpha = BetaGammaPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.1) # d, theta = 0.1, 1
        self.beta = BetaGammaPrior(1.0, 1.0, 1.0, 1.0, 0.1, 1.1) # d, theta = 0.1, 1
        # TODO share base / fix likelihood
        self.document_topic = [PYP(Uniform(n_topics), self.alpha) for _ in xrange(n_docs)]
        self.topic_word = [PYP(Uniform(n_words), self.beta) for _ in xrange(n_topics)]
        self.alpha.tie(self.document_topic)
        self.beta.tie(self.topic_word)