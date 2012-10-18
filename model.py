from collections import defaultdict
from prob import mult_sample, remove_random

class TopicModel:
    def __init__(self, n_topics, n_docs, doc_process, topic_process):
        self.n_topics = n_topics
        self.document_topic = [doc_process() for _ in xrange(n_docs)]
        self.topic_word = [topic_process() for _ in xrange(n_topics)]
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
                + sum(d.log_likelihood() for d in self.document_topic))

    def map_estimate(self, n_words):
        for topic in self.topic_word:
            yield [topic.prob(word) for word in range(n_words)]

    def __str__(self):
        return 'TopicModel(#topics={self.n_topics})'.format(self=self)
