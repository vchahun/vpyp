from collections import deque
from itertools import chain, repeat

START, STOP = 0, 1

class OOV(Exception): pass

class Vocabulary:
    def __init__(self, start_stop=True, init=None):
        if start_stop:
            self.word2id = {'<s>': START, '</s>': STOP}
            self.id2word = ['<s>', '</s>']
        else:
            self.word2id = {}
            self.id2word = []
        self.frozen = False
        if init is not None:
            for word in init:
                self[word]

    def __getitem__(self, word):
        if isinstance(word, int):
            assert word >= 0
            return self.id2word[word]
        if word not in self.word2id:
            if self.frozen: raise OOV(word)
            self.word2id[word] = len(self)
            self.id2word.append(word)
        return self.word2id[word]

    def __setitem__(self, word, value):
        if isinstance(word, int):
            assert 0 <= word < len(self)
            self.id2word[word] = value
            self.word2id[value] = word
        else:
            assert 0 <= value < len(self)
            self.id2word[value] = word
            self.word2id[word] = value

    def update(self, vocabulary):
        self.word2id = vocabulary.word2id
        self.id2word = vocabulary.id2word

    def __iter__(self):
        return iter(self.id2word)

    def __len__(self):
        return len(self.id2word)

def read_corpus(stream, vocabulary):
    return [[vocabulary[word] for word in seg.decode('utf8').split()] for seg in stream]

class Corpus:
    def __init__(self, stream, vocabulary=None):
        self.vocabulary = (Vocabulary() if vocabulary is None else vocabulary)
        self.segments = read_corpus(stream, self.vocabulary)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

def ngrams(sentence, order):
    ngram = deque(maxlen=order)
    for w in chain(repeat(START, order-1), sentence, (STOP,)):
        ngram.append(w)
        if len(ngram) == order:
            yield tuple(ngram)
