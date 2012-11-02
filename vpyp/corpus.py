from collections import deque
from itertools import chain, repeat

START, STOP = 0, 1

class Vocabulary:
    def __init__(self):
        self.word2id = {'<s>': START, '</s>': STOP}
        self.id2word = ['<s>', '</s>']

    def __getitem__(self, word):
        if isinstance(word, int):
            assert word >= 0
            return self.id2word[word]
        if word not in self.word2id:
            self.word2id[word] = len(self)
            self.id2word.append(word)
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

def read_corpus(stream, vocabulary):
    return [[vocabulary[word] for word in doc.split()] for doc in stream]

def ngrams(sentence, order):
    ngram = deque(maxlen=order)
    for w in chain(repeat(START, order-1), sentence, (STOP,)):
        ngram.append(w)
        if len(ngram) == order:
            yield tuple(ngram)
