class OOV(Exception): pass

class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.frozen = False

    def __getitem__(self, word):
        if isinstance(word, int):
            assert word >= 0
            return self.id2word[word]
        if word not in self.word2id:
            if self.frozen: raise OOV(word)
            self.word2id[word] = len(self)
            self.id2word.append(word)
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

def read_corpus(stream, vocabulary):
    return [[vocabulary[word] for word in doc.split()] for doc in stream]
