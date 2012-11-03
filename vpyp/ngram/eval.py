import argparse
import logging
import math
import cPickle
from ..corpus import read_corpus, ngrams

def print_ppl(model, corpus):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    n_oovs = 0
    ll = 0
    for sentence in corpus:
        for seq in ngrams(sentence, model.order):
            p = model.prob(seq[:-1], seq[-1])
            if p == 0:
                n_oovs += 1
            else:
                ll += math.log(p)
    ppl = math.exp(-ll/(n_sentences + n_words - n_oovs))
    logging.info('Sentences: %d\tWords: %d\tOOVs: %d', n_sentences, n_words, n_oovs)
    logging.info('LL: %.0f\tppl: %.3f', ll, ppl)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Evaluate n-gram model')
    parser.add_argument('--test', help='evaluation corpus', required=True)
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)

    logging.info('Reading evaluation corpus')
    with open(args.test) as test:
        test_corpus = read_corpus(test, model.vocabulary)

    logging.info('Computing perplexity')
    print_ppl(model, test_corpus)

if __name__ == '__main__':
    main()
