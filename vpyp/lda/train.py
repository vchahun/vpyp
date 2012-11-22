import argparse
import logging
import math
import cPickle
from ..corpus import Vocabulary, read_corpus
from ..prob import Uniform
from model import LDA, LPYA

mh_iter = 100 # number of Metropolis-Hastings sampling iterations

def run_sampler(model, corpus, n_iter, cb=None):
    assignments = [[None]*len(document) for document in corpus]
    n_words = sum(len(document) for document in corpus)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for d, document in enumerate(corpus):
            document_assignments = assignments[d]
            for i, word in enumerate(document):
                if it > 0: model.decrement(d, word, document_assignments[i])
                document_assignments[i] = model.increment(d, word)
        if it % 10 == 0:
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / n_words)
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        if it % 30 == 29:
            logging.info('Resampling hyperparameters...')
            acceptance, rejection = model.resample_hyperparemeters(mh_iter)
            arate = acceptance / float(acceptance + rejection)
            logging.info('Metropolis-Hastings acceptance rate: %.4f', arate)
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / n_words)
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        if cb: cb(it)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train LDA model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--pyp', help='use pyp priors', action='store_true')
    parser.add_argument('--output', help='model output path')

    args = parser.parse_args()

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = read_corpus(train, vocabulary)

    if args.pyp:
        topic_base = Uniform(len(vocabulary))
        model = LPYA(args.topics, len(training_corpus), topic_base)
    else:
        model = LDA(args.topics, len(training_corpus), len(vocabulary))

    logging.info('Training model with %d topics', args.topics)
    run_sampler(model, training_corpus, args.iter)

    if args.output:
        model.vocabulary = vocabulary
        with open(args.output, 'w') as f:
            cPickle.dump(model, f, protocol=-1)

if __name__ == '__main__':
    main()
