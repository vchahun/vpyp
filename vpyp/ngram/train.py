import argparse
import logging
import math
import cPickle
from ..corpus import Vocabulary, read_corpus, ngrams
from ..prob import Uniform
from ..pyp import PYP
from ..prior import PYPPrior
from model import PYPLM

mh_iter = 100 # number of Metropolis-Hastings sampling iterations

def run_sampler(model, corpus, n_iter):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence in corpus:
            for seq in ngrams(sentence, model.order):
                if it > 0: model.decrement(seq[:-1], seq[-1])
                model.increment(seq[:-1], seq[-1])
        if it % 10 == 0:
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        if it % 30 == 29:
            logging.info('Resampling hyperparameters...')
            acceptance, rejection = model.resample_hyperparemeters(mh_iter)
            arate = acceptance / float(acceptance + rejection)
            logging.info('Metropolis-Hastings acceptance rate: %.4f', arate)
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train n-gram model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--order', help='order of the model', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--pyp', help='backoff to PYP(CharLM)', action='store_true')
    parser.add_argument('--charlm', help='use a character LM as a base distribution')
    parser.add_argument('--output', help='model output path')

    args = parser.parse_args()

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = read_corpus(train, vocabulary)

    if args.charlm:
        from ..charlm import CharLM
        char_lm = CharLM(args.charlm, vocabulary)
        if args.pyp:
            base = PYP(char_lm, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0))
        else:
            base = char_lm
    else:
        base = Uniform(len(vocabulary))
    model = PYPLM(args.order, base)

    logging.info('Training model of order %d', args.order)
    run_sampler(model, training_corpus, args.iter)

    if args.output:
        model.vocabulary = vocabulary
        with open(args.output, 'w') as f:
            cPickle.dump(model, f, protocol=-1)

if __name__ == '__main__':
    main()
