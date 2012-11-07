import argparse
import logging
import math
import cPickle
from itertools import izip
from ..corpus import Vocabulary
from model import AlignmentModel

NULL = 'NULL'

def read_parallel_corpus(stream, source_vocabulary, target_vocabulary):
    def sentences():
        for line in stream:
            f, e = line.split(' ||| ')
            yield ([source_vocabulary[w] for w in [NULL]+f.split()], 
                    [target_vocabulary[w] for w in e.split()])
    return list(sentences())

mh_iter = 100 # number of Metropolis-Hastings sampling iterations

def run_sampler(model, corpus, n_iter):
    n_words = sum(len(e) for f, e in corpus)
    alignments = [None] * len(corpus)
    samples = []
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for i, (f, e) in enumerate(corpus):
            if it > 0: model.decrement(f, e, alignments[i])
            alignments[i] = list(model.increment(f, e))
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
        if it % 100 == 99:
            logging.info('Estimating sample')
            samples.append(model.map_estimate())

    logging.info('Combining %d samples', len(samples))
    align = AlignmentModel.combine(samples)
    for i, (f, e) in enumerate(corpus):
        alignments[i] = list(align(f, e))
    return alignments

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train alignment model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--output', help='model output path')

    args = parser.parse_args()

    source_vocabulary = Vocabulary()
    source_vocabulary[NULL]
    target_vocabulary = Vocabulary()

    logging.info('Reading parallel training data')
    with open(args.train) as train:
        training_corpus = read_parallel_corpus(train, source_vocabulary, target_vocabulary)

    model = AlignmentModel(len(source_vocabulary), len(target_vocabulary))

    logging.info('Training alignment model')
    alignments = run_sampler(model, training_corpus, args.iter)

    if args.output:
        with open(args.output, 'w') as f:
            model.source_vocabulary = source_vocabulary
            model.target_vocabulary = target_vocabulary
            cPickle.dump(model, f, protocol=-1)

    for a, (f, e) in izip(alignments, training_corpus):
        f_sentence = ' '.join(source_vocabulary[w] for w in f[1:])
        e_sentence = ' '.join(target_vocabulary[w] for w in e)
        al = ' '.join('{0}-{1}'.format(j-1, i) for i, j in enumerate(a) if j > 0)
        print f_sentence, '|||', e_sentence, '|||', al

if __name__ == '__main__':
    main()
