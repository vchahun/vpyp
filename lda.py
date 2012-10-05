import argparse
import logging
import math
from corpus import Vocabulary, read_corpus
from prob import Uniform, DirichletMultinomial
from pyp import PYP
from model import TopicModel

theta_doc = 1.0
d_doc = 0.8
theta_topic = 1.0
d_topic = 0.8

def run_sampler(model, corpus, n_iters):
    for it in range(n_iters):
        n_words = 0
        logging.info('Iteration %d/%d', it+1, n_iters)
        for d, document in enumerate(corpus):
            for word in document:
                n_words += 1
                if it > 0: model.decrement(d, word)
                model.increment(d, word)
        if it % 10 == 0:
            ll = model.log_likelihood()
            ppl = math.exp(-ll / n_words)
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
            logging.info('Model: %s', model)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train LDA model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--topics', help='number of topics', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--pyp', help='use pyp priors', action='store_true')
    args = parser.parse_args()

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = read_corpus(train, vocabulary)

    if args.pyp:
        logging.info('Using a PYP prior')
        doc_process = lambda: PYP(theta_doc, d_doc, Uniform(args.topics))
        topic_process = lambda: PYP(theta_topic, d_topic, Uniform(len(vocabulary)))
    else:
        logging.info('Using a Dirichlet prior')
        doc_process = lambda: DirichletMultinomial(args.topics, theta_doc)
        topic_process = lambda: DirichletMultinomial(len(vocabulary), theta_topic)

    model = TopicModel(args.topics, len(training_corpus), doc_process, topic_process) 

    logging.info('Training model with %d topics', args.topics)
    run_sampler(model, training_corpus, args.iter)

if __name__ == '__main__':
    main()
