import argparse
import logging
import cPickle
import heapq

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print alignment model')
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    with open(args.model) as m:
        model = cPickle.load(m)

    for f, t_word in enumerate(model.t_table):
        t_best = heapq.nlargest(10, ((t_word.prob(e), e) for e in  t_word.tables))
        for p, e in t_best:
            if p < 0.1: continue
            print model.source_vocabulary[f], '->', model.target_vocabulary[e], ' = ', p

if __name__ == '__main__':
    main()
