import argparse
import logging
import cPickle
import heapq

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print LDA model')
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    with open(args.model) as m:
        model = cPickle.load(m)
    
    for topic in model.topic_word:
        word_prob = ((topic.prob(w), w) for w in xrange(len(model.vocabulary)))
        for prob, w in heapq.nlargest(10, word_prob):
            print model.vocabulary[w], prob
    print

if __name__ == '__main__':
    main()
