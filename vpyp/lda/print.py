import argparse
import logging
import cPickle
import heapq

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Print LDA model')
    parser.add_argument('model', help='trained model')
    args = parser.parse_args()

    with open(args.model) as m:
        model = cPickle.load(m)
    
    for i, topic in enumerate(model.topic_word):
        print('Topic {0}'.format(i))
        word_prob = ((topic.prob(w), w) for w in xrange(len(model.vocabulary)))
        for prob, w in heapq.nlargest(10, word_prob):
            print(u'{0} {1}'.format(model.vocabulary[w], prob).encode('utf8'))
        print('---------')

if __name__ == '__main__':
    main()
