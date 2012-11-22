import argparse
import logging
import math
import sys
import cPickle
from ..corpus import START, STOP

def print_arpa(model, vocabulary):
    vocabulary |= {START, STOP}
    levels = [None for _ in range(model.order)]
    level = model
    for k in range(model.order):
        levels[model.order-k-1] = level
        if k < model.order - 1:
            level = level.backoff

    def backoff_field(ctx, n):
        if n > model.order - 1: return ''
        m = levels[n].models.get(ctx, None)
        if not m: return ''
        return '\t'+str(math.log10((m.theta + m.d * m.ntables)/(m.theta + m.total_customers)))

    def lines():
        yield '\\data\\'
        for n in range(model.order):
            level = levels[n]
            if n == 0:
                n_ngrams = len(vocabulary)
            else:
                n_ngrams = 0
                for ctx, m in level.models.iteritems():
                    if any(c not in vocabulary for c in ctx): continue
                    if sum(c == START for c in ctx) > 1: continue
                    for w in m.tables.iterkeys():
                        if w not in vocabulary: continue
                        n_ngrams += 1
            yield 'ngram {0}={1}'.format(n+1, n_ngrams)
        yield ''
        for n in range(model.order):
            level = levels[n]
            yield '\\{0}-grams:\\'.format(n+1)
            if n == 0:
                m = level[()]
                for w in vocabulary:
                    if w == START:
                        yield u'-99\t<s>'+backoff_field((START,), 1)
                    else:
                        yield u'{0}\t{1}{2}'.format(math.log10(m.prob(w)), model.vocabulary[w],
                            backoff_field((w,), 1))
            else:
                for ctx, m in level.models.iteritems():
                    if any(c not in vocabulary for c in ctx): continue
                    if sum(c == START for c in ctx) > 1: continue # <s> <s>+ *
                    context = ' '.join(model.vocabulary[c] for c in ctx)
                    if ctx[0] == START: # extend to full context
                        full_ctx = (START,)*(model.order-n-1)+ctx
                        m = levels[model.order-1][full_ctx]
                    for w in m.tables.iterkeys():
                        if w not in vocabulary: continue
                        yield u'{0}\t{1} {2}'.format(math.log10(m.prob(w)), context,
                                model.vocabulary[w])+backoff_field(ctx+(w,), n+1)
            yield ''
        yield '\\end\\'
    sys.stdout.writelines(l.encode('utf8')+'\n' for l in lines())

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Export n-gram model as ARPA file')
    parser.add_argument('--vocab', help='test corpus vocabulary', required=True)
    parser.add_argument('--model', help='trained model', required=True)

    args = parser.parse_args()

    logging.info('Loading model')
    with open(args.model) as model_file:
        model = cPickle.load(model_file)

    logging.info('Reading vocabulary')
    with open(args.vocab) as vocab:
        vocabulary = set(model.vocabulary[w.strip().decode('utf8')] for w in vocab)

    print_arpa(model, vocabulary)

if __name__ == '__main__':
    main()
