"""
Ensemble the predictions from different model outputs.
"""
import argparse
import json
import pickle
import numpy as np
from collections import Counter

from utils import scorer, constant


def parse_args():
    parser = argparse.ArgumentParser ()

    parser.add_argument ('--data_dir', default='dataset/tacred')
    parser.add_argument ('--dataset', default='test', help='Evaluate on dev or test set.')
    args = parser.parse_args ()
    return args


def main():
    args = parse_args ()
    print ("Loading data file...")
    filename = args.data_dir + '/{}.json'.format (args.dataset)
    with open (filename, 'r') as infile:
        data = json.load (infile, encoding='utf8')
    labels = [d['relation'] for d in data]

    # read predictions

    #args.pred_files = ['./res_pkl/trans.pkl','./res_pkl/palstm.pkl']
    args.pred_files = ['./res_pkl/palstm.pkl']
    print ("Loading {} prediction files...".format (len (args.pred_files)))
    scores_list = []
    for path in args.pred_files:
        with open (path, 'rb') as infile:
            scores = pickle.load (infile)
            scores_list += [scores]

    print ("Calculating ensembled predictions...")
    predictions = []
    scores_by_examples = list (zip (*scores_list))
    assert len (scores_by_examples) == len (data)
    for scores in scores_by_examples:
        pred = ensemble (scores)
        predictions += [pred]
    id2label = dict ([(v, k) for k, v in constant.LABEL_TO_ID.items ()])
    predictions = [id2label[p] for p in predictions]
    scorer.score (labels, predictions, verbose=True)


def ensemble(scores):
    """
    Ensemble by majority vote.
    """
    c = Counter ()
    for probs in zip (scores):
        idx = int (np.argmax (np.array (probs)))
        c.update ([idx])
    best = c.most_common (1)[0][0]
    return best

# def ensemble(scores):
#     """
#     Ensemble by majority vote.
#     """
#
#     probs = [0.5*p1+0.5*p2 for p1,p2 in zip(scores[0],scores[1])]
#     idx = int (np.argmax (np.array (probs)))
#     return idx

if __name__ == '__main__':
    main ()
