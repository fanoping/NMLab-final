import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import json
import argparse


def main(args):
    """
        Scenario A:
            Feature extractor: CfsSubsetEval+BestFirst (SE+BF)
            Classifier: K nearest neighboring (K = 3), decision tree classifier
            Validation: 10-folded cross validation
    """
    csv_file = pd.read_csv(args.input_csv)
    config = json.load(open(args.config))['Scenario-A']

    attributes = [csv_file[attr] for attr, usage in config["attribute"].items() if usage]
    label = csv_file["label"] if config["label"] else ValueError("No label specified!")

    train_x = np.array(attributes).T
    train_label = [1 if item == 'TOR' else 0 for item in label]
    train_label = np.array(train_label).T

    def k_fold_cross_validation(k_fold, train_x, label):
        split = np.array_split(train_x, k_fold)
        split_label = np.array_split(label, k_fold)

        for val_idx in range(k_fold):
            train = [split[idx] for idx in range(len(split)) if idx != val_idx]
            valid = split[val_idx]

            train_label = [split_label[idx] for idx in range(len(split_label)) if idx != val_idx]
            valid_label = split_label[val_idx]
            yield train, valid, train_label, valid_label

    splitted_data = list(k_fold_cross_validation(args.k, train_x, train_label))

    total = 0
    for idx, (train, valid, train_label, valid_label) in enumerate(splitted_data):
        if args.arch.lower() == 'knn':
            neigh = KNeighborsClassifier(n_neighbors=3)
        elif args.arch.lower() == 'tree':
            neigh = DecisionTreeClassifier()
        else:
            return NotImplementedError

        train = np.concatenate(train)
        train_label = np.concatenate(train_label)
        neigh.fit(train, train_label)
        score = neigh.score(valid, valid_label)

        print('{} fold:'.format(idx+1))
        print('\tAccuracy: {:.6f}'.format(score))
        # print('\tPrecision: {:.6f}'.format(precision_score(valid_label, neigh.predict(valid))))
        # print('\tRecall: {:.6f}'.format(recall_score(valid_label, neigh.predict(valid))))

        total += score
    print("Ave: {:.6f}".format(total / args.k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classifier for Scenario A')
    parser.add_argument('-k', default=10, type=int,
                        help='k folded cross validation')
    parser.add_argument('--input-csv', default='CSV/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv',
                        help='input information from csv file')
    parser.add_argument('--config', default='config.json',
                        help='specify the selected feature')
    parser.add_argument('--arch', default='knn', type=str,
                        help='classification method')
    main(parser.parse_args())
