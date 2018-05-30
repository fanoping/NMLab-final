import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse


def main(args):
    """
        Scenario A:
            Feature extractor: CfsSubsetEval+BestFirst (SE+BF)
            Classifier: K nearest neighboring (K = 3), decision tree classifier
            Validation: 10-folded cross validation
    """
    data = pd.read_csv('CSV/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv')

    min_flowiat = data['Flow IAT Min']
    std_biat = data['Bwd IAT Std']
    mean_biat = data['Bwd IAT Mean']
    max_biat = data['Bwd IAT Max']
    label = data['label']

    train_x = np.array([min_flowiat, std_biat, mean_biat, max_biat]).T
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
        print('{} fold score: {:.6f}'.format(idx+1, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classifier for Scenario A')
    parser.add_argument('-k', default=10, type=int,
                        help='k folded cross validation')
    parser.add_argument('--arch', default='knn', type=str,
                        help='classification method')
    main(parser.parse_args())
