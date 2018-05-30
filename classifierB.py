import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse


def main(args):
    """
        Scenario B:
            Feature extractor: CfsSubsetEval+BestFirst (SE+BF)
            Classifier: K nearest neighboring, decision tree classifier
            Validation: 10-folded cross validation
    """
    data = pd.read_csv('CSV/Scenario-B/TimeBasedFeatures-10s-Layer2.csv')

    # attributes
    duration = data['Flow Duration']
    flowBytesPerSecond = data['Flow Bytes/s']
    mean_flowiat = data['Flow IAT Mean']
    max_flowiat = data['Flow IAT Max']
    min_flowiat = data['Flow IAT Min']
    mean_fiat = data['Fwd IAT Mean']
    max_fiat = data['Fwd IAT Max']
    min_fiat = data['Fwd IAT Min']
    min_biat = data['Bwd IAT Min']
    label = data['label']

    # labels
    labels = {'BROWSING': 0, 'AUDIO': 1, 'CHAT': 2, 'MAIL': 3, 'P2P': 4,
              'FILE-TRANSFER': 5, 'VOIP':6, 'VIDEO': 7}


    train_x = np.array([duration, flowBytesPerSecond, mean_flowiat, max_flowiat, min_flowiat,
                        mean_fiat, max_fiat, min_fiat, min_biat]).T
    train_label = [labels[item] for item in label]
    train_label = np.array(train_label).T

    def k_fold_cross_validation(k_fold, train_x, label):
        split = np.array_split(train_x, k_fold)
        split_label = np.array_split(label, k_fold)

        for val_idx in range(k):
            train = [split[idx] for idx in range(len(split)) if idx != val_idx]
            valid = split[val_idx]

            train_label = [split_label[idx] for idx in range(len(split_label)) if idx != val_idx]
            valid_label = split_label[val_idx]
            yield train, valid, train_label, valid_label

    k = 10
    splitted_data = list(k_fold_cross_validation(k, train_x, train_label))

    for idx, (train, valid, train_label, valid_label) in enumerate(splitted_data):
        if args.arch.lower() == 'knn':
            neigh = KNeighborsClassifier(n_neighbors=3)
        elif args.arch.lower() == 'tree':
            neigh = DecisionTreeClassifier()
        elif args.arch.lower() == 'forest':
            neigh = RandomForestClassifier(n_estimators=20, random_state=2)
        else:
            raise NotImplementedError(args.arch)

        train = np.concatenate(train)
        train_label = np.concatenate(train_label)
        neigh.fit(train, train_label)
        score = neigh.score(valid, valid_label)
        print('{} fold score: {:.6f}'.format(idx+1, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classifier for Scenario A')
    parser.add_argument('--arch', default='knn', type=str,
                        help='classification method [knn, tree, forest]')
    main(parser.parse_args())
