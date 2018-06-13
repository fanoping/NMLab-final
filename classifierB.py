import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import argparse
import json


def main(args):
    """
        Scenario B:
            Feature extractor: CfsSubsetEval+BestFirst (SE+BF)
            Classifier: K nearest neighboring (K = 3), decision tree classifier, random forest
            Validation: k-folded cross validation
    """
    csv_file = pd.read_csv(args.train_csv)
    test_csv_file = pd.read_csv(args.test_csv)
    config = json.load(open(args.config))['Scenario-B']

    attributes = [csv_file[attr] for attr, usage in config["attribute"].items() if usage]
    test_attributes = [test_csv_file[attr] for attr, usage in config["attribute"].items() if usage]

    # labels
    label = csv_file["label"] if config["label"] else ValueError("No label specified!")
    labels = {'BROWSING': 0, 'AUDIO': 1, 'CHAT': 2, 'MAIL': 3, 'P2P': 4,
              'FILE-TRANSFER': 5, 'VOIP': 6, 'VIDEO': 7}

    train_x = np.array(attributes).T
    train_label = [labels[item] for item in label]
    train_label = np.array(train_label).T

    test_x = np.array(test_attributes).T

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

    if args.arch.lower() == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=3)
    elif args.arch.lower() == 'tree':
        neigh = DecisionTreeClassifier()
    elif args.arch.lower() == 'forest':
        neigh = RandomForestClassifier(n_estimators=20, random_state=2)
    else:
        raise NotImplementedError(args.arch)

    total = 0
    for idx, (train, valid, train_label, valid_label) in enumerate(splitted_data):
        train = np.concatenate(train)
        train_label = np.concatenate(train_label)
        neigh.fit(train, train_label)
        score = neigh.score(valid, valid_label)

        print('{} fold:'.format(idx + 1))
        print('\tAccuracy: {:.6f}'.format(score))
        # print('\tPrecision: {:.6f}'.format(precision_score(valid_label, neigh.predict(valid), average='micro')))
        # print('\tRecall: {:.6f}'.format(recall_score(valid_label, neigh.predict(valid), average='micro')))

        total += score
    print("Ave:", total/args.k)

    #print("Test data fitting accuracy: {:.6f}".format(neigh.score(test_x, test_label)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classifier for Scenario B')
    parser.add_argument('-k', default=20, type=int,
                        help='k folded cross validation')
    parser.add_argument('--train-csv', default='CSV/Scenario-B/TimeBasedFeatures-10s-Layer2.csv',
                        help='input information from csv file')
    parser.add_argument('--test-csv', default='realtime.pcap_Flow.csv',
                        help='input information from csv file for testing')
    parser.add_argument('--config', default='config.json',
                        help='specify the selected feature')
    parser.add_argument('--arch', default='knn', type=str,
                        help='classification method [knn, tree, forest]')
    main(parser.parse_args())
