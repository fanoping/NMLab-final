import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

"""
    Scenario A, 
    Feature extractor: CfsSubsetEval+BestFirst (SE+BF)
    using K nearest neighboring, 10-folded cross validation
"""

data = pd.read_csv('CSV/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv')

min_flowiat = data[' Flow IAT Min']
std_biat = data[' Bwd IAT Std']
mean_biat = data['Bwd IAT Mean']
max_biat = data[' Bwd IAT Max']
label = data['label']


train_x = np.array([min_flowiat, std_biat, mean_biat, max_biat]).T
train_label = [1 if item == 'TOR' else 0 for item in label]
train_label = np.array(train_label).T

"""
valid_x = train_x[60000:]
train_x = train_x[:60000]
#print(train_x.shape)
valid_label = train_label[60000:]
train_label = train_label[:60000]


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, train_label)

predict = neigh.predict(valid_x)
score = neigh.score(valid_x, valid_label)
print(score)

zeros = np.zeros(predict.shape)
print(np.mean(np.array(predict==zeros)))
"""


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
    neigh = KNeighborsClassifier(n_neighbors=3)
    train = np.concatenate(train)
    train_label = np.concatenate(train_label)
    neigh.fit(train, train_label)
    score = neigh.score(valid, valid_label)
    print('{} fold score: {:.6f}'.format(idx+1, score))
