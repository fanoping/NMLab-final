import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('CSV/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv', dtype={' Flow IAT Min': int,' Bwd IAT Std': float, 'Bwd IAT Mean': float, ' Bwd IAT Max': float, 'label': str})
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

k = 10
split = np.array_split(train_x, k)
split_label = np.array_split(train_label, k)

def k_fold_cross_validation(k):
	for i in range(k):
		train = [split[l] for l in range(len(split)) if l != i]
		valid = split[i]

		train_label = [split_label[l] for l in range(len(split_label)) if l != i] 
		valid_label = split_label[i]
		yield train, valid, train_label, valid_label

splitted_data = list(k_fold_cross_validation(k))


for idx, (train, valid, train_label, valid_label) in enumerate(splitted_data):
	neigh = KNeighborsClassifier(n_neighbors=3)
	train = np.concatenate(train)
	train_label = np.concatenate(train_label)
	neigh.fit(train, train_label)
	score = neigh.score(valid, valid_label)
	print('{0:2d} fold score: {:.6f}'.format(idx+1, score))
