import sys
from collections import Counter

import numpy as np

from grakel import WeisfeilerLehman, VertexHistogram, RandomWalk
from grakel.datasets import fetch_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle

from graph.ExperimentConductor import ExperimentConductor

# Loopwalks

ec = ExperimentConductor()
ec.setDatasets(ec.sets.large)
with open("results.txt", "a+") as f:
    sys.stdout = f
    print("rw")
    ec.gridSearchLabel()

# Comparison

# with open("results.txt", "a+") as f:
#     sys.stdout = f
#     for name in ["NCI1"]:
#         print(name)
#         dset = fetch_dataset(name, verbose=False)
#         X, y = shuffle(dset.data, dset.target)
#
#         skf = StratifiedKFold(n_splits=5)
#         splits = skf.split(X, y)
#         accuracies = []
#         for train_indices, test_indices in splits:
#             for normalize in [True, False]:
#                 bestClf = None
#                 bestGk = None
#                 for lamda in [2 ** i for i in [-12, -8, -5, -3, 1, 3, 5, 8, 12]]:
#                 # for n_iter in range(1, 10):
#                     try:
#                         # gk = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=normalize)
#                         gk = RandomWalk(lamda=lamda, verbose=True, normalize=normalize)
#                         K_train = gk.fit_transform([X[index] for index in train_indices])
#                         svc = SVC(kernel="precomputed", max_iter=3000)
#                         clf = GridSearchCV(svc, {'C': [2 ** i for i in [-12, -8, -5, -3, -1, 1, 3, 5, 8, 12]]},
#                                            scoring='accuracy', cv=3, n_jobs=6)
#                         clf.fit(K_train, [y[index] for index in train_indices])
#
#                         if bestClf is None:
#                             bestClf = clf
#                             bestGk = gk
#                         elif clf.best_score_ > bestClf.best_score_:
#                             bestClf = clf
#                             bestGk = gk
#                     except ValueError:
#                         # print("ValueError for lambda", lamda)
#                         pass
#                 K_test = bestGk.transform([X[index] for index in test_indices])
#                 y_pred = bestClf.predict(K_test)
#                 accuracies.append(accuracy_score([y[index] for index in test_indices], y_pred))
#         print(np.average(accuracies), np.std(accuracies))

# from graph.LabelConverter import LabelConverter
#
# graph = {
#     0: [1, 2, 3],
#     1: [0, 3, 4],
#     2: [0],
#     3: [0, 1],
#     4: [1]
# }
# labels = [0, 0, 1, 1, 1]
#
# conv = LabelConverter(6)
# counter = Counter()
# counter.update(conv.convertNode(1, 1, graph, labels, "", 5, []))
# print(counter)
