import itertools
from typing import Dict, List

import numpy as np
from grakel import Graph, WeisfeilerLehman, VertexHistogram
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from graph.LabelConverter import LabelConverter
from graph.MatrixConverter import MatrixConverter
from graph.UniqueLabelConverter import UniqueLabelConverter
from graph.UnlabeledLabelConverter import UnlabeledLabelConverter


class GridSearch:
    _additionalParams: Dict[str, List[int]]

    def __init__(self, additionalParams: Dict):
        self._additionalParams = additionalParams

    def calculateAccuracies(self, X: List[Graph], y: List[int], converter: str, n_fold=5):
        X, y = shuffle(X, y)
        skf = StratifiedKFold(n_splits=n_fold)
        splits = skf.split(X, y)

        accuracies = []
        for train_indices, test_indices in splits:
            score = self.doGridSplit(X, y, converter, train_indices, test_indices)
            accuracies.append(score)
        return np.average(accuracies), np.std(accuracies)

    def doGridSplit(self, X: List[Graph], y: List[int], converter: str, train_indices, test_indices):
        bestScore = 0
        bestParams = {}
        bestClf = None

        keys, values = zip(*self._additionalParams.items())
        for v in itertools.product(*values):
            experiment = dict(zip(keys, v))
            data = self.calculateMatrix(converter, experiment, X)
            X_train, y_train = self._getXy(data, y, train_indices)

            params, clf = self.doGridSearch(experiment, X_train, y_train)
            if clf.best_score_ > bestScore:
                bestScore = clf.best_score_
                bestParams = params
                bestClf = clf

        data = self.calculateMatrix(converter, bestParams, X)
        X_test, y_test = self._getXy(data, y, test_indices)

        y_pred = bestClf.predict(X_test)
        print(bestParams)
        return accuracy_score(y_test, y_pred)

    def doGridSearch(self, experiment: Dict[str, int], X: List[List[int]], y: List[int]):
        svc = SVC(kernel="linear", max_iter=3000)
        clf = GridSearchCV(svc, self.getParams(), scoring='accuracy', cv=3, n_jobs=6)
        clf.fit(X, y)

        paramAttempt = clf.best_params_
        paramAttempt.update(experiment)
        return paramAttempt, clf

    def _getXy(self, X, y, indices):
        return [X[index] for index in indices], [y[index] for index in indices]

    def calculateMatrix(self, converter: str, params: Dict[str, int], data: List[Graph]):
        paramK = params.get("k", 4)
        paramNorm = params.get("normalize", False)
        if converter == "matrix":
            paramOffset = params.get("offset", 0)
            paramRange = params.get("range", 1)
            conv = MatrixConverter(paramK, paramOffset, paramRange)
        elif converter == "label":
            conv = LabelConverter(paramK)
        elif converter == "unique_label":
            conv = UniqueLabelConverter(paramK)
        else:
            conv = UnlabeledLabelConverter(paramK)

        return conv.convertSet(data, paramNorm)

    def getParams(self) -> Dict[str, List[int]]:
        # return {'C': [2 ** i for i in [1, 3, 5, 8, 12]]}
        return {'C': [2 ** i for i in [-12, -8, -5, -3, -1, 1, 3, 5, 8, 12]]}
        # return {'C': [2 ** i for i in [-8, -3, 1, 5, 8]]}
