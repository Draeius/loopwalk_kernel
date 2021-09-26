import itertools
from enum import Enum

from graph.GraphLoader import DatasetLoader
from graph.GridSearch import GridSearch


class ExperimentConductor:
    sets = Enum('Sets', 'all small large reduced custom')

    datasets = {
        sets.all: ["BZR", "COX2", "DHFR", "MUTAG", "NCI1", "PTC_FR"],
        sets.small: ["BZR", "COX2", "DHFR", "MUTAG", "PTC_FR"],
        sets.large: ["NCI1"],
        sets.reduced: ["BZR", "COX2", "MUTAG"],
        sets.custom: ["DHFR", "PTC_FR"]
    }

    _currentSet = sets.all

    def setDatasets(self, newSet):
        if newSet in self.sets:
            self._currentSet = newSet

    # done
    def gridSearchMatrix(self):
        # need to check for NCI1 because dimension gets too big
        if self._currentSet == self.sets.small:
            searchMatrix = {'k': range(2, 10),
                            'offset': [0] + [2 ** i for i in [3, 5, 12]],
                            'range': [2 ** i for i in [0, 3, 5, 12]],
                            'normalized': [True, False]}
        else:
            searchMatrix = {'k': range(2, 7),
                            'offset': [0] + [2 ** i for i in [3, 5, 12]],
                            'range': [2 ** i for i in [0, 3, 5, 12]],
                            'normalized': [True, False]}

        self._doGridSearch('matrix', searchMatrix)

    # done
    def gridSearchLabel(self):
        # need to check for NCI1 because dimension gets too big
        if self._currentSet == self.sets.small:
            searchMatrix = {'k': range(2, 10), 'normalized': [True, False]}
        else:
            searchMatrix = {'k': range(2, 7), 'normalized': [True, False]}
        self._doGridSearch('label', searchMatrix)

    def gridSearchUniqueLabel(self):
        if self._currentSet == self.sets.small:
            searchMatrix = {'k': range(2, 10), 'normalized': [True, False]}
        else:
            searchMatrix = {'k': range(2, 7), 'normalized': [True, False]}
        self._doGridSearch('unique_label', searchMatrix)

    # done
    def staticKParamMatrix(self):
        # need to check for NCI1 because dimension gets too big
        if self._currentSet == self.sets.small:
            kList = range(2, 10)
            searchMatrix = {'offset': [0] + [2 ** i for i in [3, 5, 12]],
                            'range': [2 ** i for i in [0, 3, 5, 12]],
                            'normalized': [True, False]}
        else:
            kList = range(2, 7)
            searchMatrix = {'offset': [0] + [2 ** i for i in [3, 5, 12]],
                            'range': [2 ** i for i in [0, 3, 5, 12]],
                            'normalized': [True, False]}

        for k in kList:
            print(k)
            searchMatrix['k'] = [k]
            self._doGridSearch('matrix', searchMatrix)

    # done
    def staticKParamLabel(self):
        # need to check for NCI1 because dimension gets too big
        if self._currentSet == self.sets.small:
            kList = range(2, 10)
        else:
            kList = range(2, 7)
        for k in kList:
            print(k)
            self._doGridSearch('label', {"k": [k], 'normalized': [True, False]})

    # done
    def unlabeledLabelKernel(self):
        searchMatrix = {'k': range(2, 12), 'normalized': [True, False]}
        self._doGridSearch('unlabeled', searchMatrix)

    # done
    def dimensions(self):
        searchMatrix = {'k': range(2, 10),
                        'offset': [0] + [2 ** i for i in [3, 5, 12]],
                        'range': [2 ** i for i in [0, 3, 5, 12]]}
        gridSearch = GridSearch(searchMatrix)
        dl = DatasetLoader()
        for name in self._getDatasets():
            keys, values = zip(*searchMatrix.items())
            dataset = dl.getDataset(name)
            print(name)
            for v in itertools.product(*values):
                experiment = dict(zip(keys, v))
                print(experiment)
                print('matrix: ', gridSearch.calculateMatrix('matrix', experiment, dataset.data))
            for k in range(2, 10):
                print({'k': k})
                print('label: ', gridSearch.calculateMatrix('label', {'k': k}, dataset.data))

    def _doGridSearch(self, converter: str, params):
        gridSearch = GridSearch(params)
        for name in self._getDatasets():
            dl = DatasetLoader()
            dataset = dl.getDataset(name)
            print(name)
            print(converter + ': ', gridSearch.calculateAccuracies(dataset.data, dataset.target, converter))

    def _getDatasets(self):
        return self.datasets[self._currentSet]
