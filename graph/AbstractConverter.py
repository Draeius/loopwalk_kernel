from typing import List, Dict

from grakel import Graph
from scipy.sparse import lil_matrix
from sklearn import preprocessing


class AbstractConverter:
    _k = 1

    def getK(self) -> int:
        return self._k

    def setK(self, k: int):
        if k < 1:
            self._k = 1
        else:
            self._k = k

    k = property(getK, setK)

    def convertSet(self, dSet: List[Graph], normalize=True):
        converted = []
        for x in dSet:
            converted.append(self.convert(x))
        converted = self._rebase(converted)

        if normalize:
            return preprocessing.normalize(converted)
        return converted

    def convert(self, graph: Graph) -> Dict[str, int]:
        pass

    def _rebase(self, dSet: List) -> List[int]:
        # determine vector dimensions
        base = {}
        for x in dSet:
            base.update(x)

        # create 0 element
        for element in base:
            base[element] = 0
        # update all elements
        for index in range(0, len(dSet)):
            dSet[index] = list({**base, **dSet[index]}.values())
        return dSet
