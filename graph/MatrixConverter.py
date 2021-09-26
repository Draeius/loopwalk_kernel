from collections import Counter
from math import floor
from typing import Dict

from grakel import Graph

from graph.AbstractConverter import AbstractConverter
import numpy as np

from graph.GraphLoader import MatrixBuilder


class MatrixConverter(AbstractConverter):
    _matrixBuilder = MatrixBuilder()

    def __init__(self, k: int, deltaOffset: int, deltaRange: int):
        self.k = k
        self._deltaOffset = deltaOffset
        self._deltaRange = deltaRange

    def convert(self, graph: Graph) -> Dict[str, int]:
        counter = Counter()
        adj = self._matrixBuilder.getAdjacencyMatrix(graph)
        for i in range(1, self._k):
            diagonal = np.diag(adj)
            counter.update(
                [
                    str(i) + ";" + str(floor((value + self._deltaOffset) / self._deltaRange))
                    for value in diagonal
                    if value != 0
                ])
            adj = np.matmul(adj, adj)

        if len(counter.values()) == 0:
            counter.update(['none'])
        return dict(counter)

    def getOffset(self):
        return self._deltaOffset

    def getRange(self):
        return self._deltaRange
