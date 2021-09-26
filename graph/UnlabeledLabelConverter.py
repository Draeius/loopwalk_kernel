from collections import Counter
from typing import Dict

from grakel import Graph

from graph.AbstractConverter import AbstractConverter
from graph.GraphLoader import MatrixBuilder
import numpy as np


class UnlabeledLabelConverter(AbstractConverter):
    # A special converter, that applies the label kernel while ignoring labels
    _matrixBuilder = MatrixBuilder()

    def __init__(self, k):
        self.k = k

    def convert(self, graph: Graph) -> Dict[str, int]:
        adj = self._matrixBuilder.getAdjacencyMatrix(graph)
        counter = {}
        for i in range(1, self._k):
            diagonal = np.diag(adj)
            counter[i] = sum(diagonal)
            adj = np.matmul(adj, adj)

        if len(counter) == 0:
            counter['none'] = 1
        return dict(counter)
