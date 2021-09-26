import math
from collections import Counter

import numpy as np
from typing import Dict

from grakel import Graph
from grakel.datasets import fetch_dataset


class DatasetLoader:

    @staticmethod
    def getDataset(name: str):
        dataset = fetch_dataset(name, verbose=False, as_graphs=True)
        return dataset

    @staticmethod
    def getXSplit(dSet, target, split=.8):
        return {
            "data": dSet[:math.floor(len(dSet) * split)],
            "target": target[:math.floor(len(dSet) * split)]
        }

    @staticmethod
    def getYSplit(dSet, target, split=.8):
        return {
            "data": dSet[math.floor(len(dSet) * split):],
            "target": target[math.floor(len(dSet) * split):]
        }


class MatrixBuilder:

    def getAdjacencyMatrix(self, graph: Graph):
        return graph.get_adjacency_matrix()

    def getLabelMatrix(self, graph: Graph):
        labels = graph.get_labels(purpose="adjacency")
        matrix = np.zeros((max(list(Counter(labels.values()))) + 1, len(labels)))

        for node in labels:
            matrix[labels[node]][node] = 1

        return matrix
