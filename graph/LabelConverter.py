from collections import Counter
from typing import List, Dict

from grakel import Graph

from graph.AbstractConverter import AbstractConverter


class LabelConverter(AbstractConverter):

    def __init__(self, k):
        self.k = k

    def convert(self, graph: Graph) -> Dict[str, int]:
        featureVector = self._convertToStrings(graph)
        return dict(featureVector)

    def _convertToStrings(self, graph: Graph) -> Counter:
        edges = graph.get_edge_dictionary()
        labels = graph.get_labels(purpose="dictionary")
        features = Counter()
        for node in edges:
            walks = self._convertNode(node, node, edges, labels, "", self.k+1, [])
            features.update(walks)
        return features

    def _convertNode(self, origin: int, node: int, edges: Dict[int, List[int]], labels: List[int], path: str,
                    maxLength, walks: List[str]) -> List[str]:
        pass

    def convertNode(self, origin: int, node: int, edges: Dict[int, List[int]], labels: List[int], path: str,
                     maxLength, walks: List[str]) -> List[str]:
        length = len(path)
        if len(path) < maxLength * 2:
            nextNodes = edges[node]
            if length > 2 and node == origin:
                walks.append(path)
            for nextNode in nextNodes:
                self.convertNode(origin, nextNode, edges, labels, f"{path}{labels[node]:02d}", maxLength, walks)

        return walks
