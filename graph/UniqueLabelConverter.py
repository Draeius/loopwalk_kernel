from collections import Counter
from typing import List, Dict

from grakel import Graph

from graph.LabelConverter import LabelConverter


class UniqueLabelConverter(LabelConverter):

    def __init__(self, k):
        super().__init__(k)
        self.k = k

    def _convertNode(self, origin: int, node: int, edges: Dict[int, List[int]], labels: List[str], path: str,
                     maxLength, walks: List[str]) -> List[str]:
        length = len(path)
        if len(path) < maxLength * 2:
            nextNodes = edges[node]
            if length > 2 and node == origin:
                walks.append(path)
            for nextNode in nextNodes:
                self._convertNode(origin, nextNode, edges, labels, f"{path}{node:02d}", maxLength, walks)

        return walks
