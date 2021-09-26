from typing import Dict, List

from grakel import Graph

from graph.LabelConverter import LabelConverter
from graph.MatrixConverter import MatrixConverter
from graph.UnlabeledLabelConverter import UnlabeledLabelConverter


class ExperimentIterator:
    _switcher = {
        'label': LabelConverter,
        'matrix': MatrixConverter,
        'unlabeled': UnlabeledLabelConverter,
        'weisfeiler': WeisfeilerConverter,
        'randomWalk': RandomWalkConverter
    }

    def __init__(self, params: Dict[str, List]):
        self._params = params

    def iterate(self, X: List[Graph], y: List[int]):
        pass

    def _getConverter(self, params: Dict):
        converter = params['converter']
        if converter in self._switcher.keys():
            if self._switcher[converter] == MatrixConverter:
                return self._switcher[converter](params.get('k', 4), params.get('offset', 0), params.get('range', 1))
            return self._switcher[converter](params.get('k', 4))
