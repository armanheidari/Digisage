import sys
from typing import List

import numpy as np
from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import SimilaritySearchConfig
from src.pipelines.similarity.strategy import SimilaritySearchStrategy


class BaseSimilaritySearch:
    def __init__(self, config: SimilaritySearchConfig, strategy: SimilaritySearchStrategy):
        self.config = config
        self.strategy = strategy

    def search(self, query_vector: np.ndarray, corpus_vectors: np.ndarray) -> List[List[int]]:        
        neighbor_indices = self.strategy.search(query_vector, corpus_vectors, self.config.threshold)
        return neighbor_indices