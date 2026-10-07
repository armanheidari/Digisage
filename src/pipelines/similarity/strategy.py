from typing import Dict, Any, List
from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimilaritySearchStrategy(ABC):
    @abstractmethod
    def search(self, query_vector: np.ndarray, corpus_vectors: np.ndarray, threshold: float) -> List[List[int]]:
        raise NotImplementedError("The method should be implemented in the subclass.")


class KNNSimilarityStrategy(SimilaritySearchStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.nn = NearestNeighbors(**config)
        
    def search(self, query_vector: np.ndarray, corpus_vectors: np.ndarray, threshold: float) -> List[List[int]]:
        self.nn.fit(corpus_vectors)
        distances, indices = self.nn.kneighbors(query_vector, return_distance=True)
        
        if threshold is None:
            return indices.tolist()

        else:
            filtered_indices = []
            for dist, idx in zip(distances, indices):
                filtered_idx = [i for d, i in zip(dist, idx) if d >= threshold]
                filtered_indices.append(filtered_idx)
            return filtered_indices