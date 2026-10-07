import sys
from typing import Dict, Type, Any

from scipy.spatial import distance

class SimilaritySearchRegistry:
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, key: str, value: Type[Any]) -> None:
        cls._registry[key] = value

    @classmethod
    def get_registered(cls, key: str) -> Type[Any]:
        try:
            return cls._registry[key]
        except KeyError:
            raise ValueError(f"{key} not found in registry.")


### Registering Preprocessors ###
SimilaritySearchRegistry.register("cosine", distance.cosine)
SimilaritySearchRegistry.register("jaccard", distance.jaccard)
SimilaritySearchRegistry.register("hamming", distance.hamming)
SimilaritySearchRegistry.register("euclidean", distance.euclidean)
SimilaritySearchRegistry.register("minkowski", distance.minkowski)