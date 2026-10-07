import sys
from typing import Dict, Type, Any

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.vectorizer.strategy import TfidfVectorizerStrategy, CountVectorizerStrategy, SentenceTransformerStrategy


class VectorizerRegistry:
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


### Registering Vectorizers ###
VectorizerRegistry.register("bow", CountVectorizerStrategy)
VectorizerRegistry.register("tf-idf", TfidfVectorizerStrategy)
VectorizerRegistry.register("sentence-transformer", SentenceTransformerStrategy)