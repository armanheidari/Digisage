import sys

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import VectorizerConfig
from src.pipelines.vectorizer.base import BaseVectorizer
from src.pipelines.vectorizer.registry import VectorizerRegistry


class VectorizerFactory:
    @classmethod
    def create(cls, config: VectorizerConfig) -> BaseVectorizer:
        strategy = VectorizerRegistry.get_registered(config.vectorizer)
        strategy = strategy(config.vectorizer_param)

        return BaseVectorizer(config, strategy)