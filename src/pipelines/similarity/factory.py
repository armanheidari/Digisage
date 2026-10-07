import sys

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.similarity.base import BaseSimilaritySearch
from src.pipelines.config.config import SimilaritySearchConfig
from src.pipelines.similarity.strategy import KNNSimilarityStrategy
from src.pipelines.similarity.registry import SimilaritySearchRegistry


class SimilaritySearchFactory:
    @classmethod
    def create(cls, config: SimilaritySearchConfig) -> BaseSimilaritySearch:
        config.metrics_param["metric"] = SimilaritySearchRegistry.get_registered(config.metrics)
        strategy = KNNSimilarityStrategy(config.metrics_param)
        
        return BaseSimilaritySearch(config, strategy)
        