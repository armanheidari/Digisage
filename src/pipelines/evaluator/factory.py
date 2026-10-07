import sys

import pandas as pd

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.evaluator.base import BaseEvaluator
from src.pipelines.config.config import EvaluatorConfig


class EvaluatorFactory: 
    @classmethod
    def create(self, config: EvaluatorConfig) -> BaseEvaluator:
        return BaseEvaluator(config)