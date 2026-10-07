import sys
from typing import List

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import VectorizerConfig
from src.pipelines.vectorizer.strategy import VectorizerStrategy


class BaseVectorizer:
    def __init__(self, config: VectorizerConfig, strategy: VectorizerStrategy):
        self.config = config
        self.strategy = strategy

    def vectorize(self, corpus: List[str]) -> np.ndarray:
        if self.config.mode == "train":
            self.strategy.fit(corpus)
            return self.strategy.transform(corpus)
        
        elif self.config.mode == "inference":
            return self.strategy.transform(corpus)
        
        else:
            raise ValueError("Invalid mode. The mode should be either 'train' or 'inference'.")


if __name__ == "__main__":
    from src.pipelines.config.default import VECTORIZER_DEFAULT_CONFIG
    from src.pipelines.vectorizer.factory import VectorizerFactory
    
    vectorizer = VectorizerFactory.create(VECTORIZER_DEFAULT_CONFIG)
    res = vectorizer.vectorize(["hello world", "goodbye world"])
    
    print(res)
