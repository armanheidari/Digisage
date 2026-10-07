import sys
from typing import List

import pandas as pd

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import PreprocessorConfig
from src.pipelines.preprocessor.components import PreprocessorComponents


class BasePreprocessor: 
    def __init__(self, config: PreprocessorConfig, steps: List[PreprocessorComponents]):
        self.steps = steps
        self.config = config

    def process(self, dataset: pd.Series) -> pd.Series:
        processed_dataset = dataset.copy()
        
        for index, query in enumerate(dataset):
            preprocessed_query = query
            
            for step in self.steps:
                preprocessed_query = step.process(preprocessed_query)
            
            if isinstance(preprocessed_query, list):
                processed_dataset[index] = " ".join(preprocessed_query)
            else:
                processed_dataset[index] = preprocessed_query
                
        return processed_dataset
                
                
if __name__ == "__main__":
    from src.pipelines.config.default import PREPROCESSOR_DEFAULT_CONFIG
    from src.pipelines.preprocessor.factory import PreprocessorFactory
    
    preprocessor = PreprocessorFactory.create(PREPROCESSOR_DEFAULT_CONFIG)
    res = preprocessor.process(pd.read_csv("../../../data/digikala_faq.csv")["question"])

    pd.DataFrame(res).to_csv("../../../data/preprocessed_digikala_faq.csv", index=False)