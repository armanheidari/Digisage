import sys
from typing import List

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.preprocessor.components import (
    StemmerComponents, 
    TokenizerComponents,
    NormalizerComponents, 
    LemmatizerComponents, 
    PreprocessorComponents,
    StopwordsRemoverComponents,
    InformalNormalizerComponents, 
)
from src.pipelines.config.config import PreprocessorConfig
from src.pipelines.preprocessor.base import BasePreprocessor
from src.pipelines.preprocessor.registry import PreprocessorRegistry


class PreprocessorFactory:
    @classmethod
    def create(cls, config: PreprocessorConfig) -> BasePreprocessor:
        steps = cls._initialize_steps(config)
        return BasePreprocessor(config, steps)
    
    @staticmethod
    def _initialize_steps(config: PreprocessorConfig) -> List[PreprocessorComponents]:
        steps = []
        
        if config.informal_normalizer:
            informal_normalizer = PreprocessorRegistry.get_registered("informal_normalizer")(**config.informal_normalizer_param)
            steps.append(InformalNormalizerComponents(informal_normalizer))
            
        if config.normalizer:
            normalizer = PreprocessorRegistry.get_registered("normalizer")(**config.normalizer_param)
            steps.append(NormalizerComponents(normalizer))

        word_tokenizer = PreprocessorRegistry.get_registered("tokenizer")(**config.tokenizer_param)
        steps.append(TokenizerComponents(word_tokenizer))
        
        if config.stopwords:
            stopwords_list = []

            with open(config.stopwords_file, 'r', encoding='utf-8') as file:
                stopwords_list = [line.strip() for line in file.readlines()]
        
            steps.append(StopwordsRemoverComponents(stopwords_list))
        
        if config.stemmer:
            stemmer = PreprocessorRegistry.get_registered("stemmer")()
            steps.append(StemmerComponents(stemmer))
            
        if config.lemmatizer:
            lemmatizer = PreprocessorRegistry.get_registered("lemmatizer")(**config.lemmatizer_param)
            steps.append(LemmatizerComponents(lemmatizer))
            
        return steps