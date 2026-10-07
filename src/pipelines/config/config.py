from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Type

import pandas as pd


@dataclass
class PreprocessorConfig:
    tokenizer_param: Dict[str, Any]
    stopwords: Optional[bool] = False
    normalizer: Optional[bool] = False
    stemmer: Optional[bool] = False
    lemmatizer: Optional[bool] = False
    informal_normalizer: Optional[bool] = False
    normalizer_param: Optional[Dict[str, Any]] = None
    lemmatizer_param: Optional[Dict[str, Any]] = None
    informal_normalizer_param: Optional[Dict[str, Any]] = None
    stopwords_file: Optional[str] = None
    
    def __post_init__(self):
        if self.lemmatizer and self.stemmer:
            raise ValueError("Stemmer and Lemmatizer can not be used together.")


@dataclass
class VectorizerConfig:
    vectorizer: Literal["tf-idf", "bow", "sentence-transformer"]
    vectorizer_param: Dict[str, Any]
    mode: Optional[Literal["train", "inference"]] = None
    
    def __post_init__(self):
        if self.mode:
            raise ValueError("Mode can not be set in initialization.")

@dataclass
class SimilaritySearchConfig:
    metrics: Literal["jaccard", "cosine", "euclidean", "minowski"]
    metrics_param: Dict[str, Any]
    threshold: Optional[float] = None


@dataclass
class VocabularyConfig:
    save: Optional[bool] = False
    save_path: Optional[str] = None
    
    def __post_init__(self):
        if self.save and not self.save_path:
            raise ValueError("Save path should be provided.")


@dataclass
class EvaluatorConfig:
    k: int
    training_dataset: pd.DataFrame
    evaluation_dataset: pd.DataFrame
    logger: Optional[bool] = None
    
    def __pos_init__(self):
        if self.logger:
            raise ValueError("Logger can not be set in initialization.")


@dataclass
class PipelineComponents:
    preprocessor: Optional[Type[Any]] = None
    vectorizer: Optional[Type[Any]] = None
    similarity: Optional[Type[Any]] = None
    vocabulary: Optional[Type[Any]] = None
    evaluator: Optional[Type[Any]] = None
    
    
@dataclass
class PipelineConfig:
    preprocessor_config: PreprocessorConfig
    vectorizer_config: VectorizerConfig
    similarity_config: SimilaritySearchConfig
    vocabulary_config: VocabularyConfig
    evaluator_config: EvaluatorConfig
    training_dataset: Optional[pd.DataFrame] = None
    evaluation_dataset: Optional[pd.DataFrame] = None
    logger: Optional[bool] = None
    
    def __pos_init__(self):
        if self.logger:
            raise ValueError("Logger can not be set in initialization.")