from typing import List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class VectorizerStrategy(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        raise NotImplementedError("The method should be implemented in the subclass.")
    
    @abstractmethod
    def transform(self, corpus: List[str]) -> np.ndarray:
        raise NotImplementedError("The method should be implemented in the subclass.")

    
class TfidfVectorizerStrategy(VectorizerStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.vectorizer = TfidfVectorizer(**config)
        
    def fit(self, corpus: List[str]):
         self.vectorizer.fit(corpus)
    
    def transform(self, corpus: List[str]) -> np.ndarray:
        vectorized_corpus = self.vectorizer.transform(corpus)
        return vectorized_corpus.toarray()
    

class CountVectorizerStrategy(VectorizerStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.vectorizer = CountVectorizer(**config)
        
    def fit(self, corpus: List[str]):
         self.vectorizer.fit(corpus)
    
    def transform(self, corpus: List[str]) -> np.ndarray:
        vectorized_corpus = self.vectorizer.transform(corpus)
        return vectorized_corpus.toarray()
    
    
class SentenceTransformerStrategy(VectorizerStrategy):
    def __init__(self, config: Dict[str, Any]):
        self.vectorizer = SentenceTransformer(**config)
        
    def fit(self, corpus: List[str]):
         ...
    
    def transform(self, corpus: List[str]) -> np.ndarray:
        vectorized_corpus = self.vectorizer.encode(corpus)
        return vectorized_corpus