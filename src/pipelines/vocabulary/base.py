import os
import sys
from collections import Counter
from typing import List, Dict

import pandas as pd

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.preprocessor.components import TokenizerComponents
from src.pipelines.config.config import VocabularyConfig


class VocabularyBuilder:
    def __init__(self, config: VocabularyConfig, tokenizer: TokenizerComponents):
        self.config = config
        self.tokenizer = tokenizer

    def build(self, documents: pd.Series):
        corpus = documents.to_list()
        tokenized_corpus = self._get_tokenized_corpus(corpus)
        word_counts = Counter(tokenized_corpus)
        
        if self.config.save:
            self._save_vocab(word_counts)
        
        return corpus
     
    def _get_tokenized_corpus(self, corpus: list) -> List[str]:
        tokenized_corpus = []
        for i in range(len(corpus)):
            tokens = self.tokenizer.process(corpus[i])
            for token in tokens:
                tokenized_corpus.append(token)
                
        return tokenized_corpus
    
    def _save_vocab(self, vocab: Dict[str, int]):
        with open(self.config.save_path, "w", encoding="utf-8") as f:
            for word, count in vocab.most_common():
                f.write(f"{word}: {count}\n")