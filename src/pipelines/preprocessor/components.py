from typing import List, Tuple
from abc import ABC, abstractmethod
from hazm import Normalizer, Stemmer, Lemmatizer, InformalNormalizer, WordTokenizer


class PreprocessorComponents(ABC):
    @abstractmethod
    def process(self, text: str) -> Tuple[str, List[str]]:
        raise NotImplementedError("The method should be implemented in the subclass.")


class NormalizerComponents(PreprocessorComponents):
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def process(self, text: str) -> str:
        return self.normalizer.normalize(text)
    

class StemmerComponents(PreprocessorComponents):
    def __init__(self, stemmer: Stemmer):
        self.stemmer = stemmer

    def process(self, tokens: List[str]) -> str:
        stemmed_text = [self.stemmer.stem(token) for token in tokens]
        return " ".join(stemmed_text)
    

class LemmatizerComponents(PreprocessorComponents):
    def __init__(self, lemmatizer: Lemmatizer):
        self.lemmatizer = lemmatizer

    def process(self, tokens: List[str]) -> str:
        lemmatized_text = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_text)


class TokenizerComponents(PreprocessorComponents):
    def __init__(self, word_tokenizer: WordTokenizer):
        self.word_tokenizer = word_tokenizer

    def process(self, text: str) -> List[str]:
        return self.word_tokenizer.tokenize(text)


class InformalNormalizerComponents(PreprocessorComponents):
    def __init__(self, informal_normalizer: InformalNormalizer):
        self.informal_normalizer = informal_normalizer

    def process(self, text: str) -> str:
        informal_normalized_text = self.informal_normalizer.normalize(text)
        informal_normalized_text = [i[0] for i in informal_normalized_text[0]]
        return " ".join(informal_normalized_text)


class StopwordsRemoverComponents(PreprocessorComponents):
    def __init__(self, stopwords_list: List[str]):
        self.stopwords_list = set(stopwords_list)

    def process(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords_list]