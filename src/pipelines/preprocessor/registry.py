import hazm
from typing import Dict, Type, Any


class PreprocessorRegistry:
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, key: str, value: Type[Any]) -> None:
        cls._registry[key] = value

    @classmethod
    def get_registered(cls, key: str) -> Type[Any]:
        try:
            return cls._registry[key]
        except KeyError:
            raise ValueError(f"{key} not found in registry.")


### Registering Preprocessors ###
PreprocessorRegistry.register("stemmer", hazm.Stemmer)
PreprocessorRegistry.register("normalizer", hazm.Normalizer)
PreprocessorRegistry.register("lemmatizer", hazm.Lemmatizer)
PreprocessorRegistry.register("tokenizer", hazm.WordTokenizer)
PreprocessorRegistry.register("informal_normalizer", hazm.InformalNormalizer)
