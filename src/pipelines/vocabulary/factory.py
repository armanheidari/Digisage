import sys
from typing import Dict, Any

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import VocabularyConfig
from src.pipelines.vocabulary.base import VocabularyBuilder
from src.pipelines.preprocessor.registry import PreprocessorRegistry
from src.pipelines.preprocessor.components import TokenizerComponents


class VocabularyBuilderFactory:
    @classmethod
    def create(cls, config: VocabularyConfig, tokenizer_param: Dict[str, Any]) -> VocabularyBuilder:
        tokenizer = PreprocessorRegistry.get_registered("tokenizer")(**tokenizer_param)
        tokenizer_component = TokenizerComponents(tokenizer)
        
        return VocabularyBuilder(config, tokenizer_component)