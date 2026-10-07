import sys
import json

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.generate.augmenter.replacer import Replacer
from src.generate.augmenter.base import BaseAugmenter
from src.generate.config.config import AugmenterConfig
from src.generate.augmenter.paraphraser import Paraphraser


class AugmenterFactory:
    @classmethod
    def create(cls, config: AugmenterConfig) -> BaseAugmenter:
        paraphraser, replacer = None, None
        
        if config.paraphrase:
            paraphraser = Paraphraser(config.paraphraser_config)
        
        if config.replacer:
            with open(config.replacer_config.synonyms_file, 'r', encoding='utf-8') as file:
                synonyms = json.load(file)
                     
            replacer = Replacer(config.replacer_config, synonyms)
        
        return BaseAugmenter(config, paraphraser, replacer)