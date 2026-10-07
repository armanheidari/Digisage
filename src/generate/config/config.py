from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AugmenterConfig:
    paraphrase: bool
    replacer: bool
    paraphraser_config: Optional["ParaphraserConfig"] = None
    replacer_config: Optional["ReplacorConfig"] = None
    save_path: Optional[str] = None
    

@dataclass
class ParaphraserConfig:
    model_name: str
    device: Literal["cuda", "cpu"]
    max_length: int
    num_beams: int
    num_beam_groups: int
    num_return_sequences: int
    repetition_penalty: float
    diversity_penalty: float
    no_repeat_ngram_size: int
    temperature: float
    source_langugae: str
    destination_language: str
    

@dataclass
class ReplacorConfig:
    probability: float
    num_augmentations: int
    synonyms_file: str
