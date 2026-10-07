import os
import sys
from dotenv import load_dotenv

import pandas as pd

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.config.config import (
    PipelineConfig,
    EvaluatorConfig,
    VectorizerConfig, 
    VocabularyConfig, 
    PreprocessorConfig,
    SimilaritySearchConfig
)


load_dotenv()
training_dataset = pd.read_csv(path_manager.get_custom_path("data/augmented_dataset.csv", based="base"))
evaluation_dataset = pd.read_csv(path_manager.get_custom_path("data/test_digikala_faq.csv", based="base"))  
    
tokenizer_param = {
    "join_verb_parts": False,
    "join_abbreviations": False,
    "separate_emoji": True,
    "replace_links": True,
    "replace_ids": True,
    "replace_numbers": True,
    "replace_hashtags": True
}

normalizer_param = {
    "correct_spacing": False,
    "remove_diacritics": True,
    "remove_specials_chars": True,
    "decrease_repeated_chars": True,
    "persian_style": True,
    "persian_numbers": True,
    "unicodes_replacement": True,
    "seperate_mi": False
}

informal_normalizer_param = {
    "seperation_flag": False
}

lemmatizer_param = {
    "joined_verb_parts": True
}

metrics_param ={
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
    "n_jobs": -1
}

sentence_transformer_param = {
    "model_name_or_path": "sentence-transformers/LaBSE",
    # "model_name_or_path": 'sentence-transformers/all-MiniLM-L6-v2',
    # "model_name_or_path": 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    "token": os.getenv('TOKEN') 
}

tf_idf_param = {
    "max_features": 10000,
    # "ngram_range": (2, 2),
    "max_df": 0.9,
    # "min_df": 0.01
}


PREPROCESSOR_DEFAULT_CONFIG = PreprocessorConfig(
    tokenizer_param=tokenizer_param,
    informal_normalizer=False,
    normalizer=False,
    stemmer=False, 
    lemmatizer=False,
    stopwords=True,
    informal_normalizer_param=informal_normalizer_param,
    normalizer_param=normalizer_param,
    lemmatizer_param=lemmatizer_param,
    stopwords_file=path_manager.get_custom_path("data/stops.txt", based="base")
)


VECTORIZER_DEFAULT_CONFIG = VectorizerConfig(
    vectorizer="tf-idf", 
    vectorizer_param=tf_idf_param
)


SIMILARITY_SEARCH_DEFAULT_CONFIG = SimilaritySearchConfig(
    metrics="cosine",
    metrics_param={
        "n_neighbors": 5,
    }, 
    # threshold=0.5
)


VOCABULARY_DEFAULT_CONFIG = VocabularyConfig(
    save=True,
    save_path=path_manager.get_custom_path("data/vocabulary.txt", based="base")
)

EVALUATOR_DEFAULT_CONFIG = EvaluatorConfig(
    k=1,
    training_dataset=training_dataset, 
    evaluation_dataset=evaluation_dataset
)


PIPELINE_DEFAULT_CONFIG = PipelineConfig(
    preprocessor_config=PREPROCESSOR_DEFAULT_CONFIG,
    vectorizer_config=VECTORIZER_DEFAULT_CONFIG,
    similarity_config=SIMILARITY_SEARCH_DEFAULT_CONFIG,
    vocabulary_config=VOCABULARY_DEFAULT_CONFIG, 
    evaluator_config=EVALUATOR_DEFAULT_CONFIG,
    training_dataset=training_dataset, 
    evaluation_dataset=evaluation_dataset
)