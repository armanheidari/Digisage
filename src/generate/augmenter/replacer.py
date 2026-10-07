import sys
import random
from typing import Dict, List

import pandas as pd
from alive_progress import alive_bar

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.generate.config.config import ReplacorConfig


class Replacer:
    def __init__(self, config: ReplacorConfig, synonyms: Dict[str, List[str]]):
        self.config = config
        self.synonyms = synonyms

    def _augment_text(self, text: str):
        words = text.split()
        augmented_texts = set()
        
        for _ in range(self.config.num_augmentations):
            new_words = words[:]
            replaced = False
            for i, word in enumerate(new_words):
                if word in self.synonyms and random.random() < self.config.probability:
                    new_words[i] = random.choice(self.synonyms[word])
                    replaced = True
            
            if replaced:
                augmented_texts.add(" ".join(new_words))
        
        return list(augmented_texts)

    def replace(self, dataset: pd.DataFrame) -> pd.DataFrame:
        augmented_rows = []
        
        with alive_bar(dataset.shape[0]) as bar:   
            for _, row in dataset.iterrows():
                question = row["question"]
                answer = row["answer"]
                category = row["category"]
                
                augmented_questions = self._augment_text(question)
                
                for augmented_question in augmented_questions:
                    augmented_rows.append({"question": augmented_question, "answer": answer, "category": category})
                    
                bar()

        augmented_df = pd.DataFrame(augmented_rows)
        return pd.concat([dataset, augmented_df], ignore_index=True)
