import sys
import asyncio
from typing import List

import torch
import pandas as pd
from alive_progress import alive_bar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.generate.config.config import ParaphraserConfig
from src.generate.augmenter.translator import Translator


class Paraphraser:
    def __init__(self, config: ParaphraserConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(config.device)

    async def _augment_query(self, text) -> List[str]:
        translated_text = await Translator.translate(text, self.config.source_langugae, self.config.destination_language)
        paraphrased_sentences = self._paraphrase(translated_text)

        translated_sentences = await asyncio.gather(
            *[Translator.translate(sentence, self.config.destination_language, self.config.source_langugae) for sentence in paraphrased_sentences]
        )
        
        return translated_sentences
    
    def _tokenize(self, inputs: str) -> torch.Tensor:
        tokenized_inputs = self.tokenizer(
            f'paraphrase: {inputs}',
            truncation=True,
            padding="longest",
            return_tensors="pt", 
            max_length=self.config.max_length,
        )
        
        tokenized_inputs = tokenized_inputs.input_ids.to(self.config.device)  
        return tokenized_inputs
        
    def _paraphrase(self, inputs: str) -> str:
        tokenized_inputs = self._tokenize(inputs)
        
        encoded_outputs = self.model.generate(
            tokenized_inputs, 
            temperature=self.config.temperature, 
            repetition_penalty=self.config.repetition_penalty,
            num_return_sequences=self.config.num_return_sequences, 
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            num_beams=self.config.num_beams, 
            num_beam_groups=self.config.num_beam_groups,
            max_length=self.config.max_length, 
            diversity_penalty=self.config.diversity_penalty
        )

        decoded_outputs = self.tokenizer.batch_decode(
            sequences=encoded_outputs, 
            skip_special_tokens=True
        )

        return decoded_outputs
    
    async def augment(self, dataset: pd.DataFrame) -> pd.DataFrame:
        augmented_rows = []

        with alive_bar(dataset.shape[0]) as bar:   
            for _, row in dataset.iterrows():
                question = row["question"]
                paraphrased_questions = await self._augment_query(question)

                for pq in paraphrased_questions:
                    augmented_rows.append({"question": pq, "answer": row["answer"], "category": row["category"]})
                    
                bar()
                
        augmented_df = pd.DataFrame(augmented_rows)
        return augmented_df


if __name__ == "__main__":
    from src.generate.config.default import PARAPHRASER_DEFAULT_CONFIG
    
    paraphraser = Paraphraser(PARAPHRASER_DEFAULT_CONFIG)
    print(paraphraser.paraphrase("Hello, how are you?"))