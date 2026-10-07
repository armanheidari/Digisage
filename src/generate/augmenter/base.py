import sys
import asyncio
from typing import Optional

import pandas as pd

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.generate.augmenter.replacer import Replacer
from src.generate.config.config import AugmenterConfig
from src.generate.augmenter.paraphraser import Paraphraser


class BaseAugmenter:
    def __init__(self, config: AugmenterConfig, paraphraser: Paraphraser, replacer: Replacer):
        self.config = config
        self.replacer = replacer
        self.paraphraser = paraphraser
    
    async def augment(self, dataset: pd.DataFrame, save: Optional[bool] = False) -> pd.DataFrame:
        augmented_dataframe = dataset.copy()
        
        if self.paraphraser:
            augmented_dataframe = await self.paraphraser.augment(augmented_dataframe)
        
        if self.replacer:
            augmented_dataframe = self.replacer.replace(augmented_dataframe)
                
        augmented_dataframe.drop_duplicates(inplace=True, ignore_index=True)
        
        if save:
            augmented_dataframe.to_csv(self.config.save_path, index=False)
            
        return augmented_dataframe
    

if __name__ == "__main__":
    from src.generate.augmenter.factory import AugmenterFactory
    from src.generate.config.default import AUGMENTER_DEFAULT_CONFIG
    
    augmenter = AugmenterFactory.create(AUGMENTER_DEFAULT_CONFIG)
    asyncio.run(augmenter.augment(pd.read_csv(path_manager.get_custom_path("data/digikala_faq.csv", based="base"), index_col=0), save=True))