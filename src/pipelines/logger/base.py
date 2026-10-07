import sys
from functools import wraps
from typing import Dict, Callable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from mlflow.entities import Metric

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.logger.log import MlFlowLogger
from src.pipelines.evaluator.analysis import EvaluatorAnalysis


class BaseLogger(ABC):
    @classmethod
    @abstractmethod
    def observe(cls, func: Callable):
        raise NotImplementedError("Method should be implemented in child class")


class EvaluatorLogger(BaseLogger):
    @staticmethod
    def _log_metric(key: str, value: str):
        metric = Metric(key, value, timestamp=0, step=0)
        MlFlowLogger.log_batch_metrics([metric])
    
    @staticmethod
    def _log_report(report: Dict[str, Dict[str, float]]):
        MlFlowLogger.log_table(pd.DataFrame(report), "IR Report/report.json")
        
    @staticmethod
    def _log_figure(dataframe: pd.DataFrame, file_name: str):
        counts_df = dataframe["category"].value_counts().reset_index()
        counts_df.columns = ["category", "count"]
        fig = EvaluatorAnalysis.plot_bar_chart(counts_df, "category")
        MlFlowLogger.log_figure(fig, f"analysis/{file_name}_category.html")
        
    @classmethod
    def observe(cls, func: Callable):
        @wraps(func)
        def decorator(*args, **kwargs):
            metrics = func(*args, **kwargs)
            
            if args[0].config.logger:
                cls._log_report(metrics["classification_report"])
                
                cls._log_metric("MRR Score", metrics["mrr_score"])
                cls._log_metric(f"Recall {args[0].config.k} Score", metrics["recall_score"])
                cls._log_metric(f"Precision {args[0].config.k} Score", metrics["precision_score"])
                
                cls._log_figure(args[0].config.training_dataset, "training")
                cls._log_figure(args[0].config.evaluation_dataset, "evaluation")
                
            return metrics
        return decorator


class PipelineLogger(BaseLogger):
    @staticmethod
    def _log_vectorizer(*args):
        if args[0].vectorizer.config.vectorizer != "sentence-transformer":
            MlFlowLogger.log_artifact(args[0].vectorizer.strategy.vectorizer, "vectorizer.pkl") 
       
    @staticmethod
    def _log_corpus(vectorized_corpus: np.ndarray):
        MlFlowLogger.log_artifact(vectorized_corpus, "corpus.pkl")
        
    @staticmethod
    def _log_figure():
        with open(path_manager.get_custom_path("data/vocabulary.txt", based="base"), 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            word, count = line.strip().split(': ')
            data.append((word, int(count)))
            
        df = pd.DataFrame(data, columns=['word', 'count'])
        fig = EvaluatorAnalysis.plot_bar_chart(df, "word")
        MlFlowLogger.log_figure(fig, "analysis/vocabulary.html")

    @classmethod
    def observe(cls, func: Callable):
        @wraps(func)
        def decorator(*args, **kwargs):
            vectorized_corpus = func(*args, **kwargs)
            
            if args[0].config.logger:
                cls._log_vectorizer(*args)
                cls._log_corpus(vectorized_corpus)
                cls._log_figure()

            return vectorized_corpus
        return decorator
