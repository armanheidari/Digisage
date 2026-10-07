import sys
from typing import List, Dict, Any

from sklearn.metrics import classification_report

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.logger.base import EvaluatorLogger
from src.pipelines.config.config import EvaluatorConfig


class BaseEvaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
    
    @EvaluatorLogger.observe
    def evaluate(self, search_result: List[List[int]]) -> float:
        relevant_classes = self.config.evaluation_dataset["category"]
        report = self._get_classification_result(search_result, relevant_classes)
        mrr_score = self._get_mean_reciprocal_rank(search_result, relevant_classes)
        recall = self._calculate_recall_at_k(search_result, relevant_classes, self.config.k)
        precision = self._calculate_precision_at_k(search_result, relevant_classes, self.config.k)
        
        return {
            "mrr_score": mrr_score,
            "precision_score": precision,
            "recall_score": recall,
            "classification_report": report
        }
        
    def _get_classification_result(self, search_result: List[List[int]], relevant_classes: List[int]) -> Dict[str, Any]:   
        try:         
            first_neighbor_indices = [neighbors[0] for neighbors in search_result]
        except:
            return None
        
        retrieved_classes = self.config.training_dataset["category"][first_neighbor_indices]
        
        report = classification_report(relevant_classes, retrieved_classes, zero_division=1, output_dict=True)
        return report
        
    def _get_mean_reciprocal_rank(self, search_result: List[List[int]], relevant_classes: List[int]) -> float:
        reciprocal_ranks = []
        for query_idx, neighbors in enumerate(search_result):
            for rank, neighbor_idx in enumerate(neighbors):
                if self.config.training_dataset["category"][neighbor_idx] == relevant_classes[query_idx]:
                    reciprocal_ranks.append(1 / (rank + 1))
                    break
            else:
                reciprocal_ranks.append(0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def _calculate_precision_at_k(self, search_results: List[List[int]], relevant_classes: List[int], k: int):
        total_precision = 0.0
        num_samples = len(search_results)

        for retrieved_docs, relevant_docs in zip(search_results, relevant_classes):
            precision_at_k = self._get_precision_score(retrieved_docs, relevant_docs, k)
            total_precision += precision_at_k

        return total_precision / num_samples

    def _calculate_recall_at_k(self, search_results: List[List[int]], relevant_classes: List[int], k: int):
        total_recall = 0.0
        num_samples = len(search_results)

        for retrieved_docs, relevant_docs in zip(search_results, relevant_classes):
            recall_at_k = self._get_recall_score(retrieved_docs, relevant_docs, k)
            total_recall += recall_at_k

        return total_recall / num_samples

    def _get_precision_score(self, retrieved_docs: List[List[int]], relevant_docs: List[int], k: int):
        if k > len(retrieved_docs):
            k = len(retrieved_docs)
            
        if k == 0:
            return 0
        
        top_k_docs = self.config.training_dataset["category"][retrieved_docs[:k]]
        relevant_count = sum(1 for doc in top_k_docs if doc in relevant_docs)
        return relevant_count / k
    
    def _get_recall_score(self, retrieved_docs: List[List[int]], relevant_docs: List[int], k: int):
        if k > len(retrieved_docs):
            k = len(retrieved_docs)

        if k == 0:
            return 0
        
        top_k_docs = self.config.training_dataset["category"][retrieved_docs[:k]]
        relevant_count = sum(1 for doc in top_k_docs if doc in relevant_docs)
        return relevant_count / len(relevant_docs)