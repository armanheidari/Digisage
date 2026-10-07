import os
import tempfile
import pickle
from typing import Literal, Dict, Any, List, Type

import mlflow
from functools import lru_cache
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


class MlFlowLogger:
    """MlFlowLogger class for logging machine learning experiments using MLflow."""
    
    mlflow.set_tracking_uri('http://localhost:9437')
    mlflow.enable_system_metrics_logging()
    mlflow_client = MlflowClient()

    def __init__(self, run_name: str, experiment_name: str, mode: Literal["production", "development"]):
        self.mode = mode
        self.run_name = run_name
        self.experiment_id = None
        self.experiment_name = experiment_name

        self.set_experiment_id()

    def set_experiment_id(self) -> str:
        """Set the experiment ID for the current experiment."""
        experiment = self.mlflow_client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
    def start_run(self):
        """Start a new MLflow run if one is not already active."""
        if not mlflow.active_run():
            run = self.mlflow_client.create_run(
                run_name=self.run_name,
                experiment_id=self.experiment_id,
                tags={"mode": self.mode}
            )
            mlflow.start_run(run_id=run.info.run_id)

    @classmethod
    def start_nested_run(cls, run_name: str, experiment_id: int):
        """Start a nested run within the current run."""
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")
        
        run = mlflow.active_run()
        child_run = cls.mlflow_client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.parentRunId": run.info.run_id,  
                "mlflow.runName": f"Trial {run_name}"
            }
        )
        return child_run
    
    @classmethod  
    def end_run(cls, run_id):
        """End a run with the given run ID."""
        cls.mlflow_client.set_terminated(run_id)
        
    @staticmethod
    @lru_cache   
    def get_run_id(run_name: str):
        """Get the run ID for the given run name."""
        #FIXME: This is not optimal way of searching, need to find a better way.
        
        runs = mlflow.search_runs(search_all_experiments=True)
        runs = runs[runs['tags.mlflow.runName'] == run_name]
        
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            return run_id
        else:
            raise ValueError("Run not found in the MLflow registry.")

    @classmethod
    def log_parameters(cls, params: Dict[str, Any]):
        """Log parameters for the current run."""        
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")
            
        mlflow.log_params(params)
        
    @classmethod
    def log_batch_metrics(cls, metrics: List[Metric]):
        """Log a batch of metrics for the current run."""
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")
        
        run = mlflow.active_run()
        cls.mlflow_client.log_batch(
            run_id=run.info.run_id, 
            metrics=metrics
        )
        
    @classmethod
    def log_figure(cls, figure: Any, artifact_file: str):
        """Log a figure as an artifact for the current run."""
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")
        
        run = mlflow.active_run()
        cls.mlflow_client.log_figure(
            run_id=run.info.run_id, 
            figure=figure, 
            artifact_file=artifact_file
        )
        
    @classmethod
    def log_table(cls, table: Any, artifact_file: str):
        """Log a table as an artifact for the current run."""
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")
        
        run = mlflow.active_run()
        cls.mlflow_client.log_table(
            run_id=run.info.run_id, 
            data=table, 
            artifact_file=artifact_file
        ) 

    @classmethod
    def log_artifact(cls, file: Any, file_name: str):
        """Log an artifact as an artifact for the current run."""
        if not mlflow.active_run():
            raise ValueError("Logger has not been started yet.")

        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, file_name)

        with open(local_path, "wb") as f:
            pickle.dump(file, f)
        
        run = mlflow.active_run()
        cls.mlflow_client.log_artifact(
            run_id=run.info.run_id, 
            local_path=local_path, 
            artifact_path=file_name.split(".")[0]
        )

        os.remove(local_path)
        
        
    @classmethod
    def download_artifact(cls, run_name: str, artifact_path: str) -> Type[Any]:
        """Download an artifact from the given run and artifact path."""
        run_id = cls.get_run_id(run_name)

        file_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path=artifact_path
        )
        
        with open(file_path, 'rb') as f:
            file_object = pickle.load(f)
            
        return file_object
