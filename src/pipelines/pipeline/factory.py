import sys

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.type_hint import ControllerType
from src.pipelines.pipeline.base import BasePipeline
from src.pipelines.config.config import PipelineConfig
from src.pipelines.pipeline.registry import PipelineRegistry


class PipelineFactory:
    @staticmethod    
    def create(controller_type: ControllerType, config: PipelineConfig) -> BasePipeline:
        pipeline_cls = PipelineRegistry.get_registered(controller_type)

        if not pipeline_cls:
            raise ValueError(f"Unknown pipeline type: {controller_type}")
        return pipeline_cls(config)
