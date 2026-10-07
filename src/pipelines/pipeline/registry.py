from typing import Dict, Type, Any


class PipelineRegistry:
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(pipeline_cls: Type[Any]):
            cls._registry[key] = pipeline_cls
            return pipeline_cls
        return decorator

    @classmethod
    def get_registered(cls, key: str) -> Type[Any]:
        pipeline_cls = cls._registry.get(key)
        if not pipeline_cls:
            raise ValueError(f"Pipeline '{key}' not found in registry.")
        return pipeline_cls