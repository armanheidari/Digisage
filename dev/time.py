import sys
import time

import pandas as pd
import matplotlib.pyplot as plt

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.pipelines.type_hint import ControllerType
from src.pipelines.pipeline.builder import PipelineBuilder
from src.pipelines.config.default import PIPELINE_DEFAULT_CONFIG


queries = ["خرید چگونه در اینجا است" * i for i in range(1, 21)]  

pipeline = (
    PipelineBuilder(controller_type=ControllerType.INFERENCE)
    .with_logger("digisage", "DigiSage", "development")
    .with_config(PIPELINE_DEFAULT_CONFIG)
    .with_preprocessor()
    .with_vectorizer()
    .with_vocabulary()
    .with_similarity()
    .with_evaluator()
    .build()
)

n = 100

results = []
for query in queries:
    total_time = 0
    for _ in range(n):
        start_time = time.time()
        
        result = pipeline.run(query)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
    average_time = total_time / n
    results.append({"query_length": len(query), "average_time": average_time})
    print(f"Query length: {len(query)}, Average Time: {average_time:.4f} seconds")


df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(df["query_length"], df["average_time"], marker='o')
plt.xlabel("Query Length")
plt.ylabel("Average Time (seconds)")
plt.title("Average Retrieval Time vs Query Length")
plt.grid(True)
plt.show()

plt.savefig("time_experiment.png")