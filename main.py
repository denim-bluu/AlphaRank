from src.config import Configuration
from src.pipeline import MainPipeline

config = Configuration()

pipeline = MainPipeline(config)
pm_scores = pipeline.run()
print(pm_scores.collect())
