from src.config import Configuration, instantiate_orchestrator

config = Configuration()

orchestrator = instantiate_orchestrator(config)
orchestrator.run_analysis(config.selected_metrics, config.metric_weights)
