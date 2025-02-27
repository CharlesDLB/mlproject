import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, RunInfo
import mlflow.sklearn
import logging
from cdlb_logger.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO, stream=True, file=True, log_file_path="logs/global_logger.log")


# def get_or_create_experiment(experiment_name) -> Experiment:
#     """
#     Creates an mlflow experiment
#     :param experiment_name: str. The name of the experiment to be set in MLFlow
#     :return: the experiment created if it doesn't exist, experiment if it is already created.
#     """
#     try:
#         client = MlflowClient()
#         experiment: Experiment = client.get_experiment_by_name(name=experiment_name)
#         if experiment and experiment.lifecycle_stage != 'deleted':
#             return experiment
#         else:
#             experiment_id = client.create_experiment(name=experiment_name)
#             return client.get_experiment(experiment_id=experiment_id)
#     except Exception as e:
#         logger.error(f'Unable to get or create experiment {experiment_name}: {e}')