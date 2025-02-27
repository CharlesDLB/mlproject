import logging
import os
import mlflow.lightgbm
import mlflow.tracing
import yaml
import pathlib
import mlflow
import mlflow.sklearn
import optuna
import optunahub
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape

from src.models.models import ModelFactory

from steps.load import DataLoader
from steps.clean import Cleaner
from steps.split import TimeSerieSpliter
from steps.train import Trainer
from steps.predict import Predictor
from cdlb_logger.logger import setup_logger

from dotenv import load_dotenv

load_dotenv()

log_file_path = "logs/global_logger.log"
my_var = os.getenv('MLFLOW_TRACKING_URI')

logger = setup_logger(__name__, level=logging.INFO, stream=True, file=True, log_file_path=log_file_path)

sampler = optunahub.load_module("samplers/ctpe").cTPESampler()
# sampler = optuna.samplers.TPESampler(seed=42)
# pruner = optuna.pruners.MedianPruner()
# pruner = optuna.pruners.HyperbandPruner()  # Works better than median but Use only for the final test with recommended 1000 trials
# storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(f"results/optuna_studies/{study_name}.log"))
# storage = optuna.storages.RDBStorage(f"sqlite:///results/optuna_studies/{study_name}.db")



def mlflow_pipeline(mlflow_uri:str, config_file:str, experiment_name:str):

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    
    with mlflow.start_run(run_id=run.info.run_id) as run:

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Load data
        df = DataLoader()
        df = df.load_data()
        logger.info("Data loaded successfully")

        # Clean data
        cleaner = Cleaner()
        logger.info("Data cleaning completed successfully")

        # Split data
        spliter = TimeSerieSpliter(df=df, time_feature="Date")
        splits = spliter.time_serie_split(predict_week_range=13, splits=5)
        logger.info("Data split completed successfully")

        for train_idx, val_idx in splits:
            X = df.iloc[train_idx].copy()
            y = df.iloc[val_idx].copy()

            X_train, X_test, y_train, y_test = 
            model = ModelFactory.get_model(model_name, **model_kwargs)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            scorers = [mae, rmse, mape]

            scores = model.get_score(scorers=scorers, y_test=y_test, y_pred=predictions)
            print("\n".join(f"{model_name} - {key}: {value}" for key, value in scores.items()))


            # Save model
            trainer.save_model()
            logger.info("Model training completed successfully")
            
            # Evaluate model
            predictor = Predictor()
            X_test, y_test = predictor.feature_target_separator(test_data)
            mae, bias, mape, rmse = predictor.evaluate_model(X_test, y_test)
            logger.info("Model evaluation completed successfully")


        mlflow.set_tag('Model developer', 'prsdm')
        mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')
        
        # Log metrics
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("bias", bias)
        mlflow.log_metric('mae+bias', mae + bias)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('rmse', rmse)
        mlflow.sklearn.log_model(trainer.pipeline, "model")
                
        # Register the model
        model_name = "insurance_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        logger.info("MLflow tracking completed successfully")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"MAE: {mae:.4f}, BIAS: {bias:.4f}")
        print("=====================================================\n")







if __name__ == "__main__":

    mlflow_pipeline(
        mlflow_uri="sqlite:///mlflow.db",
        config_file="config.yml",
        experiment_name="LGBM_predictor"
    )
