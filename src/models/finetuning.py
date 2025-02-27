import pandas as pd
import os
import pathlib
from typing import Callable
import joblib


import optuna
from optuna.trial import TrialState
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import score


class ClusteringObjective:

    def __init__(self, df:pd.DataFrame, model:str, artifact_store:FileSystemArtifactStore = None) -> None:
        self._artifact_store = artifact_store
        self.X = df
        self.model = model


    def _analyse_suggested_values(self, trial:optuna.Trial) -> float|str:
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value

        return "not_found"


    def __call__(self, trial:optuna.Trial) -> float:

        if self.model == "KMeans":
            n_clusters = trial.suggest_int("n_clusters", 2, 15)

            existing_run_score = self._analyse_suggested_values(trial)
            
            if existing_run_score == "not_found":
                model = KMeans(n_clusters=n_clusters, random_state=42)

                labels = model.fit_predict(self.X)

                mae = mean_absolute_error(self.X, labels)
                bias = self.X - labels
                custom_score = score.custom_score(self.X, labels)
                rmse = root_mean_squared_error(self.X, labels)
                mape = mean_absolute_percentage_error(self.X, labels)
                
                if self._artifact_store is not None :

                    path = f"{self._artifact_store._base_path}/trial_{trial.number}.pkl"
                    joblib.dump(model, filename=path)

                    artifact_id = upload_artifact(artifact_store=self._artifact_store,
                                                file_path=path,
                                                study_or_trial=trial,
                                                )
                    trial.set_user_attr("artifact_id", artifact_id)
                
                return mae, bias, custom_score, rmse, mape
            
            return existing_run_score


def run_optuna_study(df:pd.DataFrame,
                     models:list[str],
                     n_trials:int,
                     store_artifacts:bool,
                     study_name:str,
                     direction:str,
                     sampler:optuna.samplers,
                     pruner:optuna.pruners,
                     storage:optuna.storages
                    ) -> optuna.Study:

    study_args = {

    }

    pathlib.Path(f"results/optuna_studies/{study_name}.db").unlink(missing_ok=True)

    study = optuna.create_study(
                            study_name=study_name,
                            direction=direction,
                            sampler=sampler,
                            pruner=pruner, 
                            storage=storage
                            )

    for model in models:

        artifact_store = None
        if store_artifacts:
            base_path = f"artifacts/optuna_artifacts/{study_name}/{model}"
            if pathlib.Path(base_path).exists():
                pathlib.Path(base_path).rmdir()
            os.makedirs(base_path, exist_ok=True)

            artifact_store = FileSystemArtifactStore(base_path=base_path)

        objective = ClusteringObjective(df=df, model=model, artifact_store=artifact_store)

        study.optimize(objective, n_trials=n_trials)

    return study





if __name__ == "__main__":

    run_optuna_study(study_name="client_clustering",
                 df=X_scaled,
                 models=["HDBSCAN"],
                 scorer=DBCV,
                 store_artifacts=True)


    # joblib.dump(study, f"results/models/{model}_clustering.pkl")
    # logger.info(f"Best {feature}_{model} params:", study.best_params)

    best_artifact_id = study.best_trial.user_attrs.get("artifact_id")
    download_file_path = ...  # Set the path to save the downloaded artifact.
    download_artifact(
        artifact_store=artifact_store, file_path=download_file_path, artifact_id=best_artifact_id
    )
    with open(download_file_path, "rb") as f:
        content = f.read().decode("utf-8")
    print(content)

    best_models[f"{feature}_{model}"] = study.best_params
            
