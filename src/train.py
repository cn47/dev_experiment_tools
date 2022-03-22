import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import pandas as pd

from logger import get_logger
from utils import rm_files, timer
from mlflow_writer import MlflowWriter

#from mlflow.tracking import MlflowClient
#from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME


pj_dir = Path('/opt')
config = OmegaConf.load(f"{pj_dir}/src/config/config.yaml")

mlflow_dir = pj_dir / config.mlflow.dir
hydra_dir = pj_dir / f"data/hydra/{datetime.now():%Y-%m-%d}/{datetime.now():%H-%M-%S}"
sys.argv.append(f"hydra.run.dir={hydra_dir}")
sys.argv.append(f"hydra.sweep.dir={hydra_dir}")

logger = get_logger("TrainOptimizer", f"{pj_dir}/log/train.log")


### Define Process #############################################################
def main():
    with timer('Load Data'):
        global X, y
        df_raw = pd.read_csv(pj_dir / "data/01_raw/train.csv")
        df_proc = preprocess(df_raw)
        X, y = df_proc.drop('Survived', axis=1), df_proc['Survived']
        positive = np.count_nonzero(y)
        negative = len(y) - np.count_nonzero(y)
        logger.info(f'positive: {positive} / negative: {negative)}')

    with timer('CrossValidHyperParamOptimizer'):
        optimizer()

### Define Function ############################################################
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df['FamilySize'] = _df['SibSp'] + _df['Parch'] + 1
    _df['Embarked'] = _df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    _df['Sex'] = _df['Sex'].map({'male': 0, 'female': 1})
    _df['IsAlone'] = 0
    _df.loc[_df['FamilySize'] == 1, 'IsAlone'] = 1
    _df.drop(['PassengerId', 'Name',  'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Ticket'], axis=1, inplace=True)

    return _df


@hydra.main(config_path=f"{pj_dir}/src/config", config_name="config")
def optimizer(config):
    cv = StratifiedKFold(
        n_splits=config.common.fold,
        shuffle=True,
        random_state=config.common.seed
    )

    model = LGBMClassifier(random_state=config.common.seed)
    hyperparams = {**config.model.fixed_params, **config.model.search_params}
    model.set_params(**hyperparams)

    fit_params = {
        "verbose": 0,
        "early_stopping_rounds": config.model.callbacks.early_stopping_rounds,
        "eval_metric": config.model.callbacks.metric,
        "eval_set": [(X, y)]
    }

    with timer("CrossValidScore", logger):
        scores = cross_validate(
            model, X, y,
            scoring=list(config.metrics), cv=cv,
            fit_params=fit_params, n_jobs=-1
        )

    logger.info("---- CrossValid Scores\n")
    logger.info(pformat(scores))

    writer = MlflowWriter(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=f"file://{mlflow_dir}/mlruns",
    )

    current_dir = Path().absolute()

    if current_dir.stem.startswith("trial_"):
        trial_num = int(current_dir.stem.split("_")[1])
        mlflow_run_name = f"SweepTrial{trial_num:03}"
    else:
        mlflow_run_name = f"ShotTrial"

    tags = {
        "RunAt": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
        "mlflow.runName": mlflow_run_name,
    }

    if config.mlflow.get("tags"):
        tags.update(config.mlflow.tags)

    writer.set_tags(tags)

    writer.log_params_from_omegaconf_dict(hyperparams)

    mean_scores = {f'mean_{k}'.replace('_test',''): v.mean() for k, v in scores.items()}
    std_scores = {f'std_{k}'.replace('_test',''): v.std() for k, v in scores.items()}

    [writer.log_metric(k, v) for k, v in mean_scores.items()]
    [writer.log_metric(k, v) for k, v in std_scores.items()]

    writer.log_artifact(current_dir / ".hydra" / "config.yaml")
    writer.log_artifact(current_dir / ".hydra" / "hydra.yaml")
    writer.log_artifact(current_dir / ".hydra" / "overrides.yaml")
    writer.log_artifact(current_dir / f"{Path(os.path.basename(__file__)).stem}.log")

    writer.set_terminated()

    return np.mean(scores["test_average_precision"])



### Execute Process ############################################################
if __name__ == '__main__':
    main()
