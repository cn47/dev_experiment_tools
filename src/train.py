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
from logger import get_logger
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from utils import rm_files, timer


pj_dir = Path('/opt')
config = OmegaConf.load(f"{pj_dir}/src/config/config.yaml")

mlflow_dir = pj_dir / config.mlflow.dir
hydra_dir = pj_dir / f"data/ydra/{datetime.now():%Y-%m-%d}/{datetime.now():%H-%M-%S}"
sys.argv.append(f"hydra.run.dir={hydra_dir}")
sys.argv.append(f"hydra.sweep.dir={hydra_dir}")

logger.get_logger("TrainOptimizer", f"{pj_dir}/log/train.log")



