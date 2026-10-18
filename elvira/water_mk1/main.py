"""
 * author Antonio Sirignano
 * created on 12-10-2026-10h-37m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import argparse

from tqdm import tqdm

from water.Advisor.advisor import IrrigationAdvisor
from water.Config.config import Config
from water.Data.data import DataLoader
from water.Features.features import FeaturesBuilder
from water.Models.models import GradientBoostingModel, PersistanceBaseline, RandomForestModel, RidgeModel
from water.Trainer.trainer import Trainer
from water.Validation.validation import WalkForwardValidator

def run_train(cfg: Config) -> None:
    raise NotImplementedError

def run_recommend(cfg: Config) -> None:
    raise NotImplementedError

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'recommend'])
    args = parser.parse_args()
    cfg = Config()

    {
        'train': run_train,
        'recommend': run_recommend
    }[args.command](cfg)


if __name__ == '__main__':
    main()