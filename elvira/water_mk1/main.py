"""
 * author Antonio Sirignano
 * created on 12-10-2026-10h-37m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from __future__ import annotations

import argparse
import json

from tqdm import tqdm
from datetime import datetime, timedelta, time
from pathlib import Path

from water.Advisor.advisor import IrrigationAdvisor
from water.Config.config import Config, BASE_DIR
from water.Data.data import DataLoader
from water.Features.features import FeaturesBuilder
from water.Models.models import BaseModel, GradientBoostingModel, PersistanceBaseline, RandomForestModel, RidgeModel, default_models
from water.Trainer.trainer import Trainer, Variant
from water.Validation.validation import WalkForwardValidator

say = tqdm.write

MODELS: dict[str, BaseModel] = {
    'ridge': RidgeModel(alpha=100.0),
    'ridge10': RidgeModel(alpha=10.0),
    'rf': RandomForestModel(),
    'gb': GradientBoostingModel()
}

VARIANTS: dict[str, Variant] = {
    'livello': Variant(use_wp_t=False, delta=False),
    'livello_wp': Variant(use_wp_t=True, delta=False),
    'delta': Variant(use_wp_t=False, delta=True),
    'delta_wp': Variant(use_wp_t=True, delta=True),
}

RESULT_DIR = BASE_DIR / 'results'

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