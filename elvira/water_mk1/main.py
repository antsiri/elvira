"""
 * author Antonio Sirignano
 * created on 12-10-2026-10h-37m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from __future__ import annotations

import argparse
import json
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta, timezone
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

#   UTILITIES  -------------------------------------------------------------------------------------

def make_config(args: argparse.Namespace) -> Config:
    kw: dict = {}
    if args.data_dir:
        kw['data_dir'] = Path(args.data_dir)
    if args.horizon:
        kw['horizon_days'] = args.horizon
        kw['gap_days'] = max(args.horizon, 1)
    if args.min_train_days:
        kw['min_train_days'] = args.min_train_days
    if args.n_splits:
        kw['n_splits'] = args.n_splits
    if args.wp_low is not None or args.wp_high is not None:
        kw['wp_low'], kw['wp_high'] = args.wp_low, args.wp_high
    return Config(**kw)

def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, FeaturesBuilder]:
    loader = DataLoader(cfg=cfg)
    daily = loader.drop_partail_days(loader.load_daily())
    fb = FeaturesBuilder(cfg=cfg)
    everything = fb.build(daily=daily, required_target=False)
    labelled = (everything.dropna(subset=[fb.target_col] + fb.features_names_ + [fb.baseline_col]).reset_index(drop=True))
    return labelled, everything, fb

def zones_from(arg: str) -> tuple[int | None, ...]:
    return {'0': (0,), '1': (1,), 'both': (0, 1), 'pooled': (None,)}[arg]

#   COMMANDS  --------------------------------------------------------------------------------------

def run_train(cfg: Config, args: argparse.Namespace) -> None:
    df, _, fb = load_dataset(cfg=cfg)
    trainer = Trainer(cfg, default_models(cfg), WalkForwardValidator(cfg))
    summary = trainer.cross_validate(df, fb.features_names_, zones=zones_from(args.zones))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_DIR / 'summary.csv', index=False)
    trainer.predictions_.to_csv(RESULT_DIR / 'predictions.csv', index=False)

    cols = ['zone', 'variant', 'model', 'n', 'mae', 'rmse', 'fold_mae_std', 'skill_vs_persistence']
    for zone, g in summary.groupby('zone', sort=False):
        say(f'\n=== zone {zone} (ordered by MAE; skill < 0 = worst than persistance) ===')
        say(g[cols].head(args.top).round(3).to_string(index=False))

    if cfg.has_irrigation_range:
        rep = IrrigationAdvisor.threshold_report(trainer.predictions_, cfg.wp_low)
        rep = rep[rep.zone != 0]
        rep.to_csv(RESULT_DIR / 'threshold_report.csv', index=False)
        say(f'\n===  Decision: "It is neccessary a irrigation?" (wp < {cfg.wp_low}) ===')
        say(rep.round(2).sort_values(['zone', 'recall'], ascending=[True, False]).head(args.top * 2).to_string(index=False))

    say(f'\nResults saved in {RESULT_DIR}')

def run_prequential(cfg: Config, args: argparse.Namespace) -> None:
    df, _, fb = load_dataset(cfg)
    trainer = Trainer(cfg, [], WalkForwardValidator(cfg))
    for zone in zones_from(args.zones):
        res = trainer.prequential(df=df,
                                  features=fb.features_names_,
                                  model=MODELS[args.model],
                                  variant=VARIANTS[args.variant],
                                  zone=zone,
                                  refit_every=args.refit_every,
                                  )
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        res.to_csv(RESULT_DIR / f'prequential_zone{zone}.csv', index=False)

def run_fit(cfg: Config, args: argparse.Namespace) -> None:
    df, _, fb = load_dataset(cfg)
    trainer = Trainer(cfg, [], WalkForwardValidator(cfg))
    variant = VARIANTS[args.variant]
    for zone in tqdm(zones_from(args.zones), desc='Finale fit', unit='zone'):
        model, cols = trainer.fit_final(MODELS[args.model], df, fb.features_names_, variant, zone)
        path = trainer.save(model, cols, variant, zone)
        say(f'Saved in: {path}')

def run_solem_probe(cfg: Config, args: argparse.Namespace) -> None:
    from water.Ingestion.solem_client import SolemClient        #Local import: for requests/dotenv

    out = RESULT_DIR / 'solem_probe'
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, obj) -> None:
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        say(f'  write {out / name}')

    def utc_day(day: str) -> datetime:
        return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)

    client = SolemClient()
    with tqdm(total=4, desc='MySOLEM Probe', unit='step') as bar:
        client.login(); bar.update()
        module_ids = client.get_module_ids()
        dump('modules.json', module_ids); bar.update()
        say(f'Founded modules: {len(module_ids)}')
        module_id = args.module_id or (module_ids[0] if len(module_ids) == 1 else None)
        if module_id is None:
            overview = []
            for mid in tqdm(module_ids, desc='Modules Reading', leave=False):
                mod, inps = client.get_module_page(mid)
                overview.append({
                    'id': mid,
                    'name': mod.get('name'),
                    'type': mod.get('type'),
                    'n_inputs': len(inps),
                    'flow_meters': [i.get('name') or i['id'] for i in client.flow_inputs(inps)],
                })
            dump('modules_overview.json', overview)
            for o in overview:
                say(f'  {o['id']} name={o['name']!r} type={o['type']} flowmeters={o['flow_meters']}')
            raise SystemExit('More modules: execute again with --module-id <id> choosing with the flowmeter one.')

        module, inputs = client.get_module_page(module_id)
        dump('module_summary.json', {k: module.get(k) for k in ('id', 'name', 'type')})
        dump('inputs.json', inputs); bar.update()
        flow = client.flow_inputs(inputs)
        say(f'sensors: {len(inputs)}, flowmeter (type 1): {len(flow)}')

        ticks = {i['id']: client.get_last_tick(i['id']) for i in tqdm(flow, desc='Last tick', leave=False)}
        dump('last_ticks.json', ticks)

        end = utc_day(args.end) if args.end else datetime.now(timezone.utc)
        start = utc_day(args.start) if args.start else end - timedelta(days=args.days)
        if start >= end:
            raise SystemExit('Window not valid: start >= end')
        data = client.get_sensor_data(module_id, start, end)
        dump(f'sensor_data_{start:%Y%m%d}_end{end:%Y%m%d}.json', data); bar.update()

        say('It is necessary a parser. We need to run for the first time so see a json structure')

def run_recommend(cfg: Config, args: argparse.Namespace) -> None:
    if not cfg.has_irrigation_range:
        raise SystemExit('--wp-low and --wp-high are necessarry')
    
    zone = args.zone
    path = Path(args.model_path) if args.model_path else cfg.models_dir / f'zone{zone}.joblib'
    if not path.exists():
        raise SystemExit(f'Model not founded: {path}. Execute first: "python main.py fit --zones {zone}')

    labelled, eveything, _ = load_dataset(cfg)
    advisor = IrrigationAdvisor.from_bundle(cfg, path, labelled, margin=args.margin)

    rows = eveything[eveything[cfg.zone_col] == zone]
    if args.date:
        rows = rows[rows[cfg.date_col] == pd.Timestamp(args.date)]
        if rows.empty:
            raise SystemExit(f'No rows valid for zone {zone} on {args.date}')
    state = rows.iloc[-1]

    rec = advisor.recommend(state)
    say(f'\nzone {zone}, day {state[cfg.date_col].date()} (wp today: {state['wp_t']:.2f})')
    say(f'state: {rec.status}')
    say(f'Estimated wp without irrigation: {rec.wp_no_irrigation:.2f} (threshold {rec.threshold:.2f})')
    if rec.liters is not None:
        say(f'Reccomended Liters: {rec.liters:.0f}' + (f' -> expected wp {rec.wp_expected:.2f}' if rec.wp_expected is not None else ''))
    if rec.note:
        say(f'note: {rec.note}')

#   CLI  -------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data_dir', help='CSV Folder (default: ./data)')
    common.add_argument('--horizon', type=int, help='Previous days before prediction (default: 1)')
    common.add_argument("--min-train-days", type=int)
    common.add_argument('--n-splits', type=int)
    common.add_argument('--wp-low', type=float, help='threshold: below this wp it is irrigated')
    common.add_argument('--wp-high', type=float, help='Upper limit for wp')

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', parents=[common], help='Models and Variants walk-forward')
    p.add_argument('--zones', choices=['0', '1', 'both', 'pooled'], default='both')
    p.add_argument('--top', type=int, default=6, help='Rows showed by zone')
    p.set_defaults(func=run_train)

    p = sub.add_parser('prequential', parents=[common], help='Simultation day-by-day')
    p.add_argument('--zones', choices=['0', '1', 'both', 'pooled'], default='1')
    p.add_argument('--model', choices=MODELS, default='rf')
    p.add_argument('--variant', choices=VARIANTS, default='delta_wp')
    p.add_argument('--refit-every', type=int, default=3)
    p.set_defaults(func=run_prequential)

    p = sub.add_parser('fit', parents=[common], help='Final fit and saving')
    p.add_argument('--zones', choices=['0', '1', 'both', 'pooled'], default='1')
    p.add_argument('--model', choices=MODELS, default='ridge')
    p.add_argument('--variant', choices=VARIANTS, default='delta_wp')
    p.set_defaults(func=run_fit)

    p = sub.add_parser('recommend', parents=[common], help='Irrigation decision')
    p.add_argument('--zone', type=int, choices=[0, 1], default=1)
    p.add_argument('--date', help='AAAA-MM-DD (default: most recent day available)')
    p.add_argument('--margin', type=float, default=0.0, help='Security margin over wp_low')
    p.add_argument('--model-path', help=f'file .joblib (default: "models/zonaN.joblib)')
    p.set_defaults(func=run_recommend)

    p = sub.add_parser('solem-probe', parents=[common], help='Save the real answer form API mySOLEM')
    p.add_argument('--module-id', help='Module id')
    p.add_argument('--days', type=int, default=2, help='lecture days to download (default 2)')
    p.add_argument('--start', help='Initial data for inegstion from mySolem')
    p.add_argument('--end', help='End data for inegstion from mySolem')
    p.set_defaults(func=run_solem_probe)

    return parser

def main() -> None:
    args = build_parser().parse_args()
    cfg = make_config(args)
    args.func(cfg, args)

if __name__ == '__main__':
    main()