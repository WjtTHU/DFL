"""Local DK daily protocol CLI: prepare, train, evaluate and predict."""
import hashlib
import json
import pickle
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from local_paths import PROJECT_ROOT, load_configs, project_path
from wip_pred.daily.data import (DailyDataset, build_index, fit_preprocessor,
    forecast_calendar, index_summary, load_dk, midnight, transform_snapshots, validate_protocol)
from wip_pred.daily.metrics import evaluate_predictions
from wip_pred.daily.model import build_daily_model, fit_target_spec, loss_per_sample, point_days
from wip_pred.training.helper import build_optimizer, build_scheduler
from wip_pred.utils.helper import seed_everything


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_hashes():
    paths = [PROJECT_ROOT/'run.py', PROJECT_ROOT/'local_paths.py',
             *sorted((PROJECT_ROOT/'src/wip_pred').rglob('*.py'))]
    return {str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'): file_hash(path) for path in paths}


def new_output(value, prefix):
    path = project_path(value) if value else PROJECT_ROOT/'outputs/dk/daily'/f'{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def sample_subset(index, smoke, seed):
    if not smoke:
        return index.reset_index(drop=True)
    groups = []
    for name, group in index.groupby('split', sort=False):
        limit = 128 if name == 'train' else 64
        groups.append(group.sample(n=min(limit, len(group)), random_state=seed).sort_index())
    return pd.concat(groups).reset_index(drop=True)


def prepare(config, out, smoke=False):
    validate_protocol(config)
    torch.set_num_threads(config['threads'])
    snapshots, labels, info = load_dk(config)
    print('Building fixed daily samples...', flush=True)
    full_index = build_index(snapshots, forecast_calendar(snapshots, config), config, labels)
    if full_index.empty:
        raise ValueError('No labelled daily samples.')
    index = sample_subset(full_index, smoke, config['seed'])
    for split in ('train', 'validation', 'test'):
        if not (index.split == split).any():
            raise ValueError(f'No {split} samples under the requested calendar/label availability rules.')
    core, training = load_configs('dk')
    preprocessor, core = fit_preprocessor(snapshots, core, config)
    transformed = transform_snapshots(snapshots, core, preprocessor)
    target_spec = fit_target_spec(index.loc[index.split == 'train', 'remaining_days'],
                                  config['distribution_horizon_quantile'],
                                  config.get('distribution_horizon_days'))
    index.to_csv(out/'sample_index.csv', index=False, encoding='utf-8-sig')
    manifest = {'protocol': config, 'data': info, 'full_sample_counts': index_summary(full_index),
                'used_sample_counts': index_summary(index), 'smoke_only': smoke,
                'sample_index_sha256': file_hash(out/'sample_index.csv'), 'target_spec': target_spec,
                'fitted_preprocessing_before': config['train_end'],
                'training_label_available_by': config['train_end'],
                'validation_label_available_by': config['validation_end'],
                'unlabelled_root_lots_excluded_from_supervision': info['root_lots']-len(labels),
                'source_hashes': code_hashes(),
                'notes': ['Offline whole-history enterprise cohort filtering is retained by agreement.',
                          'Labels retain enterprise rounded completion times; inputs retain raw observed timestamps.',
                          'Label availability conservatively requires completion + boundary margin before the split deadline.',
                          'Censored/unknown and not-yet-available labels are omitted, never set to zero.',
                          'Complete-case sampling can underrepresent long running Lots; this is not a survival model.',
                          'Test forecasts are held out from fitting, early stopping and model selection.']}
    write_json(out/'data_manifest.json', manifest)
    print(json.dumps(manifest['used_sample_counts'], ensure_ascii=False, indent=2), flush=True)
    return transformed, index, preprocessor, core, training, target_spec, manifest


def infer_loader(model, dataset, batch_size, mode, spec):
    model.eval()
    outputs, losses, ordered = [], [], []
    with torch.inference_mode():
        for ids, inputs, targets in dataset.loader(batch_size, shuffle=False):
            output = model(inputs)
            if not torch.isfinite(output).all():
                raise ValueError('Non-finite model outputs.')
            outputs.append(output.cpu().numpy())
            ordered.extend(ids.tolist())
            if targets is not None:
                losses.extend(loss_per_sample(output, targets, mode, spec).cpu().tolist())
    if ordered != list(range(len(dataset))):
        raise ValueError('Prediction output order differs from sample order.')
    return np.concatenate(outputs), np.asarray(losses) if losses else None


def train_mode(mode, transformed, index, preprocessor, core, training, spec, protocol, manifest, out):
    folder = out/mode
    folder.mkdir()
    seed_everything(protocol['seed'])
    model = build_daily_model(core, mode, spec)
    train_data = DailyDataset(transformed, index.loc[index.split == 'train'], core)
    val_data = DailyDataset(transformed, index.loc[index.split == 'validation'], core)
    loader = train_data.loader(protocol['batch_size'], shuffle=True, seed=protocol['seed'])
    optimizer = build_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)
    checkpoint_config = {'protocol': protocol, 'mode': mode, 'core': core, 'training': training,
                         'target_spec': spec, 'data_manifest': manifest,
                         'point_postprocessing': 'max(0, predicted days); dates use ceil, errors use unrounded days'}
    write_json(folder/'daily_config.json', checkpoint_config)
    with (folder/'preprocessor.pkl').open('wb') as stream:
        pickle.dump(preprocessor, stream)
    state = {'status': 'running', 'mode': mode, 'best_epoch': None, 'best_validation_score': None,
             'monitor': 'validation_mae_days' if mode == 'point' else 'validation_coarsened_nll',
             'test_evaluated_during_training': False, 'smoke_only': manifest['smoke_only'],
             'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
             'threads': protocol['threads'], 'seed': protocol['seed'],
             'config_sha256': file_hash(folder/'daily_config.json'),
             'preprocessor_sha256': file_hash(folder/'preprocessor.pkl')}
    write_json(folder/'training_state.json', state)
    best, stale, history = float('inf'), 0, []
    try:
        for epoch in range(1, protocol['epochs']+1):
            model.train()
            total, count = 0.0, 0
            for _, inputs, targets in loader:
                output = model(inputs)
                loss = loss_per_sample(output, targets, mode, spec).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss.')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())*len(targets)
                count += len(targets)
            scheduler.step()
            val_output, val_losses = infer_loader(model, val_data, protocol['batch_size'], mode, spec)
            if mode == 'point':
                _, predicted = point_days(torch.from_numpy(val_output), spec)
                score = float(np.abs(predicted.numpy()-val_data.index.remaining_days.to_numpy()).mean())
            else:
                score = float(val_losses.mean())
            if not np.isfinite(score):
                raise ValueError('Non-finite validation score.')
            record = {'epoch': epoch, 'training_loss': total/count, 'validation_loss': float(val_losses.mean()),
                      'validation_score': score, 'learning_rate': optimizer.param_groups[0]['lr']}
            history.append(record)
            with (folder/'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(f'{mode} epoch {epoch}: train={total/count:.6f}, {state["monitor"]}={score:.6f}', flush=True)
            if score < best:
                best, stale = score, 0
                state.update(best_epoch=epoch, best_validation_score=best)
                torch.save(model.state_dict(), folder/'best_model.pth')
            else:
                stale += 1
            if stale >= protocol['patience']:
                break
        model.load_state_dict(torch.load(folder/'best_model.pth', weights_only=True, map_location='cpu'))
        output, losses = infer_loader(model, val_data, protocol['batch_size'], mode, spec)
        predictions, metrics = evaluate_predictions(val_data.index, output, mode, spec, losses)
        key = 'mae_days' if mode == 'point' else 'coarsened_nll'
        if not np.isclose(metrics[key], best, atol=1e-5):
            raise ValueError('Reloaded selected checkpoint does not match its validation score.')
        predictions.to_csv(folder/'validation_predictions.csv', index=False, encoding='utf-8-sig')
        write_json(folder/'validation_metrics.json', metrics)
        state.update(status='completed', epochs_completed=len(history), early_stopped=stale >= protocol['patience'],
                     checkpoint_validation_verified=True, best_model_sha256=file_hash(folder/'best_model.pth'))
        write_json(folder/'training_state.json', state)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(folder/'training_state.json', state)
        raise


def load_checkpoint(folder):
    config = read_json(folder/'daily_config.json')
    state = read_json(folder/'training_state.json')
    if state['status'] != 'completed':
        raise ValueError('Daily checkpoint training has not completed successfully.')
    if file_hash(folder/'best_model.pth') != state['best_model_sha256']:
        raise ValueError('Checkpoint weights differ from the recorded selected model.')
    for filename, key in [('daily_config.json', 'config_sha256'), ('preprocessor.pkl', 'preprocessor_sha256')]:
        if file_hash(folder/filename) != state[key]:
            raise ValueError(f'Checkpoint {filename} differs from the saved training artifact.')
    validate_protocol(config['protocol'])
    torch.set_num_threads(config['protocol']['threads'])
    with (folder/'preprocessor.pkl').open('rb') as stream:
        preprocessor = pickle.load(stream)
    model = build_daily_model(config['core'], config['mode'], config['target_spec'])
    model.load_state_dict(torch.load(folder/'best_model.pth', weights_only=True, map_location='cpu'))
    model.eval()
    return model, preprocessor, config


def evaluate_checkpoint(folder, out, split='test'):
    model, preprocessor, config = load_checkpoint(folder)
    previous = config['data_manifest']
    current = code_hashes()
    if any(current.get(path) != digest for path, digest in previous['source_hashes'].items()):
        raise ValueError('Prediction source differs from this checkpoint. Evaluate in its recorded code version.')
    protocol, core = config['protocol'], config['core']
    snapshots, labels, info = load_dk(protocol)
    if info['csv_sha256'] != previous['data']['csv_sha256']:
        raise ValueError('Evaluation CSV differs from the recorded training source.')
    index = sample_subset(build_index(snapshots, forecast_calendar(snapshots, protocol), protocol, labels),
                          previous['smoke_only'], protocol['seed'])
    data = DailyDataset(transform_snapshots(snapshots, core, preprocessor), index.loc[index.split == split], core)
    if not len(data):
        raise ValueError(f'No {split} samples.')
    output, losses = infer_loader(model, data, protocol['batch_size'], config['mode'], config['target_spec'])
    predictions, metrics = evaluate_predictions(data.index, output, config['mode'], config['target_spec'], losses)
    predictions.to_csv(out/f'{split}_predictions.csv', index=False, encoding='utf-8-sig')
    write_json(out/f'{split}_metrics.json', {'split': split, 'mode': config['mode'],
               'checkpoint': str(folder), 'smoke_only': previous['smoke_only'],
               'used_for_model_selection': split == 'validation', 'metrics': metrics})
    return metrics


def predict_checkpoint(folder, out, date, lots, known_completed=()):
    model, preprocessor, config = load_checkpoint(folder)
    day = midnight(date)
    snapshots, _, _ = load_dk(config['protocol'], cutoff_exclusive=day+pd.Timedelta(days=1), with_labels=False)
    index = build_index(snapshots, [day], config['protocol'], requested_lots=lots,
                        known_completed_lots=known_completed)
    if index.empty:
        raise ValueError('No unfinished requested Lots with available history.')
    data = DailyDataset(transform_snapshots(snapshots, config['core'], preprocessor), index, config['core'])
    output, losses = infer_loader(model, data, config['protocol']['batch_size'], config['mode'], config['target_spec'])
    assert losses is None
    predictions, _ = evaluate_predictions(index, output, config['mode'], config['target_spec'])
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    write_json(out/'prediction_manifest.json', {'checkpoint': str(folder), 'forecast_date': str(day.date()),
               'mode': config['mode'], 'timezone': config['protocol']['timezone'],
               'requested_lots': lots, 'excluded_known_completed_lots': list(known_completed),
               'truth_labels_used': False,
               'active_population': 'Caller-supplied unfinished root Lots; no retrospective completion filtering.'})
    return predictions


def main(args):
    action = args.command.removeprefix('daily-')
    out = new_output(args.output, action)
    if action in ('prepare', 'train'):
        config = yaml.safe_load(project_path(args.protocol).read_text(encoding='utf-8'))
        if args.smoke:
            config['epochs'] = 1
            config['batch_size'] = 16
        if getattr(args, 'epochs', None) is not None:
            if args.smoke and args.epochs != 1:
                raise ValueError('--smoke only permits one epoch; omit it for a full experiment.')
            config['epochs'] = args.epochs
        transformed, index, preprocessor, core, training, spec, manifest = prepare(config, out, args.smoke)
        if action == 'train':
            for mode in ('point', 'distribution') if args.mode == 'both' else (args.mode,):
                train_mode(mode, transformed, index, preprocessor, core, training, spec, config, manifest, out)
    elif action == 'evaluate':
        evaluate_checkpoint(project_path(args.checkpoint), out, args.split)
    elif action == 'predict':
        predict_checkpoint(project_path(args.checkpoint), out, args.at, args.lot, args.completed_lot)
    print(f'Daily {action} completed: {out}', flush=True)
    return 0
