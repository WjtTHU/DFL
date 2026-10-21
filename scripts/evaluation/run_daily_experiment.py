"""Run the authorized full daily experiment sequentially, with durable phase records.

Training -> point test evaluation -> distribution test evaluation -> local report.
An existing run directory is never reused, so rerunning cannot silently restart training.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_paths import configure_imports, dataset_paths, project_path


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--protocol', default='config/daily_dk.yaml')
    parser.add_argument('--preflight', help='Verified preflight JSON, frozen into this run.')
    parser.add_argument('--order-policy', help='APP metadata policy only; not predictor features.')
    args = parser.parse_args()
    run = project_path(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    manifest_path = run/'run_manifest.json'
    manifest = {'run_id': run.name, 'run_dir': str(run), 'started_at': now(),
                'status': 'preparing', 'controller_pid': os.getpid(),
                'mode_order': ['point', 'distribution'], 'steps': {},
                'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'training_dir': str(run/'training'), 'report': str(run/'experiment_report.html')}

    def save():
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(manifest_path)

    def execute(name, arguments):
        manifest['status'] = name
        record = {'started_at': now(), 'arguments': arguments,
                  'log': str(run/f'{name}.log'), 'exit_code': None}
        manifest['steps'][name] = record
        save()
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1', PYTHONUTF8='1')
        with (run/f'{name}.log').open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            save()
            record['exit_code'] = process.wait()
        record['finished_at'] = now()
        save()
        if record['exit_code']:
            raise RuntimeError(f'{name} failed with exit code {record["exit_code"]}; see {record["log"]}')

    save()
    try:
        configure_imports()
        from wip_pred.daily.runner import code_hashes
        source = code_hashes()
        (run/'source_hashes.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
        with zipfile.ZipFile(run/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in source:
                archive.write(ROOT/relative, relative)
            archive.write(Path(__file__), str(Path(__file__).relative_to(ROOT)))
        shutil.copyfile(project_path(args.protocol), run/'source_daily_protocol.yaml')
        if args.preflight:
            preflight_path = project_path(args.preflight)
            preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
            from wip_pred.daily.runner import file_hash
            if any(file_hash(project_path(p)) != h for p, h in preflight['frozen_input_hashes'].items()):
                raise RuntimeError('Preflight inputs changed; do not launch training.')
            shutil.copyfile(preflight_path, run/'preflight.json')
        if args.order_policy:
            shutil.copyfile(project_path(args.order_policy), run/'source_app_order_policy.yaml')
        shutil.copyfile(ROOT/'scripts/evaluation/report_daily_experiment.py', run/'report_generator_at_start.py')
        for name in ('core_template.yaml', 'train_template.yaml'):
            shutil.copyfile(dataset_paths('dk')['config_dir']/name, run/f'source_{name}')
        (run/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'], cwd=ROOT))
        (run/'environment_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze'], cwd=ROOT))
        execute('training', ['run.py', 'daily-train', '--mode', 'both', '--protocol',
                            str(run/'source_daily_protocol.yaml'), '--output', str(run/'training')])
        for mode in manifest['mode_order']:
            state = json.loads((run/'training'/mode/'training_state.json').read_text(encoding='utf-8'))
            if state['status'] != 'completed' or state['smoke_only']:
                raise RuntimeError(f'{mode} is not a successfully completed full training run.')
        for mode in manifest['mode_order']:
            execute(f'evaluate_{mode}', ['run.py', 'daily-evaluate', '--checkpoint',
                    str(run/'training'/mode), '--split', 'test', '--output', str(run/'evaluation'/mode)])
        execute('report', ['scripts/evaluation/report_daily_experiment.py', '--run-dir', str(run)])
        manifest.update(status='report_completed', finished_at=now())
        save()
        print(json.dumps({'status': manifest['status'], 'report': manifest['report']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=now())
        save()
        (run/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
