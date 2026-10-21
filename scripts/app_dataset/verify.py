"""Compare two independently generated datasets and verify protected source hashes."""
import argparse
import json
from pathlib import Path
from .generate import require, sha, write_json


def verify(primary, replica):
    primary, replica = Path(primary).resolve(), Path(replica).resolve()
    def inventory(root):
        return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*')
                if p.is_file() and p.name != 'reproducibility.json'}
    left, right = inventory(primary), inventory(replica)
    require(left == right, f'Non-reproducible outputs: {sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))}')
    manifest = json.loads((primary/'manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['file_sha256'].items():
        require(sha(primary/path) == expected, f'Output integrity failure: {path}')
    for path, expected in manifest['protected_source_sha256'].items():
        require(sha(path) == expected, f'Source/model changed: {path}')
    result = {'passed': True, 'independent_generations': 2, 'files_compared': len(left),
              'byte_identical_outputs': True, 'source_csv_predictor_and_models_unchanged': True,
              'protected_files_checked': len(manifest['protected_source_sha256']),
              'seed': manifest['config']['seed'], 'rng_streams':{
                  split:spec['rng_stream'] for split,spec in manifest['config']['splits'].items()}}
    write_json(primary/'reproducibility.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('primary', type=Path)
    parser.add_argument('replica', type=Path)
    args = parser.parse_args()
    verify(args.primary, args.replica)
