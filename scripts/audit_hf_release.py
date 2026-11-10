"""Audit the explicit upload whitelist; never traverse private release records."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_hf_release import (ALLOWED_METADATA, HARDWARE, PRIVATE_KEY,
    check_public_config, digest, header)


def audit(output):
    output = Path(output).resolve()
    public = output / 'public'
    plan = json.loads((output / 'release-plan.json').read_text())
    manifest = json.loads((public / 'models-manifest.json').read_text())
    declared, repositories = set(), set()
    models = datasets = rows = 0
    for item in plan['repositories']:
        repo = item['repo_id']
        if repo in repositories:
            raise ValueError('Duplicate repository')
        repositories.add(repo)
        folder = output / item['folder']
        if not folder.resolve().is_relative_to(public):
            raise ValueError('Upload directory escapes public/')
        expected = set(item['files'])
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError('Upload whitelist differs from directory contents: ' + repo)
        for relative in expected:
            path = folder / relative
            if path.is_symlink() or not path.resolve().is_relative_to(public):
                raise ValueError('Unsafe upload path')
            declared.add(path)
            if path.suffix == '.json':
                value = json.loads(path.read_text())
                def check(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if PRIVATE_KEY.search(key):
                                raise ValueError('Operational metadata key in ' + str(path))
                            check(child)
                    elif isinstance(value, list):
                        for child in value:
                            check(child)
                    elif isinstance(value, str) and (HARDWARE.search(value) or '/Users/' in value):
                        raise ValueError('Operational metadata value in ' + str(path))
                check(value)
            if path.suffix == '.jsonl':
                def check_row(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if PRIVATE_KEY.search(key):
                                raise ValueError('Operational metadata key in frozen row: ' + str(path))
                            check_row(child)
                    elif isinstance(value, list) and value and isinstance(value[0], (dict, list)):
                        for child in value:
                            check_row(child)
                with path.open() as stream:
                    for line in stream:
                        check_row(json.loads(line))
                        rows += 1
            if path.name == 'README.md':
                text = path.read_text()
                if PRIVATE_KEY.search(text) or HARDWARE.search(text) or '/Users/' in text:
                    raise ValueError('Operational metadata in model/dataset card')
        if item['repo_type'] == 'model':
            models += 1
            config_path = folder / 'config.json'
            config = json.loads(config_path.read_text())
            check_public_config(config)
            identifier = f'{config["experiment_id"]:02d}'
            if manifest['configs'][identifier] != digest(config_path):
                raise ValueError('Config hash differs')
            weight = config['weight']
            if manifest['repositories'][identifier] != repo or manifest['files'][identifier] != [weight['filename'], weight['size'], weight['sha256']]:
                raise ValueError('Model manifest differs')
            path = folder / weight['filename']
            if path.stat().st_size != weight['size']:
                raise ValueError('Weight size differs')
            _, tensors = header(path)
            metadata = tensors.pop('__metadata__', {})
            if not set(metadata) <= ALLOWED_METADATA:
                raise ValueError('Unexpected weight metadata')
            for key, value in metadata.items():
                if PRIVATE_KEY.search(key) or HARDWARE.search(value) or '/Users/' in value:
                    raise ValueError('Operational weight metadata')
            schema = {key: {'dtype': value['dtype'], 'shape': value['shape']} for key, value in tensors.items()}
            if schema != config['tensor_schema']:
                raise ValueError('Tensor schema differs')
        elif item['repo_type'] == 'dataset':
            datasets += 1
        else:
            raise ValueError('Unknown repository type')
    actual = {p for p in public.rglob('*') if p.is_file()}
    if actual != declared | {public / 'models-manifest.json'}:
        raise ValueError('Unexpected file outside upload whitelist')
    if models != 30 or datasets != 3:
        raise ValueError('Incomplete release')
    return {'models': models, 'datasets': datasets, 'upload_files': len(declared), 'frozen_rows_checked_for_operational_keys': rows,
        'public_header_and_sidecar_checks': 'passed',
        'scope': 'Upload whitelist, model headers, cards and JSON sidecars; frozen problem text is retained verbatim.',
        'weight_payload_integrity': 'Recorded separately during export; this audit does not rehash tensor payloads.',
        'uploaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
