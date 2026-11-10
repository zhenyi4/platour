"""Prepare separately countable Hub repositories without changing original weights.

Only public/ is uploadable. Private source metadata and audit records are kept
outside it. Private source provenance stays outside every upload directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_NOTICE = ('This repository distributes historical inference weights without benchmark scores. '
    'No claim about checkpoint-selection protocol or unbiased held-out performance is made. '
    'Independent evaluation and provenance review are required before using these artifacts '
    'to support new comparative research claims.')
QWEN3_MATH_REPRODUCTION_NOTICE = (
    'This checkpoint comes from a subsequent reproduction run of **PLatour (GRPO), '
    'Qwen3-8B→Qwen3-1.7B, on MATH500**, using the reproduction code. '
    'The authors report **79.35% MATH500 accuracy** for this reproduced checkpoint. '
    'The original paper reports **76.50%** for this configuration in Table 1; '
    'the reproduction result is separate from that paper-reported result.')
ALLOWED_METADATA = {'format', 'family', 'task', 'method', 'experiment', 'layer_mapping',
    'layer_map_0_based', 'sender', 'receiver', 'injection', 'algorithm'}
PRIVATE_KEY = re.compile(r'(?:selected_.*step|global_step|training_steps?|total_steps|'
    r'gpu_(?:type|name)|hardware|training_(?:time|duration)|elapsed|wall.?clock|hostname)', re.I)
HARDWARE = re.compile(r'\b(?:[AHV](?:100|200|800)|B(?:100|200|300)|RTX\s*\d{4}|L40S?)\b', re.I)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def header(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 100_000_000:
            raise ValueError('Unexpected safetensors header size')
        return length, json.loads(stream.read(length))


def clone(source, destination):
    """APFS copy-on-write keeps tensor storage shared while headers remain separate."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(None, use_errno=True)
    method = getattr(library, 'clonefile', None)
    if method is not None:
        method.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        method.restype = ctypes.c_int
        if method(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 2_000_000_000:
        raise OSError('Insufficient disk space for a non-shared copy')
    shutil.copyfile(source, destination)


def file_and_payload_hash(path, offset):
    whole, payload = hashlib.sha256(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        whole.update(stream.read(offset))
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            whole.update(chunk)
            payload.update(chunk)
    return whole.hexdigest(), payload.hexdigest()


def public_weight(source, destination):
    length, original = header(source)
    public = {key: value for key, value in original.items() if key != '__metadata__'}
    metadata = {key: value for key, value in original.get('__metadata__', {}).items()
                if key in ALLOWED_METADATA}
    if metadata:
        public['__metadata__'] = metadata
    encoded = json.dumps(public, ensure_ascii=False, separators=(',', ':')).encode()
    if len(encoded) > length:
        raise ValueError('Public header must fit the existing tensor-data offset')
    clone(source, destination)
    with Path(destination).open('r+b') as stream:
        stream.seek(8)
        stream.write(encoded + b' ' * (length - len(encoded)))
    before, before_payload = file_and_payload_hash(source, 8 + length)
    after, after_payload = file_and_payload_hash(destination, 8 + length)
    if before_payload != after_payload or Path(source).stat().st_size != Path(destination).stat().st_size:
        raise RuntimeError('Tensor payload changed during public export')
    return {'original_sha256': before, 'public_sha256': after,
            'tensor_payload_sha256': after_payload, 'original_metadata': original.get('__metadata__', {}),
            'tensor_payload_unchanged': True}


def experiment_records():
    from qwen35.catalog import EXPERIMENTS
    catalog = json.loads((ROOT / 'qwen3/experiments.json').read_text())
    records = []
    task_names = {'arc_c': 'ARC-C', 'mmlu_redux': 'MMLU-Redux',
        'chemistry_no_cot': 'Chemistry-NoCoT', 'chemistry_cot': 'Chemistry-CoT',
        'gsm8k': 'GSM8K', 'math500': 'MATH500', 'dapo': 'DAPO', 'deepcoder': 'DeepCoder'}
    for case in catalog['experiments']:
        sender = case['sender']
        sender_size = '235B' if '235B' in sender else '8B'
        method = case['method']
        role = 'PLatour-SFT' if method == 'sft' else 'PLatour'
        records.append({'id': case['id'], 'family': 'qwen3', 'method': method,
            'task': case['task'], 'role': role, 'sender': 'Qwen/' + sender,
            'receiver': 'Qwen/Qwen3-1.7B', 'artifact_type': 'bridge',
            'slug': f'{role}-Qwen3-{sender_size}-to-1.7B-{task_names[case["task"]]}',
            'dataset_group': 'qwen3', 'layer_mapping': catalog[method]['layer_mapping']})
    for case in EXPERIMENTS:
        task = task_names[case.task]
        if case.method == 'bridge' and case.sender != 'Qwen3.5-4B':
            size = '397B-FP8' if '397B' in case.sender else '35B'
            slug = f'PLatour-Qwen3.5-{size}-to-4B-{task}'
        else:
            prefix = {'bridge': 'SC', 'random': 'VS', 'full': 'TR-FullGRPO', 'lora': 'TR-LoRA'}[case.method]
            slug = f'PLatour-{prefix}-Qwen3.5-4B-{task}'
        records.append({'id': case.identifier, 'family': 'qwen35', 'method': case.method,
            'task': case.task, 'role': case.paper_role,
            'sender': 'Qwen/' + case.sender if case.sender else None,
            'receiver': 'Qwen/Qwen3.5-4B', 'slug': slug, 'dataset_group': case.task,
            'artifact_type': {'bridge': 'bridge', 'random': 'virtual_sender',
                              'full': 'full_receiver', 'lora': 'custom_lora'}[case.method],
            'layer_mapping': list(case.layer_map) if case.method in ('bridge', 'random') else None})
    return records


def check_public_config(config):
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if PRIVATE_KEY.search(key):
                    raise ValueError('Non-public operational field in configuration: ' + key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and HARDWARE.search(value):
            raise ValueError('Hardware model in public configuration')
    visit(config)
    if config['evaluation']['unbiased_heldout_result_claimed'] is not False:
        raise ValueError('Inference packaging does not establish an unbiased evaluation')


def model_card(record, dataset_repo):
    bases = [base for base in (record['sender'], record['receiver']) if base]
    front = ('---\nlicense: apache-2.0\nbase_model:\n' + ''.join(f'  - {base}\n' for base in bases) +
             f'datasets:\n  - {dataset_repo}\ntags:\n  - platour\n  - safetensors\n  - custom-loader\n---\n')
    selection = INFERENCE_NOTICE
    if record['id'] == 18:
        selection = QWEN3_MATH_REPRODUCTION_NOTICE + '\n\n' + INFERENCE_NOTICE.removeprefix(
            'This repository distributes historical inference weights without benchmark scores. ')
    kind = {'bridge': 'Bridge projections; download the referenced sender and receiver separately.',
        'virtual_sender': 'Bridge projections, virtual hidden features and template token IDs are included. An external sender is not needed.',
        'full_receiver': 'Full receiver tensor state in the PLatour project format; use the referenced receiver configuration and tokenizer.',
        'custom_lora': 'Receiver LoRA tensors in the PLatour project format. This is not a PEFT-format adapter.'}[record['artifact_type']]
    card = front + f'''# {record['slug']}

Experiment {record['id']:02d}: {record['role']} on {record['task']}.

{kind}

## Evaluation provenance

{selection}

## Loading

Use the PLatour project loader with this repository's `config.json`. It supplies the experiment identity, dependencies, tensor mapping, weight filename and SHA-256. This repository is not advertised as directly loadable through `AutoModel.from_pretrained` or PEFT.

```sh
platour-exps download --experiment {record['id']}
```

For evaluation, use the matching experiment command in the project documentation, the separately downloaded base models and the signed task inputs. Programming evaluation additionally requires the isolated official judge.

## Reproducibility and licensing

Original tensor payloads are unchanged; file hashes reflect a public inference export. Exact backbone revisions used for the historical run have not been independently verified and must not be inferred from the latest Hub revision. This export does not attach benchmark results.

The export uses the project's Apache-2.0 license. The referenced upstream models retain their own licenses and notices. This inference export is not optimizer/RNG resume state.
'''
    if record['id'] == 18:
        card = card.replace('used for the historical run', 'used for the reproduction run').replace(
            'This export does not attach benchmark results.',
            'The reproduction score above is author-reported.')
    from qwen35.release_card import add_paper_sections, update_usage_sections
    card = add_paper_sections(card)
    # Legacy Full/LoRA staging has a different format from the published packages.
    # Update those published cards separately; never change weight/config export.
    return (update_usage_sections(card, record)
            if record['artifact_type'] in ('bridge', 'virtual_sender') else card)


def dataset_card(group, namespace, tracked):
    title = {'qwen3': 'PLatour-Qwen3-Data', 'dapo': 'PLatour-DAPO-Data', 'deepcoder': 'PLatour-DeepCoder-Data'}[group]
    configs = []
    if group == 'qwen3':
        for directory in sorted({Path(name).parts[2] for name in tracked if name.startswith('data/qwen3/')}):
            configs.append((directory, [('train', f'qwen3/{directory}/train.jsonl'),
                ('validation', f'qwen3/{directory}/validation.jsonl'), ('test', f'qwen3/{directory}/test.jsonl')]))
    else:
        configs.append((group, [('train', f'{group}/train.jsonl'), ('validation', f'{group}/validation.jsonl')]))
        suites = ['aime', 'hmmt', 'imo'] if group == 'dapo' else ['lcb', 'humaneval', 'mbpp', 'humaneval_template_adapted', 'mbpp_template_adapted']
        configs.extend((suite, [('test', f'{group}/{suite}.jsonl')]) for suite in suites)
    front = '---\npretty_name: ' + title + '\ntags:\n  - platour\nconfigs:\n'
    for name, splits in configs:
        front += f'  - config_name: {name}\n    data_files:\n'
        for split, path in splits:
            front += f'      - split: {split}\n        path: {path}\n'
    front += '---\n'
    body = f'# {title}\n\nFrozen PLatour task inputs. File bytes and question order are preserved for exact SHA-256 checks.\n\n'
    if group == 'qwen3':
        body += '## Evaluation use\n\n' + INFERENCE_NOTICE.replace('historical inference weights', 'frozen task inputs') + '\n\n'
    if group == 'deepcoder':
        body += ('TRAIN/VAL checker cases are rebuilt from the pinned public DeepCoder revision using '
                 '`python -m qwen35.data --output <fresh-case-directory>`. The original case archives are not included. '
                 'LCB v6 checker cases are obtained separately with `qwen35.judge_cases`.\n\n'
                 'Template-adapted HumanEval/MBPP prompts have separate subsets and must be labeled as such in results.\n\n')
    body += ('## Sources and use\n\nConsult the manifest for source identities and checksums and the project task documentation for prompt and scoring contracts. '
             'This collection combines upstream datasets with different licenses. Their original terms continue to apply; '
             'the project code license does not replace the source dataset licenses.\n')
    from qwen35.release_card import update_dataset_usage
    return update_dataset_usage(front + body, group)


def prepare(source_root, output, namespace='zen-E', *, update_manifest=False, workers=2):
    output, source_root = Path(output), Path(source_root)
    if output.exists():
        raise FileExistsError('Use a fresh release staging directory')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', namespace):
        raise ValueError('Invalid Hub namespace')
    output.mkdir(parents=True)
    public, private = output / 'public', output / 'private'
    private.mkdir(mode=0o700)
    from platour_data import download, release_inventory
    download(range(1, 31))
    tracked = release_inventory()
    dataset_names = {group: f'{namespace}/{name}' for group, name in {
        'qwen3': 'PLatour-Qwen3-Data', 'dapo': 'PLatour-DAPO-Data', 'deepcoder': 'PLatour-DeepCoder-Data'}.items()}
    source_manifest = json.loads((source_root / 'manifest.json').read_text())
    records = experiment_records()
    audits = []
    manifest = {'schema_version': 2, 'files': {}, 'repositories': {}, 'configs': {}}
    jobs = []
    for record in records:
        identifier = f'{record["id"]:02d}'
        filename, size, source_sha = source_manifest['files'][identifier]
        source = source_root / filename
        if source.name != filename or source.stat().st_size != size:
            raise ValueError('Original checkpoint size differs: ' + identifier)
        repository = namespace + '/' + record['slug']
        directory = public / 'models' / record['slug']
        jobs.append((record, source, directory / filename, source_sha, repository))

    def export(job):
        record, source, destination, source_sha, repository = job
        audit = public_weight(source, destination)
        if audit['original_sha256'] != source_sha:
            raise ValueError('Original checkpoint SHA differs from its supplied manifest')
        identifier = f'{record["id"]:02d}'
        _, h = header(destination)
        config = {'schema_version': 1, 'experiment_id': record['id'],
            'family': record['family'], 'task': record['task'], 'method': record['method'],
            'artifact_type': record['artifact_type'], 'role': record['role'],
            'base_models': {'sender': record['sender'], 'receiver': record['receiver']},
            'historical_base_revisions_verified': False, 'layer_mapping': record['layer_mapping'],
            'weight': {'filename': destination.name, 'size': destination.stat().st_size,
                       'sha256': audit['public_sha256']},
            'tensor_schema': {key: {'shape': value['shape'], 'dtype': value['dtype']}
                              for key, value in h.items() if key != '__metadata__'},
            'dataset': dataset_names[record['dataset_group']],
            'evaluation': {'scope': 'inference_artifact_only',
                'independent_evaluation_required': True, 'unbiased_heldout_result_claimed': False}}
        if record['family'] == 'qwen3':
            config['sender_layer_map_0_based'] = json.loads(h.get('__metadata__', {})['layer_map_0_based'])
        check_public_config(config)
        dump(destination.parent / 'config.json', config)
        card = model_card(record, config['dataset'])
        if HARDWARE.search(card) or PRIVATE_KEY.search(card):
            raise ValueError('Operational metadata in generated card')
        (destination.parent / 'README.md').write_text(card)
        shutil.copyfile(ROOT / 'LICENSE', destination.parent / 'LICENSE')
        public_record = {'repo_id': repository, 'repo_type': 'model',
            'folder': destination.parent.relative_to(output).as_posix(),
            'files': [destination.name, 'config.json', 'README.md', 'LICENSE']}
        audit.update(experiment_id=record['id'], source_filename=source.name)
        print(f'Prepared model {identifier}/30: {repository}', flush=True)
        return identifier, destination.name, config, public_record, audit

    plan = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for identifier, filename, config, item, audit in pool.map(export, jobs):
            manifest['files'][identifier] = [filename, config['weight']['size'], config['weight']['sha256']]
            manifest['repositories'][identifier] = item['repo_id']
            manifest['configs'][identifier] = digest(output / item['folder'] / 'config.json')
            audits.append(audit)
            plan.append(item)
    for group, repo_id in dataset_names.items():
        directory = public / 'datasets' / repo_id.split('/')[1]
        directory.mkdir(parents=True)
        names = []
        for name in tracked:
            if group == 'qwen3' and name.startswith('data/qwen3/'):
                relative = name.removeprefix('data/')
            elif group in ('dapo', 'deepcoder') and name.startswith('data/qwen35/' + group + '/'):
                relative = name.removeprefix('data/qwen35/')
            elif group in ('dapo', 'deepcoder') and name == 'data/qwen35/manifest.json':
                relative = 'manifest.json'
            else:
                continue
            clone(ROOT / name, directory / relative)
            names.append(relative)
        (directory / 'README.md').write_text(dataset_card(group, namespace, tracked))
        names.append('README.md')
        plan.append({'repo_id': repo_id, 'repo_type': 'dataset',
            'folder': directory.relative_to(output).as_posix(), 'files': sorted(names)})
    dump(public / 'models-manifest.json', manifest)
    dump(private / 'source-audit.json', audits)
    dump(output / 'release-plan.json', {'schema_version': 1, 'repositories': plan,
        'publication_status': 'local_review_only', 'collection_title': 'PLatour',
        'review_items': ['Complete independent evaluation and private provenance review before publishing research claims.',
            'Verify exact base-model revisions and downstream license notices.',
            'Review paper PDF and existing public code/history separately.',
            'Original training recipes retain algorithmic budgets for reproducibility.',
            'No Hub upload or repository creation was performed.']})
    if update_manifest:
        dump(ROOT / 'models/manifest.json', manifest)
    print(f'Prepared {len(plan)} repositories; only public/ is uploadable.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', default='zen-E')
    parser.add_argument('--update-manifest', action='store_true')
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.namespace)
    # A successful metadata audit is required before changing the code's manifest.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_hf_release import audit
    dump(args.output / 'private/public-audit.json', audit(args.output))
    if args.update_manifest:
        dump(ROOT / 'models/manifest.json', manifest)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
