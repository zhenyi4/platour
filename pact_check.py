"""PACT-Check: selectable DT, IT, SC, and TR sanity checks on one held-out split.

SC/TR must be separately trained; this module never relabels untrained weights as controls.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Callable

import torch
from safetensors.torch import load_file, save_file


def _fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _prompt(row):
    tokens = row['prompt_token_ids']
    if not isinstance(tokens, list) or len(tokens) < 3 or not all(
            type(token) is int and token >= 0 for token in tokens):
        raise ValueError('PACT-Check requires a fully rendered prompt-token list')
    return tokens


@dataclass(frozen=True)
class DummyStatistics:
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    prefix_means: tuple[torch.Tensor, ...]
    suffix_means: tuple[torch.Tensor, ...]
    training_digest: str
    sample_count: int
    seed: int

    def features(self, prompt_ids):
        prefix, suffix = len(self.prefix_ids), len(self.suffix_ids)
        original = prompt_ids[:-1]
        if (len(original) <= prefix + suffix or
                original[:prefix] != list(self.prefix_ids) or
                (suffix and original[-suffix:] != list(self.suffix_ids))):
            raise ValueError('Question/template positions differ from fixed DT statistics')
        layers = []
        for left, right in zip(self.prefix_means, self.suffix_means, strict=True):
            if left.ndim != 2 or right.ndim != 2 or left.shape[0] != prefix or right.shape[0] != suffix:
                raise RuntimeError('Dummy statistics layer/template shape differs')
            result = torch.zeros((1, len(original), left.shape[1]), dtype=left.dtype)
            result[0, :prefix] = left
            if suffix:
                result[0, -suffix:] = right
            layers.append(result)
        return tuple(layers)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Never overwrite TRAIN-only sender statistics')
        tensors = {f'{side}.{index}': tensor.contiguous()
                   for side, layers in (('prefix', self.prefix_means), ('suffix', self.suffix_means))
                   for index, tensor in enumerate(layers)}
        metadata = {'prefix_ids': json.dumps(self.prefix_ids),
                    'suffix_ids': json.dumps(self.suffix_ids),
                    'training_digest': self.training_digest,
                    'sample_count': str(self.sample_count), 'seed': str(self.seed)}
        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        with safe_open(path, framework='pt', device='cpu') as source:
            meta = source.metadata() or {}
            names = set(source.keys())
            count = len([name for name in names if name.startswith('prefix.')])
            if count < 1 or names != {f'{side}.{index}'
                                    for side in ('prefix', 'suffix') for index in range(count)}:
                raise ValueError('DT template sender layer statistics are incomplete')
            data = load_file(path, device='cpu')
        return cls(tuple(json.loads(meta['prefix_ids'])), tuple(json.loads(meta['suffix_ids'])),
                   tuple(data[f'prefix.{index}'] for index in range(count)),
                   tuple(data[f'suffix.{index}'] for index in range(count)),
                   meta['training_digest'], int(meta['sample_count']), int(meta['seed']))


def build_dummy_statistics(bridge, train_rows, *, prefix_ids, suffix_ids,
                           sample_count=500, seed=42, features=None,
                           allow_duplicate_prompts=False):
    """Offline, TRAIN-only block-input means; no sender is called by DT inference.

    ``features(prompt)`` returns the per-layer sender inputs (Qwen3 default: the bridge's
    own sender pass; Qwen3.5: ``Actor.sender_features``).
    """
    extract = features or bridge._features
    rows = list(train_rows)
    source_rows = rows
    if allow_duplicate_prompts:
        unique = {}
        for row in rows:
            unique.setdefault(tuple(_prompt(row)), row)
        rows = list(unique.values())
    prefix, suffix = tuple(prefix_ids), tuple(suffix_ids)
    if (not prefix or not suffix or sample_count < 1 or len(rows) < sample_count or
            len({tuple(_prompt(row)) for row in rows}) != len(rows) or
            any(type(token) is not int or token < 0 for token in (*prefix, *suffix))):
        raise ValueError('DT needs fixed prefix/suffix tokens and distinct TRAIN-only prompts')
    selected = random.Random(seed).sample(range(len(rows)), sample_count)
    from qwen3.parallel import dist
    sums = None
    with torch.inference_mode():
        for index in dist().local_slice(selected):
            prompt = _prompt(rows[index])
            content = prompt[:-1]
            if (len(content) <= len(prefix) + len(suffix) or
                    content[:len(prefix)] != list(prefix) or content[-len(suffix):] != list(suffix)):
                raise ValueError('TRAIN prompt has a different template or no question tokens')
            features = extract(prompt)
            if sums is None:
                sums = [[torch.zeros((len(prefix), value.shape[-1]), dtype=torch.float64),
                         torch.zeros((len(suffix), value.shape[-1]), dtype=torch.float64)]
                        for value in features]
            if len(features) != len(sums):
                raise RuntimeError('The sender layer mapping changed during DT calibration')
            for pair, value in zip(sums, features, strict=True):
                pair[0].add_(value[0, :len(prefix)].detach().cpu().double())
                pair[1].add_(value[0, -len(suffix):].detach().cpu().double())
    if dist().world() > 1:
        parts = [part for part in dist().gather(sums) if part is not None]
        sums = [[sum(part[layer][position] for part in parts) for position in (0, 1)]
                for layer in range(len(parts[0]))]
    return DummyStatistics(prefix, suffix,
        tuple((parts[0] / sample_count).float() for parts in sums),
        tuple((parts[1] / sample_count).float() for parts in sums),
        _fingerprint(source_rows), sample_count, seed)


def select_irrelevant_donors(train_rows, eval_rows, *, seed=42,
                             unrelated: Callable[[dict, dict], bool] | None = None):
    """Select distinct TRAIN-only full-length donors; never shorten, pad, or see a gold.

    A semantic duplicate/paraphrase exclusion callback is mandatory: token inequality
    alone cannot prove two differently phrased problems are unrelated.
    """
    if unrelated is None:
        raise ValueError('IT requires a duplicate/paraphrase exclusion policy')
    train, evaluation = list(train_rows), list(eval_rows)
    if not train or not evaluation:
        raise ValueError('IT needs TRAIN donors and separate evaluation questions')
    choices = {}
    rng = random.Random(seed)
    hidden = {'gold', 'answer', 'response', 'response_token_ids', 'judge', 'tests'}
    for row in evaluation:
        target = _prompt(row)
        index = row['index']
        if index in choices:
            raise ValueError('Evaluation question indices must be unique')
        public_target = {key: value for key, value in row.items() if key not in hidden}
        eligible = [candidate for candidate in train
                    if len(_prompt(candidate)) == len(target) and
                    _prompt(candidate) != target and unrelated(
                        {key: value for key, value in candidate.items() if key not in hidden},
                        public_target)]
        if not eligible:
            raise ValueError('No unrelated exact-length TRAIN donor; explicitly rephrase a donor without target content')
        choices[index] = list(_prompt(rng.choice(eligible)))
    return choices


class IrrelevantFeatures:
    def __init__(self, bridge, donor_prompts, *, features=None):
        self.bridge = bridge
        self.extract = features
        self.donors = {tuple(prompt): tuple(donor) for prompt, donor in donor_prompts.items()}

    def __call__(self, prompt):
        donor = self.donors.get(tuple(prompt))
        if donor is None or len(donor) != len(prompt) or donor == tuple(prompt):
            raise ValueError('IT donor must be unrelated and fully length-matched')
        if self.extract is not None:
            with torch.inference_mode():
                return tuple(self.extract(list(donor)))
        with torch.inference_mode():
            output = self.bridge.sender.model(
                input_ids=torch.tensor([donor[:-1]], device=self.bridge.sender.device),
                use_cache=False, output_hidden_states=True)
        return tuple(output.hidden_states[index] for index in self.bridge.indices)


def _release_accelerator_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_control_test(system):
    """Return training gradients/cache before a separate vLLM process starts."""
    model = getattr(system, 'bridge', None)
    if model is None:
        model = getattr(system, 'actor', None)
    if isinstance(model, torch.nn.Module):
        model.zero_grad(set_to_none=True)
        model.eval()
    _release_accelerator_memory()


def _accuracy(result, expected):
    correct = result.get('correct')
    if isinstance(correct, float) and correct.is_integer():
        result = dict(result, correct=int(correct))
    if (result.get('responses') != expected or type(result.get('correct')) is not int or
            not 0 <= result['correct'] <= expected or
            not result.get('complete', False)):
        raise RuntimeError('PACT-Check control has missing, duplicate or unscored responses')
    return result['correct'] / expected


CONTROLS = ('DT', 'IT', 'SC', 'TR')


def selected_controls(controls):
    selected = tuple(controls)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(CONTROLS):
        raise ValueError('Choose each of DT, IT, SC, TR at most once, with at least one control')
    return selected


class PACTCheck:
    """Orchestrate selected controls with one dataset and identical decoding/scoring.

    `evaluate(system, rows, *, feature_provider)` must return complete response counts.
    The two factories must *train separate* SC and TR systems using TRAIN/VAL only,
    returning `(system, receipt)` with a non-test training split digest and method.
    """

    def __init__(self, bridge, train_rows, validation_rows, test_rows, *, evaluator,
                 sender_features=None, allow_duplicate_prompts=False):
        self.bridge = bridge
        self.sender_features = sender_features
        self.train = list(train_rows)
        self.validation = list(validation_rows)
        self.test = list(test_rows)
        self.evaluate = evaluator
        if not self.train or not self.validation or not self.test:
            raise ValueError('PACT-Check requires separate nonempty TRAIN/VAL/TEST splits')
        splits = (self.train, self.validation, self.test)
        groups = [{tuple(_prompt(row)) for row in rows} for rows in splits]
        if any((not allow_duplicate_prompts and len(prompt_set) != len(rows)) or
               len({row['index'] for row in rows}) != len(rows)
               for prompt_set, rows in zip(groups, splits, strict=True)):
            raise ValueError('PACT-Check split has duplicate question prompts or indices')
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('PACT-Check cannot train or calibrate on evaluation prompts')
        self.train_digest = _fingerprint(self.train)
        self.validation_digest = _fingerprint(self.validation)

    def run(self, *, dummy=None, donors=None, train_sc=None, train_tr=None,
            samples_per_question, controls=CONTROLS, skip_test=False):
        controls = selected_controls(controls)
        if skip_test and not set(controls) <= {'SC', 'TR'}:
            raise ValueError('Training-only checks allow SC/TR only')
        if samples_per_question < 1:
            raise ValueError('Positive sample count required')
        if 'DT' in controls and (dummy is None or dummy.training_digest != self.train_digest):
            raise ValueError('DT requires original TRAIN-only calibration')
        if 'IT' in controls:
            if not donors or {row['index'] for row in self.test} != set(donors):
                raise ValueError('IT donor assignment must cover exactly the evaluation questions')
            prompts = {tuple(_prompt(row)): donors[row['index']] for row in self.test}
            it = IrrelevantFeatures(self.bridge, prompts, features=self.sender_features)
        expected = len(self.test) * samples_per_question
        results, checkpoints, receipts = {}, set(), {}
        for label, factory in (('SC', train_sc), ('TR', train_tr)):
            if label not in controls:
                continue
            if factory is None:
                raise ValueError(label + ' requires a training factory')
            system, receipt = factory(self.train, self.validation)
            if (system is self.bridge or not isinstance(receipt, dict) or
                    not receipt.get('trained') or receipt.get('control') != label or
                    receipt.get('train_digest') != self.train_digest or
                    receipt.get('validation_digest') != self.validation_digest or
                    not receipt.get('checkpoint') or receipt.get('uses_test_for_training')):
                raise RuntimeError(label + ' needs a separately trained TRAIN-only control')
            if receipt['checkpoint'] in checkpoints:
                raise RuntimeError('SC and TR must use separately trained control models')
            checkpoints.add(receipt['checkpoint'])
            receipts[label] = receipt
            # The training allocator's cached blocks must be returned before vLLM
            # checks free memory in its separate process, not only after TEST.
            try:
                if skip_test:
                    if not receipt.get('training_only') or receipt.get('selection_performed') is not False:
                        raise RuntimeError('Skipping TEST requires an explicit training-only smoke receipt')
                else:
                    _prepare_control_test(system)
                    results[label] = self.evaluate(system, self.test, feature_provider=None)
            finally:
                del system
                _release_accelerator_memory()
        if skip_test:
            return dict(complete=True, controls=list(controls), smoke_only=True,
                        test_performed=False, responses_per_setting=0, raw={}, accuracies={},
                        training_receipts=receipts, train_digest=self.train_digest)
        if 'DT' in controls:
            results['DT'] = self.evaluate(self.bridge, self.test, feature_provider=dummy.features)
        if 'IT' in controls:
            results['IT'] = self.evaluate(self.bridge, self.test, feature_provider=it)
        backends = {value.get('backend') for value in results.values()}
        if len(backends) != 1 or None in backends:
            raise RuntimeError('Every PACT-Check setting must report the same HF/vLLM backend')
        measured = {label: _accuracy(value, expected) for label, value in results.items()}
        report = dict(complete=True, controls=list(controls), accuracies=measured,
                      responses_per_setting=expected, raw=results, train_digest=self.train_digest,
                      test_performed=True, training_receipts=receipts)
        if 'DT' in controls:
            report.update(dummy_training_samples=dummy.sample_count,
                          dt_paper_calibration_count_met=dummy.sample_count == 500)
        return report


def _sealed_settings(root, settings):
    from qwen3.parallel import root_call, sign_topology
    sign_topology(settings)
    return root_call(lambda: _sealed_settings_local(root, settings))


def _sealed_settings_local(root, settings):
    from qwen3.checkpoints import atomic_json
    protocol = root / 'pact_protocol.json'
    if root.exists():
        if (not protocol.is_file() or json.loads(protocol.read_text()) != settings or
                (root / 'pact_check.json').exists()):
            raise RuntimeError('PACT-Check restart must use the same incomplete private run')
    else:
        root.mkdir(parents=True)
        atomic_json(settings, protocol)


def _signed_dummy(root, train, args, build):
    statistics_path = root / 'dummy_train_only.safetensors'
    if statistics_path.is_file():
        dummy = DummyStatistics.load(statistics_path)
        if (dummy.training_digest != _fingerprint(train) or dummy.sample_count != args.dt_samples or
                dummy.prefix_ids != tuple(args.prefix_ids) or
                dummy.suffix_ids != tuple(args.suffix_ids)):
            raise RuntimeError('Previously signed DT statistics use different TRAIN/template inputs')
        return dummy
    dummy = build()
    from qwen3.parallel import root_call
    root_call(lambda: dummy.save(statistics_path))
    return dummy


def _run_qwen35(args, train, validation, test, donors):
    """Qwen3.5 PACT-Check: PACT bridge, DT/IT interventions, 4B self bridge SC, 4B TR."""
    from platour_api import Qwen35Session
    from qwen3.parallel import api_json as atomic_json
    controls = selected_controls(getattr(args, 'controls', CONTROLS))
    root = args.output.absolute()
    validation_samples = getattr(args, 'validation_samples', None) or (4 if args.task == 'dapo' else 16)
    test_cap = getattr(args, 'test_response_cap', None) or (32768 if args.task == 'dapo' else 16384)
    judge = dict(code_cases=args.code_cases, official_checker=args.official_checker,
                 docker_image=args.docker_image, allow_rebuilt_image=args.allow_rebuilt_image,
                 device=args.device, receiver_shape=args.receiver_shape)
    settings = {'family': 'qwen35', 'receiver_shape': list(args.receiver_shape),
                'sender': str(args.sender.resolve()),
                'receiver': str(args.receiver.resolve()),
                'bridge_sha256': hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
                'train_sha256': _fingerprint(train), 'validation_sha256': _fingerprint(validation),
                'test_sha256': _fingerprint(test), 'semantic_group_field': args.semantic_group_field,
                'prefix_ids': args.prefix_ids, 'suffix_ids': args.suffix_ids,
                'dt_samples': args.dt_samples, 'control_epochs': 1, 'control_steps': (len(train) + args.batch_size - 1) // args.batch_size,
                'batch_size': args.batch_size, 'samples': args.samples,
                'response_cap': args.response_cap, 'validation_samples': validation_samples,
                'test_response_cap': test_cap, 'tr_method': args.tr_method,
                'task': args.task, 'backend': args.backend,
                'control_objective': args.control_objective}
    settings['controls'] = list(controls)
    _sealed_settings(root, settings)
    base = (Qwen35Session.load(sender=args.sender, bridge=args.bridge, receiver=args.receiver,
                              task=args.task, method='bridge', **judge)
            if set(controls) & {'DT', 'IT'} else None)
    dummy = _signed_dummy(root, train, args, lambda: build_dummy_statistics(
        base.actor, train, prefix_ids=args.prefix_ids, suffix_ids=args.suffix_ids,
        sample_count=args.dt_samples, features=base.actor.sender_features)) if 'DT' in controls else None
    def evaluate_control(system, rows, *, feature_provider):
        label = (getattr(system, 'pact_control', None)
                 or ('DT' if dummy is not None and getattr(feature_provider, '__self__', None) is dummy
                     else 'IT' if feature_provider is not None else None))
        if label not in controls:
            raise RuntimeError('Attempted to evaluate an unselected control')
        return system.evaluate(rows, backend=args.backend, samples=args.samples,
                               cap=test_cap, output=root / 'test' / label,
                               vllm_python=args.vllm_python, feature_provider=feature_provider)

    pact = PACTCheck(base, train, validation, test, evaluator=evaluate_control,
                     sender_features=base.actor.sender_features if base is not None else None)

    def control(label, load):
        def train_control(training, heldout):
            system = load()
            folder = root / label
            receipt = system.fit(training, heldout, output=folder,
                objective=args.control_objective, epochs=1,
                batch_size=args.batch_size, validation_backend='hf' if args.control_objective == 'sft' else args.backend,
                validation_samples=validation_samples, response_cap=args.response_cap,
                vllm_python=args.vllm_python, control=label)
            system.pact_control = label
            return system, receipt
        return train_control

    report = pact.run(dummy=dummy, donors=donors,
        train_sc=control('SC', lambda: Qwen35Session.load(
            sender=args.receiver, receiver=args.receiver, task=args.task, method='bridge', **judge)),
        train_tr=control('TR', lambda: Qwen35Session.load(
            receiver=args.receiver, task=args.task, method=args.tr_method, **judge)),
        samples_per_question=args.samples, controls=controls)
    report.update(family='qwen35', control_objectives={'SC': args.control_objective,
                                                       'TR': args.control_objective},
                  tr_method=args.tr_method, evaluation_backend=args.backend,
                  sc_epochs=1, tr_epochs=1, selection_policy=('final_sft_checkpoint_v1' if args.control_objective == 'sft' else 'validation_best_earliest_tie_v1'),
                  protocol_scope='user-selected TRAIN/VAL/TEST and control budgets')
    report['control_objectives'] = {label: value for label, value in report['control_objectives'].items() if label in controls}
    for label in ('SC', 'TR'):
        if label not in controls:
            report.pop(label.lower() + '_epochs', None)
    atomic_json(report, root / 'pact_check.json')
    return report


def main(argv=None):
    """Qwen3 defaults to the frozen full suite; Qwen3.5 keeps its legacy interface."""
    import argparse
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    from qwen3.parallel import maybe_launch, dist
    if maybe_launch(['pact-check', *argv]):
        return
    if not any(flag in argv for flag in ('--help', '-h', '--prepare-only')):
        dist().init()
    route = argparse.ArgumentParser(add_help=False)
    route.add_argument('--family', choices=('qwen3', 'qwen35'), default='qwen3')
    route.add_argument('--it-source', choices=('archived-it-v1', 'custom'), default='archived-it-v1')
    args, remaining = route.parse_known_args(argv)
    if args.family == 'qwen3' and args.it_source == 'archived-it-v1':
        from qwen3.pact_cli import main as run_frozen
        return run_frozen(remaining)
    return legacy_main(['--family', args.family, *remaining])


def legacy_main(argv=None):
    """Run all four actual controls; resume SC/TR only from their signed private roots."""
    import argparse
    from platour_api import load_dataset

    parser = argparse.ArgumentParser(description='PACT-Check four-control, held-out evaluation')
    for option in ('sender', 'bridge', 'receiver', 'dataset', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--family', choices=('qwen3', 'qwen35'), default='qwen3',
                        help='qwen3: Qwen3 bridges; qwen35: Qwen3.5 sender to 4B receiver')
    parser.add_argument('--code-cases', type=Path, help='Qwen3.5 DeepCoder: verified cases/ root')
    parser.add_argument('--official-checker', type=Path)
    parser.add_argument('--docker-image', default='lcb-checker:local')
    parser.add_argument('--allow-rebuilt-image', action='store_true')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Qwen3.5 receiver device')
    parser.add_argument('--receiver-shape', default='32x2560',
                        type=lambda text: tuple(int(part) for part in text.lower().split('x')),
                        help='Qwen3.5 receiver LAYERSxWIDTH; default requires Qwen3.5-4B')
    parser.add_argument('--task', required=True)
    parser.add_argument('--mapping', choices=('last_aligned', 'uniform_endpoints'),
                        default='last_aligned')
    parser.add_argument('--prefix-ids', type=json.loads, required=True,
                        help='JSON array of exact fixed template prefix token IDs')
    parser.add_argument('--suffix-ids', type=json.loads, required=True,
                        help='JSON array of exact fixed template suffix token IDs ending just '
                             'before the final prompt token')
    parser.add_argument('--semantic-group-field',
                        help='TRAIN and TEST metadata field audited to exclude duplicates/paraphrases')
    parser.add_argument('--it-experiment', type=int,
                        help='Use pinned Qwen3 IT donors; TEST prompts and protocol must match exactly')
    parser.add_argument('--it-data-root', type=Path,
                        help='Optional root of downloaded archived-it-v1 inputs')
    parser.add_argument('--it-local-files-only', action='store_true')
    parser.add_argument('--controls', nargs='+', choices=CONTROLS, default=list(CONTROLS))
    from qwen3.parallel import add_arguments
    add_arguments(parser)
    parser.add_argument('--dt-samples', type=int, default=500)
    parser.add_argument('--samples', type=int, help='TEST responses: Qwen3 task recipe; Qwen3.5 math 16 (IMO: set 4), code 8')
    parser.add_argument('--validation-samples', type=int, help='Qwen3.5 VAL responses: DAPO 4, DeepCoder 16')
    parser.add_argument('--test-response-cap', type=int, help='Qwen3.5 TEST token cap: math 32768, code 16384')
    parser.add_argument('--response-cap', type=int, help='TRAIN/VAL cap: Qwen3 task recipe; Qwen3.5 16384')
    parser.add_argument('--control-steps', type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int,
                        help='Effective batch: Qwen3 published recipe; Qwen3.5 64')
    parser.add_argument('--micro-batch', type=int, help='Qwen3 SFT micro-batch; default follows the recipe')
    parser.add_argument('--checkpoint-every', type=int, help='Qwen3 extra recovery interval; default 0 (VAL/final only)')
    parser.add_argument('--tr-method', choices=('full', 'lora'), default='full',
                        help='TR training method (default: full)')
    parser.add_argument('--tr-lora-rank', type=int, default=128)
    parser.add_argument('--control-objective', choices=('sft', 'grpo'), default='grpo')
    parser.add_argument('--backend', choices=('vllm',), default='vllm')
    parser.add_argument('--vllm-python', type=Path,
                             help='Optional vLLM interpreter override; defaults to current project Python')
    args = parser.parse_args(argv)
    try:
        args.controls = selected_controls(args.controls)
    except ValueError as error:
        parser.error(str(error))
    if args.family == 'qwen35':
        if args.samples is None:
            args.samples = 16 if args.task == 'dapo' else 8
        if args.response_cap is None:
            args.response_cap = 16384
        if args.validation_samples is None:
            args.validation_samples = 4 if args.task == 'dapo' else 16
        if args.test_response_cap is None:
            args.test_response_cap = 32768 if args.task == 'dapo' else 16384
        if args.validation_samples < 1 or args.test_response_cap < 1:
            parser.error('Validation samples and TEST response cap must be positive')
    else:
        if args.validation_samples is not None or args.test_response_cap is not None:
            parser.error('--validation-samples/--test-response-cap apply to Qwen3.5 only')
        from qwen3.cli import select
        config = next(select(i) for i in range(1, 19)
                      if select(i)['task'] == args.task)
        if args.samples is None:
            args.samples = config['eval_samples']
        if args.response_cap is None:
            args.response_cap = config['response_cap']
    if args.batch_size is None:
        from training_protocol import qwen3_batch_size
        args.batch_size = (qwen3_batch_size(args.control_objective, args.task)
                           if args.family == 'qwen3' else 64)
    if args.family == 'qwen35' and (args.micro_batch is not None or args.checkpoint_every is not None):
        parser.error('--micro-batch/--checkpoint-every apply to Qwen3 controls only')
    if args.micro_batch is None:
        from training_protocol import qwen3_micro_batch
        args.micro_batch = qwen3_micro_batch(args.control_objective, args.task) if args.family == 'qwen3' else 1
    if args.checkpoint_every is None:
        args.checkpoint_every = 0
    if args.micro_batch < 1 or args.checkpoint_every < 0:
        parser.error('Micro-batch must be positive and checkpoint interval nonnegative')
    if args.control_steps is not None:
        parser.error('SC/TR use family epoch budgets (Qwen3: 3; Qwen3.5: 1); remove --control-steps')
    if args.control_steps is None:
        args.control_steps = None  # Formal budgets are derived from TRAIN and family epochs.
    if args.backend == 'vllm':
        from platour_exps import require_vllm_python
        require_vllm_python(args.vllm_python)
    if (args.dt_samples < 1 or args.samples < 1 or args.response_cap < 1 or
            args.batch_size < 1 or args.tr_lora_rank < 1):
        parser.error('Sampling, training and LoRA budgets must be positive')
    if any(not isinstance(ids, list) or not ids or
           any(type(token) is not int or token < 0 for token in ids)
           for ids in (args.prefix_ids, args.suffix_ids)):
        parser.error('Fixed template prefix/suffix must be nonempty JSON token-ID arrays')
    source = args.dataset
    train = load_dataset(source / 'train.jsonl')
    validation = load_dataset(source / 'validation.jsonl')
    test = load_dataset(source / 'test.jsonl')
    field = args.semantic_group_field
    if 'IT' in args.controls and args.it_experiment is None and (not field or any(field not in row or not isinstance(row[field], str) or not row[field]
           for row in train + test)):
        raise ValueError('Every TRAIN and TEST prompt needs an audited semantic group')
    if set(args.controls) & {'SC', 'TR'} and args.control_objective == 'sft' and any(
            not row.get('response_token_ids') for row in train + validation):
        raise ValueError('Separate SC/TR SFT controls need TRAIN and VAL response_token_ids')
    if any(not isinstance(row.get('gold'), str) for row in train + validation + test):
        raise ValueError('PACT train, held-out validation and TEST need separate gold labels')
    fixed_config = fixed_it_audit = None
    if 'IT' not in args.controls:
        donors = None
    elif args.it_experiment is not None:
        if args.family != 'qwen3':
            parser.error('--it-experiment applies only to the published Qwen3 IT inputs')
        from platour_it import DEFAULT_ROOT, load_inputs
        _, donors, fixed_config, fixed_it_audit = load_inputs(args.it_experiment,
            args.it_data_root or DEFAULT_ROOT, test_rows=test,
            local_files_only=args.it_local_files_only)
        if fixed_it_audit['duplicate_test_prompt_rows']:
            parser.error('Custom mode requires unique TEST prompts; use default --experiment <ID> for the full archived suite')
        if (args.task != fixed_config['task'] or args.mapping != fixed_config['mapping']
                or args.samples != fixed_config['samples'] or args.response_cap != fixed_config['cap']):
            parser.error('Task, mapping, samples and response-cap must match the frozen IT protocol')
        from platour_exps import verify_weight, weight_record
        _, _, size, digest = weight_record(args.it_experiment)
        verify_weight(args.bridge, size, digest)
    else:
        group = lambda donor, target: donor[field] != target[field]
        donors = select_irrelevant_donors(train, test, unrelated=group)
    if args.family == 'qwen35':
        if args.task not in ('dapo', 'deepcoder'):
            parser.error('Qwen3.5 PACT-Check task must be dapo or deepcoder')
        report = _run_qwen35(args, train, validation, test, donors)
        print(json.dumps({key: report[key] for key in
                          ('controls', 'accuracies', 'responses_per_setting')},
                         sort_keys=True))
        return
    return _run_qwen3(args, train, validation, test, donors,
                      fixed_config=fixed_config, fixed_it_audit=fixed_it_audit)


def _run_qwen3(args, train, validation, test, donors, *, fixed_config=None,
               fixed_it_audit=None, allow_duplicate_prompts=False):
    from platour_api import PLatourSession, ReceiverSession
    from qwen3.parallel import api_json as atomic_json, root_call
    sft = args.control_objective == 'sft'
    smoke_steps = getattr(args, 'grpo_smoke_steps', None)
    controls = selected_controls(getattr(args, 'controls', CONTROLS))
    if smoke_steps is not None and (sft or smoke_steps < 1 or not set(controls) <= {'SC', 'TR'}):
        raise ValueError('GRPO smoke needs positive steps and only SC/TR')
    training_budget = dict(epochs=3)
    from training_protocol import qwen3_micro_batch
    training_budget.update(micro_batch=getattr(args, 'micro_batch', qwen3_micro_batch(args.control_objective, args.task)),
                           checkpoint_interval=getattr(args, 'checkpoint_every', 0))
    if smoke_steps is not None:
        training_budget.update(max_steps=smoke_steps, smoke=True)
    training_validation = dict(task=None if sft else args.task,
        validation_backend='hf' if sft else args.backend,
        validation_schedule='initial' if smoke_steps else 'epoch' if sft else 'half_epoch')
    tr_validation_samples = 4 if args.task == 'gsm8k' and args.control_objective == 'grpo' else args.samples
    control_steps = smoke_steps or 3 * ((len(train) + args.batch_size - 1) // args.batch_size)
    if sft:
        from training_protocol import qwen3_sft_warmup
        training_budget['warmup_steps'] = qwen3_sft_warmup(args.task, control_steps)
    field = args.semantic_group_field
    root = args.output.absolute()
    settings = {'sender': str(args.sender.resolve()), 'receiver': str(args.receiver.resolve()),
                'bridge_sha256': hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
                'train_sha256': _fingerprint(train), 'validation_sha256': _fingerprint(validation),
                'test_sha256': _fingerprint(test), 'semantic_group_field': field,
                'prefix_ids': args.prefix_ids, 'suffix_ids': args.suffix_ids,
                'dt_samples': args.dt_samples, 'control_epochs': 3,
                'control_training_steps': {'SC': control_steps, 'TR': control_steps},
                'batch_size': args.batch_size, 'samples': args.samples,
                'response_cap': args.response_cap, 'tr_method': args.tr_method,
                'tr_lora_rank': args.tr_lora_rank, 'task': args.task, 'mapping': args.mapping,
                'backend': args.backend, 'control_objective': args.control_objective}
    settings.update(control_validation=training_validation, sc_epochs=3, tr_epochs=3,
                    micro_batch=training_budget['micro_batch'], checkpoint_every=training_budget['checkpoint_interval'],
                    sc_selection_policy=('final_sft_checkpoint_v1' if args.control_objective == 'sft' else 'validation_best_earliest_tie_v1'),
                    tr_selection_policy=('final_sft_checkpoint_v1' if args.control_objective == 'sft' else 'validation_best_earliest_tie_v1'),
                    tr_rollout_backend='vllm' if args.control_objective == 'grpo' else None)
    if tr_validation_samples != args.samples:
        settings['tr_validation_samples'] = tr_validation_samples
    if allow_duplicate_prompts:
        settings['allow_duplicate_prompts'] = True
    if fixed_it_audit is not None:
        settings['it_data_audit'] = fixed_it_audit
    settings['controls'] = list(controls)
    if smoke_steps is not None:
        settings.update(smoke_only=True, test_performed=False, control_epochs=None,
                        sc_epochs=None, tr_epochs=None,
                        sc_selection_policy='final_training_state_no_selection_v1',
                        tr_selection_policy='final_training_state_no_selection_v1')
    _sealed_settings(root, settings)
    base = (PLatourSession.load(args.sender, args.receiver, bridge=args.bridge,
                               mapping=args.mapping) if set(controls) & {'DT', 'IT'} else None)
    statistics_path = root / 'dummy_train_only.safetensors'
    if 'DT' not in controls:
        dummy = None
    elif statistics_path.is_file():
        dummy = DummyStatistics.load(statistics_path)
        if (dummy.training_digest != _fingerprint(train) or
                dummy.sample_count != args.dt_samples or
                dummy.prefix_ids != tuple(args.prefix_ids) or
                dummy.suffix_ids != tuple(args.suffix_ids)):
            raise RuntimeError('Previously signed DT statistics use different TRAIN/template inputs')
    else:
        dummy = build_dummy_statistics(base.bridge, train, prefix_ids=args.prefix_ids,
            suffix_ids=args.suffix_ids, sample_count=args.dt_samples,
            allow_duplicate_prompts=allow_duplicate_prompts)
        root_call(lambda: dummy.save(statistics_path))
    def evaluate_control(system, rows, *, feature_provider):
        label = (getattr(system, 'pact_control', None)
                 or ('DT' if dummy is not None and getattr(feature_provider, '__self__', None) is dummy
                     else 'IT' if feature_provider is not None else None))
        if label not in controls:
            raise RuntimeError('Attempted to evaluate an unselected control')
        if label in ('SC', 'TR'):
            test_identity = hashlib.sha256(json.dumps(
                dict(pact=settings, training=system.training_receipt), sort_keys=True).encode()).hexdigest()
            saved_test = root / 'test' / f'{label}-summary.json'
            if saved_test.is_file():
                saved = json.loads(saved_test.read_text())
                if saved.get('identity') != test_identity:
                    raise RuntimeError(f'{label} TEST cache differs from the selected checkpoint or TEST protocol')
                _accuracy(saved['result'], len(rows) * args.samples)
                return saved['result']
        result = (base if base is not None and system is base.bridge else system).evaluate(
            rows, task=args.task, backend=args.backend, samples=args.samples,
            cap=args.response_cap, feature_provider=feature_provider,
            output=root / 'test' / label, vllm_python=args.vllm_python,
            **({'sampling': fixed_config['sampling']} if fixed_it_audit is not None else {}))
        if label in ('SC', 'TR'):
            _accuracy(result, len(rows) * args.samples)
            saved_test.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(dict(identity=test_identity, evaluation_role='test',
                selected_checkpoint=system.training_receipt, result=result), saved_test)
        return result

    pact = PACTCheck(base.bridge if base is not None else None, train, validation, test, evaluator=evaluate_control,
                     allow_duplicate_prompts=allow_duplicate_prompts)

    def train_sc(training, heldout):
        system = (PLatourSession.self_communication(base.bridge.receiver, base.tokenizer,
                                                    mapping=args.mapping) if base is not None else
                  PLatourSession.load_self_communication(args.receiver, mapping=args.mapping))
        trained = system.fit(training, heldout, output=root / 'SC', **training_validation,
            objective=args.control_objective, **training_budget,
            batch_size=args.batch_size,
            vllm_python=args.vllm_python, response_cap=args.response_cap,
            validation_samples=tr_validation_samples, control='SC')
        system.training_receipt = trained
        system.pact_control = 'SC'
        return system, trained

    def train_tr(training, heldout):
        system = ReceiverSession.load(args.receiver, method=args.tr_method,
                                      lora_rank=args.tr_lora_rank,
                                      lora_alpha=args.tr_lora_rank)
        # fit also verifies completed runs and restores their validation-selected
        # checkpoint; the final optimizer checkpoint is only for crash recovery.
        trained = system.fit(training, heldout, output=root / 'TR',
            **training_budget, batch_size=args.batch_size, control='TR',
            objective=args.control_objective, **training_validation, vllm_python=args.vllm_python,
            validation_samples=tr_validation_samples, response_cap=args.response_cap)
        system.training_receipt = trained
        system.pact_control = 'TR'
        return system, trained

    report = pact.run(dummy=dummy, donors=donors, train_sc=train_sc,
                      train_tr=train_tr, samples_per_question=args.samples, controls=controls,
                      skip_test=smoke_steps is not None)
    report.update(control_objectives={'SC': args.control_objective,
                                      'TR': args.control_objective},
                  control_validation_samples={'SC': tr_validation_samples, 'TR': tr_validation_samples},
                  control_training_steps={'SC': control_steps, 'TR': control_steps},
                  control_validation=training_validation,
                  micro_batch=training_budget['micro_batch'], checkpoint_every=training_budget['checkpoint_interval'],
                  sc_epochs=3, sc_selection_policy=('final_sft_checkpoint_v1' if args.control_objective == 'sft' else 'validation_best_earliest_tie_v1'),
                  tr_epochs=3, tr_selection_policy=('final_sft_checkpoint_v1' if args.control_objective == 'sft' else 'validation_best_earliest_tie_v1'),
                  evaluation_backend=args.backend,
                  protocol_scope='user-selected TRAIN/VAL/TEST and control budgets')
    if fixed_it_audit is not None:
        report['it_data_audit'] = fixed_it_audit
        report['protocol_scope'] = 'frozen archived IT test/pairings; recorded TRAIN/VAL preparation and user-selected control budgets'
    for key in ('control_objectives', 'control_validation_samples', 'control_training_steps'):
        report[key] = {label: value for label, value in report[key].items() if label in controls}
    for label in ('SC', 'TR'):
        if label not in controls:
            report.pop(label.lower() + '_epochs', None)
            report.pop(label.lower() + '_selection_policy', None)
    if sft:
        report.pop('control_validation_samples')  # SFT VAL measures loss, not sampled answers.
    if smoke_steps is not None:
        for label in controls:
            report[label.lower() + '_epochs'] = None
            report[label.lower() + '_selection_policy'] = 'final_training_state_no_selection_v1'
    atomic_json(report, root / 'pact_check.json')
    print(json.dumps({key: report[key] for key in
                      ('controls', 'accuracies', 'responses_per_setting')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
