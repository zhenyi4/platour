"""Dataset-agnostic Qwen3/Qwen3.5 model loading, generation and training API.

The experiment-number CLI remains the fixed paper-reproduction interface. This
API accepts caller-owned models, prompts and datasets; an initialized process
group enables collective training/evaluation (all ranks must call together).
"""

from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import torch
from safetensors.torch import save_file

from qwen3.bridge import Bridge
from qwen3.checkpoints import load_step, save_step
from qwen3.parallel import (api_json as atomic_json, api_write_json, protocol_root,
    sign_topology, append_metric, root_call, root_only, dist, local_device,
    parallel_evaluate, sync_parameters, sum_values, rank_output, save_actor, load_actor)
from qwen3.runner import (clip_flat_gradients, grpo_update, heldout_nll, load_bridge, read_rows,
                          sft_update, reference_sft_numerics)
from qwen3.scoring import evaluation_score
from qwen35.checkpoints import reconcile_history as _reconcile_history, sha256
reconcile_history = root_only(_reconcile_history)


def load_dataset(data):
    rows = read_rows(Path(data)) if isinstance(data, (str, Path)) else list(data)
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get('prompt_token_ids'), list)
                       or len(row['prompt_token_ids']) < 2 for row in rows):
        raise ValueError('Dataset must contain nonempty, fully rendered prompt_token_ids')
    if any(type(row.get('index')) is not int for row in rows) or len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Dataset needs distinct integer question indices')
    return rows


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_session(*, family, sender=None, bridge=None, receiver, method='bridge',
                 task=None, **options):
    """Load every released method role via its Qwen3 or Qwen3.5 text backend."""
    if receiver is None or family not in ('qwen3', 'qwen35'):
        raise ValueError('Specify qwen3 or qwen35 and a public receiver')
    if family == 'qwen3':
        if method == 'self':
            if sender is not None and Path(sender).resolve() != Path(receiver).resolve():
                raise ValueError('Self-Communication uses the receiver as sender')
            sender, method = receiver, 'bridge'
        if method in ('full', 'lora'):
            if sender is not None:
                raise ValueError('Trained Receiver has no external sender')
            session = ReceiverSession.load(receiver, method=method, **options)
            if bridge is not None:
                from safetensors.torch import load_file
                from qwen35.checkpoints import sha256
                session.actor.load_inference_state(load_file(bridge, device='cpu'))
                session.model_identity += ':' + sha256(bridge)
            return session
        if method not in ('bridge', 'sft', 'grpo') or sender is None:
            raise ValueError('Qwen3 supports a sender bridge, GRPO, SFT, or receiver-only control')
        options.setdefault('mapping', 'uniform_endpoints' if method == 'grpo' else 'last_aligned')
        return PLatourSession.load(sender, receiver, bridge=bridge, **options)
    if method in ('self', 'self_communication'):
        if sender is not None and Path(sender).resolve() != Path(receiver).resolve():
            raise ValueError('Self-Communication uses a receiver-as-sender')
        sender, method = receiver, 'bridge'
    elif method in ('virtual', 'virtual_sender'):
        method = 'random'
    if task not in ('dapo', 'deepcoder') or method not in ('bridge', 'random', 'full', 'lora'):
        raise ValueError('Qwen3.5 needs dapo/deepcoder and an original paper method')
    return Qwen35Session.load(sender=sender, bridge=bridge, receiver=receiver,
                              task=task, method=method, **options)


class _UnavailableSender(torch.nn.Module):
    def forward(self, *_args, **_kwargs):
        raise RuntimeError('DT inference cannot query sender weights; supply fixed template means')


class _ConfigOnlySender(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _UnavailableSender()

    @property
    def device(self):
        return torch.device('cpu')


class PLatourSession:
    """A loaded sender + trainable bridge + receiver, independent of experiment IDs."""

    def __init__(self, bridge: Bridge, tokenizer, *, model_identity='user-supplied'):
        self.bridge, self.tokenizer = bridge, tokenizer
        self.model_identity = str(model_identity)
        self.training_receipt = None

    @classmethod
    def load(cls, sender, receiver, bridge=None, *, mapping='last_aligned',
             initialization='zero', device_map='auto', dtype='bfloat16',
             dummy_sender=False):
        if dist().world() > 1 and 'PLATOUR_GPU_GROUP' not in os.environ:
            device_map = local_device(device_map)
        if initialization != 'zero':
            raise ValueError('New bridge projections require zero initialization')
        if device_map != 'auto' and device_map != 'cpu' and not (
                device_map.startswith('cuda:') and device_map[5:].isdigit()):
            raise ValueError('Choose auto, cpu or one visible cuda:N device')
        if dummy_sender:
            if bridge is None:
                raise ValueError('DT must retain the original trained bridge checkpoint')
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            source_config = AutoConfig.from_pretrained(str(sender), local_files_only=True)
            if (getattr(source_config, 'quantization_config', None) or {}).get('quant_method') == 'fp8':
                raise ValueError('Qwen3 FP8 DT needs the audited native prefix layout')
            precision = getattr(torch, dtype)
            if device_map == 'auto' and torch.cuda.device_count() == 1:
                device_map = 'cuda:0'
            if device_map == 'auto':
                target = AutoModelForCausalLM.from_pretrained(
                    str(receiver), torch_dtype=precision, device_map='auto')
            else:
                target = AutoModelForCausalLM.from_pretrained(
                    str(receiver), torch_dtype=precision).to(device_map)
            tokenizer = AutoTokenizer.from_pretrained(str(receiver), local_files_only=True)
            model = Bridge(_ConfigOnlySender(source_config), target, mapping, initialization)
            model.pad_token_id = tokenizer.pad_token_id
        else:
            model, tokenizer = load_bridge(str(sender), str(receiver), mapping,
                                           initialization, dtype, device_map=device_map)
        if bridge is not None:
            model.load_public(Path(bridge))
        return cls(model, tokenizer, model_identity=json.dumps({
            'sender': str(Path(sender).resolve()), 'receiver': str(Path(receiver).resolve()),
            'mapping': mapping, 'initial_bridge_sha256': sha256(bridge) if bridge is not None else None},
            sort_keys=True))

    @classmethod
    def self_communication(cls, receiver, tokenizer, *, mapping='last_aligned'):
        """SC: freeze one receiver as *both* sender and receiver; train a separate bridge."""
        bridge = Bridge(receiver, receiver, mapping)
        bridge.pad_token_id = tokenizer.pad_token_id
        return cls(bridge, tokenizer, model_identity='self:' + str(receiver.config._name_or_path))

    @classmethod
    def load_self_communication(cls, receiver, *, mapping='last_aligned',
                                device_map='auto', dtype='bfloat16'):
        """Load one frozen receiver shared by SC's sender and receiver roles."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device_map = local_device(device_map)
        if device_map != 'auto' and device_map != 'cpu' and not (
                device_map.startswith('cuda:') and device_map[5:].isdigit()):
            raise ValueError('Choose auto, cpu or one visible cuda:N device')
        if device_map == 'auto' and torch.cuda.device_count() == 1:
            device_map = 'cuda:0'
        options = dict(torch_dtype=getattr(torch, dtype), local_files_only=True)
        if device_map == 'auto':
            model = AutoModelForCausalLM.from_pretrained(str(receiver), device_map='auto', **options)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(receiver), **options).to(device_map)
        tokenizer = AutoTokenizer.from_pretrained(str(receiver), local_files_only=True)
        return cls.self_communication(model, tokenizer, mapping=mapping)

    def prompt_ids(self, *, prompt_ids=None, messages=None):
        if (prompt_ids is None) == (messages is None):
            raise ValueError('Provide exactly one of prompt_ids or messages')
        if messages is not None:
            if (not isinstance(messages, list) or not messages or
                    any(not isinstance(row, dict) or row.get('role') not in
                        ('system', 'user', 'assistant') or not isinstance(row.get('content'), str)
                        for row in messages)):
                raise ValueError('messages must contain explicit chat roles and text content')
            encoded = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            prompt_ids = encoded['input_ids'] if hasattr(encoded, 'keys') and 'input_ids' in encoded else encoded
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        if (not isinstance(prompt_ids, list) or len(prompt_ids) < 2 or
                any(type(token) is not int or token < 0 for token in prompt_ids)):
            raise ValueError('Prompt must contain at least two nonnegative token IDs')
        return prompt_ids

    def generate(self, *, prompt_ids=None, messages=None, backend='vllm',
                 max_new_tokens=256, seed=42, temperature=0.7, top_p=0.8, top_k=20,
                 vllm_python=None, output=None, feature_provider=None, memory_fraction=.25):
        prompt = self.prompt_ids(prompt_ids=prompt_ids, messages=messages)
        if max_new_tokens < 1 or not 0 < memory_fraction < 1:
            raise ValueError('Positive response budget and a valid vLLM memory fraction required')
        scope = self.bridge.use_features(feature_provider) if feature_provider else nullcontext()
        with scope:
            if backend == 'hf':
                raise ValueError('Answer generation requires vLLM')
            elif backend == 'vllm':
                from platour_exps import require_vllm_python
                from qwen3.vllm_decode import VLLMDecoder
                if output is None:
                    raise ValueError('Native vLLM needs a caller-owned output directory')
                require_vllm_python(vllm_python)
                directory = Path(output)
                directory.mkdir(parents=True, exist_ok=True)
                with VLLMDecoder(self.bridge, directory,
                                 max_model_len=max(3072, len(prompt) + max_new_tokens),
                                 memory_fraction=memory_fraction, worker_python=str(vllm_python) if vllm_python else None,
                                 max_num_seqs=1, prefill_batch_size=1) as decoder:
                    answer = decoder.generate([{'index': 0, 'prompt_token_ids': prompt}], count=1,
                        cap=max_new_tokens, sampling={'temperature': temperature, 'top_p': top_p,
                                                       'top_k': top_k}, seed_override=seed)[0]
                tokens = answer['response_token_ids']
                text = answer['response']
            else:
                raise ValueError('backend must be vllm')
        return {'response': text, 'response_token_ids': tokens,
                'backend': backend, 'prompt_tokens': len(prompt), 'response_tokens': len(tokens)}

    @parallel_evaluate
    def evaluate(self, data, *, task=None, scorer=None, backend='vllm',
                 samples=1, cap=256, output=None, feature_provider=None,
                 vllm_python=None, memory_fraction=.25,
                 sampling=None):
        rows = load_dataset(data)
        if scorer is None and task is None:
            raise ValueError('Specify a task or a scorer for independent validation')
        if not isinstance(samples, int) or samples < 1 or cap < 1:
            raise ValueError('Sample and response budgets must be positive')
        sampling = sampling or {'temperature': .7, 'top_p': .8, 'top_k': 20}
        check = scorer or (lambda row, text: evaluation_score(
            task, text, row['gold'], valid_labels=row.get('valid_labels', row.get('labels'))))
        scope = self.bridge.use_features(feature_provider) if feature_provider else nullcontext()
        with scope:
            if backend == 'vllm':
                from platour_exps import require_vllm_python
                from qwen3.vllm_decode import decoder_session
                if output is None:
                    raise ValueError('Native vLLM validation needs a separate output root')
                require_vllm_python(vllm_python)
                Path(output).mkdir(parents=True, exist_ok=True)
                with decoder_session(self.bridge, Path(output),
                                 max_model_len=max(3072, max(map(lambda r: len(r['prompt_token_ids']), rows)) + cap),
                                 memory_fraction=memory_fraction,
                                 worker_python=str(vllm_python) if vllm_python else None) as decoder:
                    generated = decoder.generate(rows, count=samples, cap=cap, sampling=sampling)
                expected = [(r['index'], k) for r in rows for k in range(samples)]
                if [(r['index'], r['sample_index']) for r in generated] != expected:
                    raise RuntimeError('vLLM validation response/question order differs')
                texts = [row['response'] for row in generated]
            elif backend == 'hf':
                raise ValueError('Answer generation requires vLLM')
            else:
                raise ValueError('backend must be vllm')
        correct = 0
        for position, text in enumerate(texts):
            result = check(rows[position // samples], text)
            if result not in (0, 1):
                raise ValueError('PACT/evaluation scorer must return 0 or 1 per response')
            correct += result
        denominator = len(rows) * samples
        if len(texts) != denominator:
            raise RuntimeError('Validation or test response denominator incomplete')
        return {'complete': True, 'questions': len(rows), 'responses': denominator,
                'correct': correct, 'accuracy': correct / denominator, 'backend': backend}

    def fit(self, train_data, validation_data, *, output, task=None, objective='sft',
            max_steps=None, epochs=3, smoke=False, batch_size=8, micro_batch=1, learning_rate=None,
            validation_backend=None, validation_samples=1, response_cap=256,
            vllm_python=None, reward_fn=None, control=None, validation_schedule='auto', checkpoint_interval=0, warmup_steps=None,
            rollout_worker_mode='persistent'):
        """Train P only. Sign optimizer/RNG at validation/recovery points; resume same root.

        Training defaults to three complete epochs; SFT restores the final bridge,
        GRPO the validation-best bridge. SFT defaults to peak LR 1e-4 and linear
        warmup (ceil(10% of steps)) then decay; warmup_steps overrides the count.
        An explicit max_steps requires smoke=True for bounded API checks.
        SFT uses the reference arithmetic, micro-batch layout, flattened gradient clipping and
        token-weighted held-out NLL (HF teacher forcing, with or without a task).
        The default validation schedule is epoch-end for SFT and half-epoch for GRPO,
        both including step zero. Explicit 'step' validation is available for debugging.
        ``validation_schedule='epoch'`` records initial and epoch-end validation;
        SFT always uses the final checkpoint, as used by Qwen3 PACT SFT.
        GRPO reuses one vLLM receiver during fit, sleeping between rollout/validation
        phases. rollout_worker_mode='restart' retains the previous per-use worker.
        """
        from qwen3.vllm_decode import training_decoder_scope
        if objective == 'grpo':
            train_data, validation_data = load_dataset(train_data), load_dataset(validation_data)
            scope = training_decoder_scope(self.bridge, rank_output(Path(output) / 'vllm-worker'),
                max_model_len=max(3072, max(len(row['prompt_token_ids']) for row in
                    [*train_data, *validation_data]) + response_cap), mode=rollout_worker_mode)
        else:
            scope = reference_sft_numerics() if objective == 'sft' else nullcontext()
        with scope:
            return self._fit(train_data, validation_data, output=output, task=task,
                             objective=objective, max_steps=max_steps, epochs=epochs, smoke=smoke, batch_size=batch_size,
                             micro_batch=micro_batch, learning_rate=learning_rate, warmup_steps=warmup_steps,
                             validation_backend=validation_backend,
                             validation_samples=validation_samples, response_cap=response_cap,
                             vllm_python=vllm_python, reward_fn=reward_fn, control=control,
                             validation_schedule=validation_schedule, checkpoint_interval=checkpoint_interval,
                             rollout_worker_mode=rollout_worker_mode)

    def _fit(self, train_data, validation_data, *, output, task, objective, max_steps, epochs, smoke,
             batch_size, micro_batch, learning_rate, warmup_steps, validation_backend, validation_samples,
             response_cap, vllm_python, reward_fn, control, validation_schedule, checkpoint_interval,
             rollout_worker_mode):
        if isinstance(self.bridge.sender, _ConfigOnlySender):
            raise ValueError('DT is inference-only; training requires actual sender weights')
        from training_protocol import training_learning_rate, sft_schedule
        learning_rate = training_learning_rate(objective, 'bridge', learning_rate)
        train, heldout = load_dataset(train_data), load_dataset(validation_data)
        select_best = True
        from training_protocol import budget, validation_backend as check_backend
        validation_backend = check_backend(validation_backend, objective, task)
        budget(len(train), batch_size, epochs, 3, max_steps, smoke)
        steps_per_epoch = (len(train) + batch_size - 1) // batch_size
        budget_epochs = epochs if max_steps is None else None
        if max_steps is None:
            max_steps = epochs * steps_per_epoch if epochs is not None else 1
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('Optimizer step budget must be positive')
        from training_protocol import validation_steps, default_validation_schedule
        if validation_schedule == 'auto':
            validation_schedule = default_validation_schedule(objective)
        val_steps = validation_steps(validation_schedule, objective, validation_backend, task,
                                     steps_per_epoch, max_steps, smoke=smoke)
        selection_steps = val_steps[1:] if validation_schedule in ('epoch', 'initial') else val_steps
        from training_protocol import checkpoint_steps
        save_steps = checkpoint_steps(val_steps, max_steps, checkpoint_interval,
                                      save_initial=0 in selection_steps)
        selection_metric = 'heldout_nll' if objective == 'sft' else 'accuracy'
        if (not {tuple(row['prompt_token_ids']) for row in train}.isdisjoint(
                {tuple(row['prompt_token_ids']) for row in heldout}) or
                objective not in ('sft', 'grpo') or max_steps < 1 or batch_size < 1 or
                micro_batch < 1 or learning_rate <= 0 or validation_samples < 1 or response_cap < 1 or
                validation_backend not in ('hf', 'vllm')):
            raise ValueError('Invalid independent training/validation split or schedule')
        if objective == 'sft' and any(not row.get('response_token_ids') for row in train):
            raise ValueError('SFT needs teacher response_token_ids for each TRAIN example')
        if objective == 'sft' and any(not row.get('response_token_ids') for row in heldout):
            raise ValueError('SFT epoch validation needs supervised VAL responses')
        if objective == 'grpo' and task is None and reward_fn is None:
            raise ValueError('GRPO needs a task or a named custom reward')
        if reward_fn is not None:
            try:
                reward_source = inspect.getsource(reward_fn)
            except (OSError, TypeError) as error:
                raise ValueError('Custom reward requires a stable inspectable source') from error
            reward_hash = hashlib.sha256(reward_source.encode()).hexdigest()
        else:
            reward_hash = None
        protocol = {'train_sha256': _digest(train), 'validation_sha256': _digest(heldout),
                    'model_identity': self.model_identity, 'task': task, 'objective': objective,
                    'max_steps': max_steps, 'batch_size': batch_size, 'micro_batch': micro_batch,
                    'learning_rate': learning_rate, 'validation_backend': validation_backend,
                    'validation_samples': validation_samples, 'response_cap': response_cap,
                    'reward_source_sha256': reward_hash, 'control': control, 'smoke_only': smoke,
                    'rollout_backend': 'vllm' if objective == 'grpo' else None,
                    'rollout_group_size': 8 if objective == 'grpo' else None,
                    'rollout_sampling': dict(temperature=1., top_p=1., top_k=-1) if objective == 'grpo' else None}
        protocol.update(sft_schedule(objective, max_steps, warmup_steps))
        if objective == 'grpo':
            previous = Path(output) / 'protocol.json'
            legacy_restart = (rollout_worker_mode == 'restart' and previous.is_file()
                              and 'rollout_worker_mode' not in json.loads(previous.read_text()))
            if not legacy_restart:
                protocol['rollout_worker_mode'] = rollout_worker_mode
        if select_best:
            protocol.update(epochs=budget_epochs, steps_per_epoch=steps_per_epoch,
                selection_policy=('final_sft_checkpoint_v1' if objective == 'sft' else 'validation_best_earliest_tie_v1'), selection_metric=selection_metric,
                test_used_for_selection=False)
        if validation_schedule == 'initial':
            protocol['selection_policy'] = 'final_training_state_no_selection_v1'
        protocol.update(validation_schedule=validation_schedule, validation_steps=val_steps,
                        selection_steps=selection_steps, checkpoint_interval=checkpoint_interval,
                        checkpoint_steps=save_steps)
        sign_topology(protocol)
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        root = Path(output).absolute()
        original = root / 'protocol.json'
        protocol_root(root, protocol)
        sync_parameters(self.bridge.P.parameters())
        optimizer = torch.optim.AdamW(self.bridge.P.parameters(), lr=learning_rate,
                                      betas=(.9, .999), weight_decay=0.)
        from training_protocol import sft_lr_factor
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            (lambda completed: sft_lr_factor(completed, max_steps, protocol['warmup_steps']))
            if objective == 'sft' else (lambda _: 1.))
        device = self.bridge.receiver.device
        cuda = (device.index if device.index is not None else torch.cuda.current_device(),) if device.type == 'cuda' else ()
        checkpoints = root / 'checkpoints'
        def candidate(step):
            import math
            path = root / 'evaluation' / f'step-{step:04d}.json'
            value = json.loads(path.read_text())
            score = value.get(selection_metric)
            if (not value.get('complete') or value.get('step') != step or
                    value.get('evaluation_role') != ('initial' if step == 0 and validation_schedule in ('epoch', 'initial') else 'validation') or
                    value.get('protocol_sha256') != identity or
                    value.get('backend') != validation_backend or
                    value.get('questions') != len(heldout) or
                    type(score) not in (int, float) or not math.isfinite(score)):
                raise RuntimeError('SC validation history is incomplete or differs from its protocol')
            if selection_metric == 'accuracy' and (not 0 <= score <= 1 or
                    value.get('responses') != len(heldout) * validation_samples or
                    type(value.get('correct')) is not int or
                    not math.isclose(value['correct'], score * value['responses'])):
                raise RuntimeError('SC validation response denominator or accuracy differs')
            return dict(step=step, score=score, validation_sha256=sha256(path),
                        checkpoint_sha256=value['checkpoint_sha256'])

        def choose(candidates):
            if objective == 'sft':
                return max(candidates, key=lambda c: c['step'])
            return min(candidates, key=lambda c: (
                c['score'] if selection_metric == 'heldout_nll' else -c['score'], c['step']))

        def selection_record():
            if validation_schedule == 'initial':
                from training_protocol import training_only_record
                return training_only_record(root, 'bridge.pt', max_steps, identity, candidate(0))
            if 0 in val_steps:
                candidate(0)  # Verify initial measurements even when ineligible for selection.
            candidates = [candidate(step) for step in selection_steps]
            best = choose(candidates)
            return dict(policy=protocol['selection_policy'], metric=selection_metric,
                best_step=best['step'], best_score=best['score'], training_steps=max_steps,
                checkpoint=f"checkpoints/step-{best['step']:04d}",
                checkpoint_sha256=best['checkpoint_sha256'], protocol_sha256=identity,
                test_used_for_selection=False, candidates=candidates)

        def restore_selected(selection):
            path = root / selection['checkpoint']
            if (not (path / 'complete.json').is_file() or
                    not (path / 'bridge.pt').is_file() or
                    sha256(path / 'bridge.pt') != selection['checkpoint_sha256']):
                raise RuntimeError('SC validation-selected checkpoint is missing or corrupt')
            step = load_step(path, self.bridge, optimizer, scheduler)
            if step != selection['best_step']:
                raise RuntimeError('SC selected checkpoint step differs')
            self.training_receipt = dict(trained=True, control=control,
                train_digest=_digest(train), validation_digest=_digest(heldout),
                uses_test_for_training=False, checkpoint=str(path), selected_step=step,
                checkpoint_sha256=selection['checkpoint_sha256'], training_steps=max_steps)
            if validation_schedule == 'initial':
                self.training_receipt.update(training_only=True, selection_performed=False,
                    selection_policy=protocol['selection_policy'])
            return self.training_receipt

        @root_only
        def prune(step, best):
            import shutil
            committed = sorted(int(p.name[5:]) for p in checkpoints.glob('step-*')
                               if p.name[5:].isdigit() and (p / 'complete.json').is_file())
            keep = {0, best['step'] if best else 0, *committed[-2:]}
            for directory in checkpoints.glob('step-*'):
                if not directory.name[5:].isdigit() or directory.is_symlink():
                    continue
                old = int(directory.name[5:])
                marker = directory / 'complete.json'
                if old < step - 1 and old not in keep and marker.is_file():
                    record = json.loads(marker.read_text())
                    if record.get('complete') and record.get('step') == old:
                        shutil.rmtree(directory)

        if select_best and (root / 'complete.json').is_file():
            selection = selection_record()
            terminal = json.loads((root / 'complete.json').read_text())
            if (json.loads((root / 'selection.json').read_text()) != selection or
                    terminal != dict(complete=True, steps=max_steps, checkpoint=selection['checkpoint'],
                                     selected_step=selection['best_step'], selection_sha256=sha256(root / 'selection.json'))):
                raise RuntimeError('SC completed selection differs from validation history')
            return restore_selected(selection)

        def resume_markers():
            signed = sorted(checkpoints.glob('step-*/complete.json')) if checkpoints.is_dir() else []
            if not signed and checkpoints.is_dir() and any(checkpoints.iterdir()):
                raise RuntimeError('Unsigned checkpoint; use a new run, do not guess optimizer state')
            return signed
        signed = root_call(resume_markers)
        if not signed:
            torch.manual_seed(42)
            random.seed(42)
            if 0 in save_steps:
                save_step(root, 0, self.bridge, optimizer, scheduler, cuda_devices=cuda)
            else:
                from training_protocol import restart_before_first_checkpoint
                root_call(lambda: restart_before_first_checkpoint(root))
            step0 = 0
        else:
            step0 = max(int(path.parent.name[5:]) for path in signed)
            step0 = load_step(checkpoints / f'step-{step0:04d}', self.bridge, optimizer, scheduler)
            marker = json.loads((checkpoints / f'step-{step0:04d}/complete.json').read_text())
            reconcile_history(root, step0, marker.get('metric'))
            @root_only
            def archive_unsigned():
                for directory in checkpoints.glob('step-*'):
                    if int(directory.name[5:]) > step0:
                        archive = root / 'recovery'
                        archive.mkdir(exist_ok=True)
                        directory.rename(archive / f'{directory.name}-unsigned-{time.time_ns()}')
            archive_unsigned()
        if step0 > max_steps:
            raise RuntimeError('Checkpoint exceeds the specified training budget')

        def validate(step):
            path = root / 'evaluation' / f'step-{step:04d}.json'
            checkpoint_hash = (sha256(checkpoints / f'step-{step:04d}/bridge.pt')
                               if step in save_steps else None)
            if path.is_file():
                if select_best:
                    saved = candidate(step)
                    if saved['checkpoint_sha256'] != checkpoint_hash:
                        raise RuntimeError('SC validation was scored with a different checkpoint')
                    return saved
                return
            if objective == 'sft':
                nll, tokens = heldout_nll(self.bridge, heldout, micro_batch)
                value = {'complete': True, 'questions': len(heldout), 'heldout_nll': nll,
                         'response_tokens': tokens, 'backend': 'hf'}
            else:
                value = self.evaluate(heldout, task=task, backend=validation_backend,
                    samples=validation_samples, cap=response_cap,
                    output=root / 'evaluation' / f'step-{step:04d}-vllm',
                    vllm_python=vllm_python)
            path.parent.mkdir(exist_ok=True)
            if select_best:
                value = dict(value, protocol_sha256=identity,
                             checkpoint_sha256=checkpoint_hash)
            role = 'initial' if step == 0 and validation_schedule in ('epoch', 'initial') else 'validation'
            atomic_json({'step': step, 'evaluation_role': role, **value}, path)
            return candidate(step) if select_best else None

        if step0 in val_steps:
            validate(step0)
        if 0 in val_steps:
            candidate(0)
        previous_candidates = [candidate(step) for step in selection_steps if step <= step0]
        best = choose(previous_candidates) if previous_candidates else None
        for step in range(step0 + 1, max_steps + 1):
            epoch, batch = divmod(step - 1, steps_per_epoch)
            offset = batch * batch_size
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            indices = order[offset:offset + batch_size]
            if not indices:
                raise RuntimeError('Empty independent training batch')
            optimizer.zero_grad(set_to_none=True)
            if objective == 'sft':
                total = sum(len(train[index]['response_token_ids']) for index in indices)
                nll, tokens = sft_update(self.bridge, optimizer, train, indices, micro_batch, total)
                nll, tokens = sum_values((nll, tokens), self.bridge.receiver.device)
                record = {'step': step, 'train_nll': nll / total, 'response_tokens': tokens}
            else:
                config = {'task': task, 'group_size': 8, 'response_cap': response_cap,
                          'rollout_output': str(root / 'rollouts'), 'vllm_python': vllm_python,
                          'train_sampling': {'temperature': 1., 'top_p': 1., 'top_k': -1}}
                loss, correct = grpo_update(self.bridge, self.tokenizer, train, dist().local_slice(indices),
                                             config, step, len(indices), reward_fn=reward_fn)
                loss, correct = sum_values((loss, correct), self.bridge.receiver.device)
                from qwen3.runner import synchronize_gradients
                synchronize_gradients(self.bridge)
                record = {'step': step, 'grpo_loss': loss, 'correct': correct,
                          'responses': len(indices) * 8}
            norm = (clip_flat_gradients(self.bridge.P.parameters()) if objective == 'sft' else
                    torch.nn.utils.clip_grad_norm_(self.bridge.P.parameters(), 1.,
                                                   error_if_nonfinite=True))
            record['learning_rate'] = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            record['grad_norm'] = float(norm)
            if select_best:
                record.update(epoch=epoch + 1, train_indices=[train[i]['index'] for i in indices])
            if step in save_steps:
                save_step(root, step, self.bridge, optimizer, scheduler, cuda_devices=cuda, record=record)
            append_metric(root, record)
            validation = validate(step) if step in val_steps else None
            if select_best:
                if validation is not None:
                    best = choose(([best] if best else []) + [validation])
                if step in save_steps:
                    prune(step, best)
        if select_best:
            selection = selection_record()
            receipt = restore_selected(selection)
            atomic_json(selection, root / 'selection.json')
            atomic_json(dict(complete=True, steps=max_steps, checkpoint=selection['checkpoint'],
                selected_step=selection['best_step'], selection_sha256=sha256(root / 'selection.json')),
                root / 'complete.json')
            return receipt
        atomic_json({'complete': True, 'steps': max_steps, 'checkpoint':
                     f'checkpoints/step-{max_steps:04d}'}, root / 'complete.json')
        self.training_receipt = {'trained': True, 'control': control,
                                 'train_digest': _digest(train),
                                 'validation_digest': _digest(heldout),
                                 'uses_test_for_training': False,
                                 'checkpoint': str(root / 'complete.json')}
        return self.training_receipt

    def save_bridge(self, path):
        output = Path(path)
        if output.exists():
            raise FileExistsError('Never overwrite an earlier bridge')
        state = {f'P.{name}': value.detach().cpu().contiguous()
                 for name, value in self.bridge.P.state_dict().items()}
        save_file(state, output, metadata={'layer_mapping': self.bridge.mapping,
            'layer_map_0_based': json.dumps(self.bridge.indices, separators=(',', ':'))})
        return output


class ReceiverSession:
    """Full/LoRA receiver training with vLLM GRPO and validation-best selection."""

    def __init__(self, receiver, tokenizer, *, method='full', lora_rank=128,
                 lora_alpha=128, model_identity='user-supplied'):
        from types import SimpleNamespace
        from torch import nn
        from qwen35.model import Actor
        if method not in ('full', 'lora'):
            raise ValueError('TR method must be full or lora')
        targets = sum(isinstance(module, nn.Linear) and name.startswith('model.layers.')
                      for name, module in receiver.named_modules())
        config = SimpleNamespace(method=method, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                 lora_target_count=targets)
        self.actor = Actor(config, receiver)
        self.tokenizer = tokenizer
        self.method = method
        self.model_identity = model_identity
        self.model_path = Path(model_identity) if Path(model_identity).is_dir() else None
        self.training_receipt = None

    @classmethod
    def load(cls, receiver, *, method='full', lora_rank=128, lora_alpha=128,
             device='auto', dtype='bfloat16'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = local_device(device)
        if device != 'auto' and device != 'cpu' and not (
                device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError('TR device must be auto, cpu or one visible CUDA card')
        if device == 'auto' and torch.cuda.device_count() == 1:
            device = 'cuda:0'
        if device == 'auto':
            model = AutoModelForCausalLM.from_pretrained(
                str(receiver), torch_dtype=getattr(torch, dtype), device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(receiver), torch_dtype=getattr(torch, dtype)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(receiver), local_files_only=True)
        return cls(model, tokenizer, method=method, lora_rank=lora_rank,
                   lora_alpha=lora_alpha, model_identity=str(Path(receiver).resolve()))

    def _decoder(self, output, cap, prompt_length, vllm_python, memory_fraction):
        from qwen3.vllm_decode import ReceiverOnlyDecoder
        if self.model_path is None or output is None:
            raise ValueError('Native receiver-only vLLM needs a local model and private output')
        from platour_exps import require_vllm_python
        require_vllm_python(vllm_python)
        Path(output).mkdir(parents=True, exist_ok=True)
        return ReceiverOnlyDecoder(self.actor, self.model_path, output,
            max_model_len=max(3072, prompt_length + cap),
            memory_fraction=memory_fraction, worker_python=str(vllm_python) if vllm_python else None)

    def generate(self, *, prompt_ids=None, messages=None, backend='vllm', max_new_tokens=256,
                 seed=42, temperature=.7, top_p=.8, top_k=20, output=None,
                 vllm_python=None, memory_fraction=.25):
        tokens = PLatourSession.prompt_ids(self, prompt_ids=prompt_ids, messages=messages)
        if max_new_tokens < 1:
            raise ValueError('TR needs a positive response cap')
        if backend == 'hf':
            raise ValueError('Answer generation requires vLLM')
        elif backend == 'vllm':
            with self._decoder(output, max_new_tokens, len(tokens), vllm_python,
                               memory_fraction) as decoder:
                row = decoder.generate([{'index': 0, 'prompt_token_ids': tokens}],
                    count=1, cap=max_new_tokens, seed_override=seed,
                    sampling={'temperature': temperature, 'top_p': top_p, 'top_k': top_k})[0]
            answer, text = row['response_token_ids'], row['response']
        else:
            raise ValueError('TR generation backend must be vllm')
        return {'response': text, 'response_token_ids': answer, 'backend': backend}

    @parallel_evaluate
    def evaluate(self, data, *, task=None, scorer=None, samples=1, cap=256,
                 backend='vllm', feature_provider=None, output=None, vllm_python=None,
                 memory_fraction=.25, sampling=None):
        if feature_provider is not None or (task is None and scorer is None):
            raise ValueError('TR evaluation needs a scorer and no sender intervention')
        rows = load_dataset(data)
        if samples < 1 or cap < 1:
            raise ValueError('TR samples/cap must be positive')
        sampling = sampling or {'temperature': .7, 'top_p': .8, 'top_k': 20}
        if backend == 'vllm':
            with self._decoder(output, cap, max(len(row['prompt_token_ids']) for row in rows),
                               vllm_python, memory_fraction) as decoder:
                answers = decoder.generate(rows, count=samples, cap=cap, sampling=sampling)
            texts = [item['response'] for item in answers]
        elif backend == 'hf':
            raise ValueError('Answer generation requires vLLM')
        else:
            raise ValueError('TR evaluation backend must be vllm')
        denominator = len(rows) * samples
        if len(texts) != denominator:
            raise RuntimeError('TR validation/test response denominator incomplete')
        correct = 0
        for index, text in enumerate(texts):
            row = rows[index // samples]
            result = (scorer(row, text) if scorer is not None else evaluation_score(
                task, text, row['gold'], valid_labels=row.get('valid_labels', row.get('labels'))))
            if result not in (0, 1):
                raise ValueError('TR scorer must return one binary verdict per response')
            correct += result
        return {'complete': True, 'questions': len(rows), 'responses': denominator,
                'correct': correct, 'accuracy': correct / denominator, 'backend': backend}

    def fit(self, train_data, validation_data, *, output, max_steps=None, epochs=3, smoke=False,
            batch_size=1, micro_batch=1, learning_rate=None, control='TR', objective='sft',
            task=None, reward_fn=None, validation_backend=None, vllm_python=None,
            validation_samples=1, response_cap=256, validation_schedule='auto', checkpoint_interval=0, warmup_steps=None):
        """Train a receiver; restore final SFT or validation-best GRPO state for TEST.

        SFT defaults to peak LR 1e-5 (full) or 1e-4 (LoRA), with linear warmup
        (ceil(10% of steps)) then decay. warmup_steps overrides the count.

        Default budget is three complete epochs, including each epoch's short final
        batch. Explicit max_steps requires smoke=True for bounded API checks.
        GRPO rollouts always use a fresh vLLM snapshot of the current actor.
        SFT accumulates padded micro-batches, normalized by the effective batch's
        total response tokens, and updates once per batch.
        Default validation is epoch-end for SFT and half-epoch for GRPO, including
        step zero. SFT epoch validation excludes the initial measurement from selection.
        Explicit 'step' validation is available for debugging.
        """
        import math
        from torch.nn import functional as F
        from qwen35.checkpoints import latest, valid
        write_json, load_step, save_step = api_write_json, load_actor, save_actor
        from training_protocol import validation_backend as check_backend
        validation_backend = check_backend(validation_backend, objective, task)
        from training_protocol import training_learning_rate, sft_schedule, set_sft_learning_rate
        learning_rate = training_learning_rate(objective, self.method, learning_rate)
        train, heldout = load_dataset(train_data), load_dataset(validation_data)
        if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1 or
                type(micro_batch) is not int or micro_batch < 1 or
                learning_rate <= 0 or objective not in ('sft', 'grpo') or
                validation_backend not in ('hf', 'vllm') or validation_samples < 1 or response_cap < 1 or
                not {tuple(row['prompt_token_ids']) for row in train}.isdisjoint(
                    {tuple(row['prompt_token_ids']) for row in heldout})):
            raise ValueError('Receiver-only control needs disjoint TRAIN/VAL and valid budgets')
        from training_protocol import budget, validation_backend as check_backend
        validation_backend = check_backend(validation_backend, objective, task)
        budget(len(train), batch_size, epochs, 3, max_steps, smoke)
        steps_per_epoch = (len(train) + batch_size - 1) // batch_size
        budget_epochs = epochs if max_steps is None else None
        max_steps = epochs * steps_per_epoch if max_steps is None else max_steps
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('TR optimizer step budget must be positive')
        from training_protocol import validation_steps, default_validation_schedule
        if validation_schedule == 'auto':
            validation_schedule = default_validation_schedule(objective)
        val_steps = validation_steps(validation_schedule, objective, validation_backend, task,
                                     steps_per_epoch, max_steps, smoke=smoke)
        selection_steps = val_steps[1:] if validation_schedule in ('epoch', 'initial') else val_steps
        from training_protocol import checkpoint_steps
        save_steps = checkpoint_steps(val_steps, max_steps, checkpoint_interval,
                                      save_initial=0 in selection_steps)
        if objective == 'sft' and any(not row.get('response_token_ids') for row in train + heldout):
            raise ValueError('Receiver-only SFT needs supervised TRAIN/VAL responses')
        if (objective == 'grpo' or validation_backend == 'vllm') and (
                task is None or any(not isinstance(row.get('gold'), str) for row in heldout)):
            raise ValueError('Response validation requires a task and VAL gold labels')
        if objective == 'grpo' and reward_fn is None and any(
                not isinstance(row.get('gold'), str) for row in train):
            raise ValueError('GRPO needs TRAIN gold labels or a fixed reward function')
        if reward_fn is not None:
            try:
                reward_source = inspect.getsource(reward_fn)
            except (OSError, TypeError) as error:
                raise ValueError('Custom GRPO reward requires inspectable stable source') from error
            reward_hash = hashlib.sha256(reward_source.encode()).hexdigest()
        else:
            reward_hash = None
        metric = 'heldout_nll' if objective == 'sft' else 'accuracy'
        protocol = {'model_identity': self.model_identity, 'method': self.method,
                    'train_sha256': _digest(train), 'validation_sha256': _digest(heldout),
                    'max_steps': max_steps, 'epochs': budget_epochs, 'steps_per_epoch': steps_per_epoch,
                    'batch_size': batch_size, 'learning_rate': learning_rate,
                    'lora_rank': getattr(self.actor.experiment, 'lora_rank'),
                    'lora_alpha': getattr(self.actor.experiment, 'lora_alpha'), 'control': control,
                    'objective': objective, 'task': task, 'reward_source_sha256': reward_hash,
                    'validation_backend': validation_backend, 'validation_samples': validation_samples,
                    'response_cap': response_cap, 'selection_policy': ('final_sft_checkpoint_v1' if objective == 'sft' else 'validation_best_earliest_tie_v1'),
                    'selection_metric': metric, 'test_used_for_selection': False,
                    'smoke_only': smoke, 'rollout_backend': 'vllm' if objective == 'grpo' else None,
                    'rollout_samples': 8 if objective == 'grpo' else None,
                    'rollout_sampling': {'temperature': 1., 'top_p': 1., 'top_k': -1} if objective == 'grpo' else None}
        protocol.update(sft_schedule(objective, max_steps, warmup_steps))
        if objective == 'sft':
            protocol.update(sft_accumulation='padded_micro_batch_response_token_mean_v2', micro_batch=micro_batch)
        if validation_schedule == 'initial':
            protocol['selection_policy'] = 'final_training_state_no_selection_v1'
        protocol.update(validation_schedule=validation_schedule, validation_steps=val_steps,
                        selection_steps=selection_steps, checkpoint_interval=checkpoint_interval,
                        checkpoint_steps=save_steps)
        sign_topology(protocol)
        root = Path(output).absolute()
        protocol_root(root, protocol)
        sync_parameters(self.actor.trainable())
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        optimizer = torch.optim.AdamW(self.actor.trainable(), lr=learning_rate,
                                      betas=(.9, .999), weight_decay=0.)
        set_sft_learning_rate(optimizer, protocol, 0)

        def candidate(step):
            path = root / 'evaluation' / f'step-{step:04d}.json'
            result = json.loads(path.read_text())
            score = result.get(metric)
            if (not result.get('complete') or result.get('step') != step or
                    result.get('evaluation_role') != ('initial' if step == 0 and validation_schedule in ('epoch', 'initial') else 'validation') or
                    result.get('protocol_sha256') != identity or
                    result.get('backend') != validation_backend or
                    result.get('questions') != len(heldout) or
                    type(score) not in (int, float) or not math.isfinite(score)):
                raise RuntimeError('TR validation history is incomplete or differs from its protocol')
            if metric == 'accuracy' and (not 0 <= score <= 1 or
                    result.get('responses') != len(heldout) * validation_samples or
                    type(result.get('correct')) is not int or
                    not math.isclose(result['correct'], score * result['responses'])):
                raise RuntimeError('TR validation response denominator or accuracy differs')
            return {'step': step, 'score': score, 'validation_sha256': sha256(path),
                    'checkpoint_sha256': result['checkpoint_sha256']}

        def choose(candidates):
            if objective == 'sft':
                return max(candidates, key=lambda c: c['step'])
            return min(candidates, key=lambda c: (c['score'] if metric == 'heldout_nll' else -c['score'], c['step']))

        def selection_record():
            if validation_schedule == 'initial':
                from training_protocol import training_only_record
                return training_only_record(root, 'model.safetensors', max_steps, identity, candidate(0))
            if 0 in val_steps:
                candidate(0)
            candidates = [candidate(step) for step in selection_steps]
            best = choose(candidates)
            return dict(policy=protocol['selection_policy'], metric=metric,
                best_step=best['step'], best_score=best['score'], training_steps=max_steps,
                checkpoint=f"checkpoints/step-{best['step']:04d}",
                checkpoint_sha256=best['checkpoint_sha256'], protocol_sha256=identity,
                test_used_for_selection=False, candidates=candidates)

        def restore_selected(selection):
            path = root / selection['checkpoint']
            if (not valid(path, expected_step=selection['best_step']) or
                    sha256(path / 'model.safetensors') != selection['checkpoint_sha256']):
                raise RuntimeError('TR validation-selected checkpoint is missing or corrupt')
            step, _ = load_step(path, self.actor, optimizer, expected_identity=identity)
            if step != selection['best_step']:
                raise RuntimeError('TR selected checkpoint step differs')
            self.training_receipt = dict(trained=True, control=control,
                train_digest=_digest(train), validation_digest=_digest(heldout),
                uses_test_for_training=False, checkpoint=str(path), selected_step=step,
                checkpoint_sha256=selection['checkpoint_sha256'], training_steps=max_steps)
            if validation_schedule == 'initial':
                self.training_receipt.update(training_only=True, selection_performed=False,
                    selection_policy=protocol['selection_policy'])
            return self.training_receipt

        if (root / 'complete.json').is_file():
            selection = selection_record()
            terminal = json.loads((root / 'complete.json').read_text())
            if (json.loads((root / 'selection.json').read_text()) != selection or
                    terminal != dict(complete=True, steps=max_steps, checkpoint=selection['checkpoint'],
                                     selected_step=selection['best_step'], selection_sha256=sha256(root / 'selection.json'))):
                raise RuntimeError('TR completed selection differs from validation history')
            return restore_selected(selection)

        previous = latest(root)
        if previous is None:
            if any((root / 'checkpoints').glob('*')) or (0 in save_steps and (root / 'metrics.jsonl').exists()):
                raise RuntimeError('Unsigned TR checkpoint exists; refuse to fabricate a resume step')
            torch.manual_seed(42)
            random.seed(42)
            if 0 in save_steps:
                save_step(root, 0, self.actor, optimizer, None, identity=identity, protected={0}, previous_step=-1)
            else:
                from training_protocol import restart_before_first_checkpoint
                root_call(lambda: restart_before_first_checkpoint(root))
            start = 0
        else:
            start, record = load_step(previous, self.actor, optimizer, expected_identity=identity)
            reconcile_history(root, start, record)
        if start > max_steps:
            raise RuntimeError('TR checkpoint exceeds new training budget')
        last_saved = start if previous is not None or 0 in save_steps else -1

        def chunks(rows):
            ordered = sorted(rows, key=lambda r: len(r['prompt_token_ids']) + len(r['response_token_ids']), reverse=True)
            for offset in range(0, len(ordered), micro_batch):
                yield ordered[offset:offset + micro_batch]

        def nll(rows):
            lengths = [len(r['prompt_token_ids']) + len(r['response_token_ids']) - 1 for r in rows]
            device = self.actor.student.device
            pad = getattr(getattr(self.actor.student, 'config', None), 'pad_token_id', None)
            ids = torch.full((len(rows), max(lengths)), 0 if pad is None else pad, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            targets = torch.full_like(ids, -100)
            for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
                prompt, answer = row['prompt_token_ids'], row['response_token_ids']
                ids[index, :length] = torch.tensor(prompt + answer[:-1], device=device)
                attention[index, :length] = 1
                targets[index, len(prompt)-1:length] = torch.tensor(answer, device=device)
            logits = self.actor.student(input_ids=ids, attention_mask=attention, use_cache=False).logits
            targets = targets.to(logits.device)
            mask = targets != -100
            return F.cross_entropy(logits[mask].float(), targets[mask], reduction='sum')

        def validate(step):
            path = root / 'evaluation' / f'step-{step:04d}.json'
            checkpoint_hash = (sha256(root / f'checkpoints/step-{step:04d}/model.safetensors')
                               if step in save_steps else None)
            if path.is_file():
                saved = candidate(step)
                if saved['checkpoint_sha256'] != checkpoint_hash:
                    raise RuntimeError('TR validation was scored with a different checkpoint')
                return saved
            if metric == 'heldout_nll':
                with torch.inference_mode():
                    count = sum(len(row['response_token_ids']) for row in heldout)
                    local_nll = sum(float(nll(rows)) for i, rows in enumerate(chunks(heldout))
                                    if i % dist().world() == dist().rank())
                    value = sum_values((local_nll,), self.actor.student.device)[0] / count
                result = {'complete': True, 'questions': len(heldout), 'heldout_nll': value, 'backend': 'hf'}
            else:
                result = self.evaluate(heldout, task=task, backend=validation_backend,
                    samples=validation_samples, cap=response_cap,
                    output=root / 'evaluation' / f'step-{step:04d}-vllm', vllm_python=vllm_python)
            role = 'initial' if step == 0 and validation_schedule in ('epoch', 'initial') else 'validation'
            write_json(path, dict(result, step=step, evaluation_role=role,
                                 protocol_sha256=identity, checkpoint_sha256=checkpoint_hash))
            return candidate(step)

        if start in val_steps:
            validate(start)
        if 0 in val_steps:
            candidate(0)
        previous_candidates = [candidate(step) for step in selection_steps if step <= start]
        best = choose(previous_candidates) if previous_candidates else None
        for step in range(start + 1, max_steps + 1):
            epoch, batch = divmod(step - 1, steps_per_epoch)
            offset = batch * batch_size
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            chosen = [train[index] for index in order[offset:offset + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            if objective == 'sft':
                count = sum(len(row['response_token_ids']) for row in chosen)
                losses = []
                for i, rows in enumerate(chunks(chosen)):
                    if i % dist().world() != dist().rank():
                        continue
                    loss = nll(rows) / count
                    loss.backward()
                    losses.append(loss.detach())
                    del loss
                local_loss = sum(float(value) for value in losses)
                result = {'step': step, 'train_nll': sum_values((local_loss,), self.actor.student.device)[0],
                          'response_tokens': count}
                updated = True
            else:
                from qwen3.runner import policy_loss
                from qwen3.scoring import score as train_score
                all_rollout_rows = [dict(row, index=step * len(train) + i) for i, row in enumerate(chosen)]
                rollout_rows = dist().local_slice(all_rollout_rows)
                local_chosen = dist().local_slice(chosen)
                answers = []
                if rollout_rows:
                    with self._decoder(rank_output(root / 'rollouts' / f'step-{step:04d}'), response_cap,
                            max(len(row['prompt_token_ids']) for row in chosen), vllm_python, .25) as decoder:
                        answers = decoder.generate(rollout_rows, count=8, cap=response_cap,
                            sampling=protocol['rollout_sampling'], logprobs=True)
                expected = [(r['index'], sample) for r in rollout_rows for sample in range(8)]
                if [(a['index'], a['sample_index']) for a in answers] != expected:
                    raise RuntimeError('TR vLLM rollout response coverage differs')
                rewards_total, contributions = 0, 0
                for i, row in enumerate(local_chosen):
                    responses, rewards = [], []
                    for generated in answers[i*8:(i+1)*8]:
                        answer = generated['response_token_ids']
                        log_probs = generated.get('log_probs', [])
                        if (not answer or len(answer) > response_cap or len(log_probs) != len(answer) or
                                any(not math.isfinite(p) for p in log_probs)):
                            raise RuntimeError('TR vLLM rollout needs complete finite token log probabilities')
                        old = torch.tensor(log_probs, dtype=torch.float32, device=self.actor.student.device)
                        text = generated['response']
                        truncated = generated.get('finish_reason') == 'length' or len(answer) >= response_cap
                        reward = (reward_fn(row, text, truncated) if reward_fn else
                                  train_score(task, text, row['gold'], truncated=truncated))
                        if type(reward) not in (int, float) or reward not in (0, 1):
                            raise ValueError('TR GRPO reward must be an exact binary verdict')
                        responses.append((answer, old))
                        rewards.append(reward)
                    scores = torch.tensor(rewards, dtype=torch.float32)
                    advantages = (scores - scores.mean()) / (scores.std(correction=1) + 1e-6)
                    rewards_total += sum(rewards)
                    for (answer, old), advantage in zip(responses, advantages, strict=True):
                        if not float(advantage):
                            continue
                        current = self.actor.response_log_probs(row['prompt_token_ids'], answer)
                        (policy_loss(current, old, float(advantage)) / (len(chosen) * 8)).backward()
                        contributions += 1
                rewards_total, contributions = sum_values((rewards_total, contributions), self.actor.student.device)
                result = {'step': step, 'train/responses': len(chosen) * 8,
                          'train/correct': rewards_total, 'train/mixed_responses': contributions,
                          'rollout_backend': 'vllm'}
                updated = bool(contributions)
            set_sft_learning_rate(optimizer, protocol, step - 1)
            result['learning_rate'] = optimizer.param_groups[0]['lr']
            result.update(epoch=epoch + 1, train_indices=[row['index'] for row in chosen])
            dist().all_reduce_gradients(tuple(self.actor.trainable()))
            norm = torch.nn.utils.clip_grad_norm_(self.actor.trainable(), 1., error_if_nonfinite=True)
            if updated:
                optimizer.step()
            set_sft_learning_rate(optimizer, protocol, step)
            result['grad_norm'] = float(norm)
            if step in save_steps:
                save_step(root, step, self.actor, optimizer, result, identity=identity,
                          protected={0, best['step'] if best else 0, max_steps}, previous_step=last_saved)
                last_saved = step
            append_metric(root, result)
            if step in val_steps:
                best = choose(([best] if best else []) + [validate(step)])
        selection = selection_record()
        receipt = restore_selected(selection)
        write_json(root / 'selection.json', selection)
        write_json(root / 'complete.json', dict(complete=True, steps=max_steps,
            checkpoint=selection['checkpoint'], selected_step=selection['best_step'],
            selection_sha256=sha256(root / 'selection.json')))
        return receipt


class Qwen35Session:
    """Independent 4B text receiver: bridge, virtual sender, full or LoRA."""

    def __init__(self, actor, tokenizer, receiver, *, identity, weights=None,
                 code_cases=None, official_checker=None, docker_image='lcb-checker:local',
                 allow_rebuilt_image=False):
        self.actor, self.tokenizer = actor, tokenizer
        self.receiver = Path(receiver)
        self.identity = identity
        self.weights = Path(weights) if weights is not None else None
        self._current = None
        self.code_cases, self.official_checker = code_cases, official_checker
        self.docker_image = docker_image
        self.allow_rebuilt_image = allow_rebuilt_image
        self.case = actor.experiment

    @classmethod
    def load(cls, sender=None, bridge=None, receiver=None, *, task='dapo', method='bridge',
             template=None, device='cuda:0', code_cases=None, official_checker=None,
             docker_image='lcb-checker:local', allow_rebuilt_image=False,
             receiver_shape=(32, 2560)):
        """``receiver_shape`` is (layers, width); the default requires Qwen3.5-4B.

        Pass another shape only for deliberately different (e.g. tiny CPU-test) models.
        """
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        from qwen35.checkpoints import sha256
        from qwen35.catalog import MAP_35B, MAP_397B, MAP_4B
        from qwen35.model import Actor, load_text
        if receiver is None or method not in ('bridge', 'random', 'full', 'lora') or task not in ('dapo', 'deepcoder'):
            raise ValueError('Select a Qwen3.5 4B receiver, method and task')
        if (method == 'bridge') != (sender is not None):
            raise ValueError('Only the real-sender bridge method takes a sender')
        if method == 'random' and template is None and bridge is None:
            raise ValueError('Virtual sender needs TRAIN-derived statistics or signed inference weights')
        device = local_device(device)
        student = load_text(receiver, device=device)
        placement = 'auto' if device != 'cpu' and dist().world() == 1 else None
        if sender and 'PLATOUR_GPU_GROUP' in os.environ:
            from qwen3.eval_parallel import sender_device_map
            configuration = json.loads((Path(sender) / 'config.json').read_text())
            text_config = configuration.get('text_config', configuration)
            family = 'qwen3' if configuration.get('model_type') == 'qwen3_5_text' else 'qwen35'
            placement = sender_device_map(family, text_config['num_hidden_layers'], torch.cuda.device_count())
        teacher = load_text(sender, device=device, device_map=placement) if sender else None
        maps = {32: MAP_4B, 40: MAP_35B, 60: MAP_397B}
        if teacher is not None and teacher.config.num_hidden_layers not in maps:
            raise ValueError('No audited layer mapping for this Qwen3.5 sender')
        layer_map = maps[teacher.config.num_hidden_layers] if teacher is not None else MAP_4B
        if method == 'random' and template is None:
            saved = load_file(bridge, device='cpu')
            template = {key: saved[key] for key in ('virtual_hidden', 'template_token_ids')}
        receiver_shape = tuple(receiver_shape)
        case = SimpleNamespace(task=task, method=method, layer_map=layer_map,
                               teacher_dim=(teacher.config.hidden_size if teacher is not None
                                            else receiver_shape[1]),
                               receiver_shape=receiver_shape)
        if method == 'lora' and receiver_shape != (32, 2560):
            from torch import nn
            case.lora_target_count = sum(isinstance(module, nn.Linear) and
                                         name.startswith('model.layers.')
                                         for name, module in student.named_modules())
        actor = Actor(case, student, teacher=teacher, template=template)
        if bridge is not None:
            actor.load_inference_state(load_file(bridge, device='cpu'))
        tokenizer = AutoTokenizer.from_pretrained(str(receiver), local_files_only=True)
        identity = json.dumps({'sender': str(Path(sender).resolve()) if sender else None,
                               'receiver': str(Path(receiver).resolve()), 'method': method,
                               'task': task, 'layer_map': layer_map,
                               'receiver_shape': list(receiver_shape),
                               'checkpoint_sha256': sha256(bridge) if bridge is not None else None},
                              sort_keys=True)
        return cls(actor, tokenizer, receiver, identity=identity, weights=bridge,
                   code_cases=code_cases, official_checker=official_checker,
                   docker_image=docker_image, allow_rebuilt_image=allow_rebuilt_image)

    def _verify_judge(self, rows):
        if self.case.task != 'deepcoder':
            return
        import os
        import re
        import subprocess
        from qwen35.checkpoints import sha256
        from qwen35.judge_cases import PINNED
        from qwen35.judge_service import ORIGINAL_IMAGE
        from qwen35.scoring import SAFE_KEY, judge_service_info
        cases, checker = Path(self.code_cases or ''), Path(self.official_checker or '')
        pinned = json.loads(PINNED.read_text())
        extractor = checker.parent / 'utils/extraction_utils.py'
        sealed_file = cases.parent / 'complete.json'
        if (not self.code_cases or not self.official_checker or not self.docker_image or
                cases.is_symlink() or checker.is_symlink() or not cases.is_dir() or
                (checker / 'testing_util.py').is_symlink() or
                not (checker / 'testing_util.py').is_file() or
                extractor.is_symlink() or not extractor.is_file() or
                not sealed_file.is_file() or sealed_file.is_symlink() or
                sha256(checker / 'testing_util.py') != pinned['checker_sha256'] or
                sha256(extractor) != pinned['extractor_sha256']):
            raise RuntimeError('Authorized original code cases/checker and image required')
        from lcb_runner.utils import extraction_utils as official
        if Path(official.__file__).resolve() != extractor.resolve():
            raise RuntimeError('Code extractor import is not the pinned LiveCodeBench checkout')
        sealed = json.loads(sealed_file.read_text())
        if (not sealed.get('complete') or not sealed.get('per_question_sha256_verified') or
                sealed.get('training_validation_cases') != 19074 or
                sealed.get('lcb_v6_cases') != 175 or
                sealed.get('checker_sha256') != pinned['checker_sha256'] or
                sealed.get('extractor_sha256') != pinned['extractor_sha256']):
            raise RuntimeError('Code test case seal differs from original approved bytes')
        service = os.environ.get('PLATOUR_CODE_JUDGE_SERVICE')
        if service:
            info = judge_service_info(service)
            experiment = os.environ.get('PLATOUR_CODE_JUDGE_EXPERIMENT')
            if (info.get('case_root') != str(cases.resolve()) or
                    info.get('checker_root') != str(checker.resolve()) or
                    info.get('image') != self.docker_image or
                    info.get('image_id') != ORIGINAL_IMAGE or
                    info.get('extractor_sha256') != pinned['extractor_sha256'] or
                    info.get('case_proof_sha256') != sha256(sealed_file) or
                    experiment not in info.get('supported_experiments', ())):
                raise RuntimeError('Isolated judge service differs from approved sources')
            image_id = info['image_id']
        else:
            result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                     self.docker_image], capture_output=True, text=True, timeout=20)
            image_id = result.stdout.strip() if result.returncode == 0 else ''
            if not re.fullmatch(r'sha256:[a-f0-9]{64}', image_id):
                raise RuntimeError('Isolated Docker judge daemon or image unavailable')
            if image_id != ORIGINAL_IMAGE and self.allow_rebuilt_image is not True:
                raise RuntimeError('Rebuilt judge image needs explicit allow_rebuilt_image=True')
        for row in rows:
            spec = row.get('judge')
            if not isinstance(spec, dict):
                raise ValueError('Code training/evaluation requires the signed judge contract')
            kind = spec.get('kind')
            if kind in ('deepcoder', 'lcb'):
                key = spec.get('key')
                if not isinstance(key, str) or not SAFE_KEY.fullmatch(key):
                    raise ValueError('Unsafe original code judge case key')
                file = cases / (key + '.json')
                expected = (spec.get('sample_sha256') if kind == 'deepcoder' else
                            pinned['case_sha256'].get(file.name))
                if (file.is_symlink() or not file.is_file() or
                        not file.resolve().is_relative_to(cases.resolve()) or
                        expected is None or sha256(file) != expected):
                    raise RuntimeError('Original programming question case SHA differs')
            elif kind not in ('humaneval', 'mbpp'):
                raise ValueError('Unsupported original code task')
        return {'image_id': image_id, 'original_image': image_id == ORIGINAL_IMAGE,
                'checker_sha256': pinned['checker_sha256'],
                'extractor_sha256': pinned['extractor_sha256'],
                'case_proof_sha256': sha256(sealed_file),
                'service_spec_id': info['spec_id'] if service else None}

    def _weights(self, output):
        if self._current is not None:
            return self._current
        if self.weights is not None:
            return self.weights
        destination = Path(output) / ('inference-' + str(time.time_ns()) + '.safetensors')
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous().clone()
                   for key, value in self.actor.inference_state().items()}, destination)
        return destination

    def _decoder(self, output, cap, vllm_python, prompt_length):
        from platour_exps import require_vllm_python
        from qwen35.vllm_decode import decoder_session
        if output is None:
            raise ValueError('Qwen3.5 vLLM needs a private output directory')
        require_vllm_python(vllm_python)
        return decoder_session(self.case, self.receiver, self._weights(output), output,
            actor=self.actor if self.case.method in ('bridge', 'random') else None,
            weight_source='checkpoint' if self._current is not None else 'inference',
            worker_python=vllm_python, max_model_len=max(3072, cap + prompt_length))

    def generate(self, *, prompt_ids=None, messages=None, backend='vllm', seed=42,
                 max_new_tokens=256, output=None, vllm_python=None):
        prompt = PLatourSession.prompt_ids(self, prompt_ids=prompt_ids, messages=messages)
        if max_new_tokens < 1:
            raise ValueError('Qwen3.5 answer cap must be positive')
        if backend == 'hf':
            raise ValueError('Answer generation requires vLLM')
        elif backend == 'vllm':
            with self._decoder(output, max_new_tokens, vllm_python, len(prompt)) as decoder:
                row = decoder.generate([{'index': 0, 'prompt_token_ids': prompt}],
                                       count=1, cap=max_new_tokens, seed_override=seed)[0]
            ids, text = row['response_token_ids'], row['response']
        else:
            raise ValueError('Qwen3.5 backend must be vllm')
        return {'response': text, 'response_token_ids': ids, 'backend': backend}

    @parallel_evaluate
    def evaluate(self, data, *, backend='vllm', samples=1, cap=256,
                 output=None, vllm_python=None, feature_provider=None):
        from qwen35.cli import evaluate, evaluate_vllm
        rows = load_dataset(data)
        judge_environment = self._verify_judge(rows)
        if samples < 1 or cap < 1:
            raise ValueError('Positive evaluation count and cap required')
        options = dict(count=samples, cap=cap, code_cases=self.code_cases,
                       official_checker=self.official_checker, docker_image=self.docker_image)
        scope = (self.actor.use_features(feature_provider) if feature_provider is not None
                 else nullcontext())
        with scope:
            if backend == 'hf':
                raise ValueError('Answer generation requires vLLM')
            elif backend == 'vllm':
                with self._decoder(output, cap, vllm_python,
                                   max(len(row['prompt_token_ids']) for row in rows)) as decoder:
                    result = evaluate_vllm(decoder, self.case, rows, **options)
            else:
                raise ValueError('Qwen3.5 backend must be vllm')
        if judge_environment is not None:
            result['judge_environment'] = judge_environment
        return result

    def _selection(self, root, protocol, through=None):
        import math
        from qwen35.checkpoints import read, sha256
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        candidates = []
        sft = protocol['objective'] == 'sft'
        metric = 'heldout_nll' if sft else 'accuracy'
        limit = protocol['max_steps'] if through is None else through
        for step in protocol.get('validation_steps', range(protocol['max_steps'] + 1)):
            if step > limit:
                continue
            path = root / 'evaluation' / f'step-{step:04d}.json'
            value = read(path)
            accuracy = value.get(metric)
            if (not value.get('complete') or value.get('step') != step or
                    value.get('evaluation_role') != 'validation' or
                    value.get('protocol_sha256') != identity or value.get('backend') != ('hf' if sft else 'vllm') or
                    value.get('questions') != protocol['validation_questions'] or
                    type(accuracy) not in (int, float) or not math.isfinite(accuracy) or (not sft and (not 0 <= accuracy <= 1 or
                    value.get('responses') != protocol['validation_questions'] * protocol['validation_samples'] or
                    type(value.get('correct')) is not int or
                    not math.isclose(value['correct'], accuracy * value['responses'])))):
                raise RuntimeError('Qwen3.5 VAL history or response denominator differs')
            candidates.append(dict(step=step, score=accuracy, validation_sha256=sha256(path),
                                   checkpoint_sha256=value['checkpoint_sha256']))
        best = (max(candidates, key=lambda c: c['step']) if protocol['objective'] == 'sft'
                else min(candidates, key=lambda c: (-c['score'], c['step'])))
        return dict(policy=protocol['selection_policy'], best_step=best['step'],
            best_score=best['score'], training_steps=protocol['max_steps'],
            checkpoint=f"checkpoints/step-{best['step']:04d}",
            checkpoint_sha256=best['checkpoint_sha256'], protocol_sha256=identity,
            test_used_for_selection=False, candidates=candidates)

    def restore_completed(self, output, *, control=None):
        """Verify validation evidence and restore the final SFT or validation-best GRPO model."""
        from qwen35.checkpoints import read, sha256
        restore = load_actor
        root = Path(output).absolute()
        protocol, terminal = read(root / 'protocol.json'), read(root / 'complete.json')
        if (protocol.get('model_identity') != self.identity or
                protocol.get('method') != self.case.method or protocol.get('control') != control or
                protocol.get('selection_policy') != (
                    'final_sft_checkpoint_v1' if protocol.get('objective') == 'sft' else 'validation_best_earliest_tie_v1')):
            raise RuntimeError('Completed Qwen3.5 run belongs to another protocol')
        selection = self._selection(root, protocol)
        if (read(root / 'selection.json') != selection or
                terminal != dict(complete=True, steps=protocol['max_steps'],
                    selected_step=selection['best_step'], checkpoint=selection['checkpoint'],
                    selection_sha256=sha256(root / 'selection.json'))):
            raise RuntimeError('Qwen3.5 completed selection differs from VAL history')
        path = root / selection['checkpoint']
        if sha256(path / 'model.safetensors') != selection['checkpoint_sha256']:
            raise RuntimeError('Qwen3.5 selected checkpoint differs from validation weights')
        optimizer = torch.optim.AdamW(self.actor.trainable(), lr=protocol['learning_rate'])
        step, _ = restore(path, self.actor, optimizer, expected_identity=selection['protocol_sha256'])
        if step != selection['best_step']:
            raise RuntimeError('Qwen3.5 selected checkpoint step differs')
        self._current = path / 'model.safetensors'
        return dict(trained=True, control=control, train_digest=protocol['train_sha256'],
            validation_digest=protocol['validation_sha256'], uses_test_for_training=False,
            objective=protocol['objective'], checkpoint=str(path), selected_step=step,
            training_steps=protocol['max_steps'], checkpoint_sha256=selection['checkpoint_sha256'])

    def fit(self, train_data, validation_data, *, output, objective='grpo',
            max_steps=None, epochs=1, smoke=False, batch_size=64, learning_rate=None,
            validation_backend=None, validation_samples=None, response_cap=16384,
            vllm_python=None, control=None, warmup_steps=None, validation_schedule='auto',
            rollout_worker_mode=None):
        """Fit independent TRAIN/VAL. Frozen-receiver GRPO reuses a sleeping worker.

        Set rollout_worker_mode='restart' for the previous per-use lifecycle or
        legacy resume. Full/LoRA receiver training retains per-use workers.
        Custom SFT uses HF held-out NLL and a warmup/linear-decay schedule;
        GRPO uses constant LR and validation accuracy to select its checkpoint.
        """
        eligible = objective == 'grpo' and self.case.method in ('bridge', 'random')
        mode = rollout_worker_mode or ('persistent' if eligible else 'restart')
        if mode not in ('persistent', 'restart') or (mode == 'persistent' and not eligible):
            raise ValueError('Persistent rollout workers require frozen-receiver GRPO')
        scope = nullcontext()
        if eligible:
            from qwen35.vllm_decode import training_decoder_scope
            train_data, validation_data = load_dataset(train_data), load_dataset(validation_data)
            scope = training_decoder_scope(rank_output(Path(output) / 'rollout-worker'),
                max_model_len=max(3072, response_cap + max(len(row['prompt_token_ids'])
                    for row in [*train_data, *validation_data])), mode=mode)
        with scope:
            return self._fit(train_data, validation_data, output=output, objective=objective,
                max_steps=max_steps, epochs=epochs, smoke=smoke, batch_size=batch_size,
                learning_rate=learning_rate, validation_backend=validation_backend,
                validation_samples=validation_samples, response_cap=response_cap,
                vllm_python=vllm_python, control=control, warmup_steps=warmup_steps,
                validation_schedule=validation_schedule, rollout_worker_mode=mode)

    def _fit(self, train_data, validation_data, *, output, objective, max_steps, epochs,
             smoke, batch_size, learning_rate, validation_backend, validation_samples,
             response_cap, vllm_python, control, warmup_steps, validation_schedule,
             rollout_worker_mode):
        from qwen35.cli import train_step
        from training_protocol import budget
        from qwen35.checkpoints import sha256, rng_state, restore_rng
        from qwen35.checkpoints import latest
        write_json, restore, commit = api_write_json, load_actor, save_actor
        from training_protocol import training_learning_rate, sft_schedule, set_sft_learning_rate
        learning_rate = training_learning_rate(objective, self.case.method, learning_rate)
        from training_protocol import validation_backend as check_backend
        validation_backend = check_backend(validation_backend, objective, self.case.task)
        if validation_samples is None:
            validation_samples = 4 if self.case.task == 'dapo' else 16
        train, heldout = load_dataset(train_data), load_dataset(validation_data)
        steps_per_epoch, max_steps = budget(len(train), batch_size, epochs, 1, max_steps, smoke)
        from training_protocol import validation_steps, default_validation_schedule
        if validation_schedule == 'auto':
            validation_schedule = default_validation_schedule(objective, 'qwen35')
        val_steps = validation_steps(validation_schedule, objective, validation_backend, self.case.task,
            steps_per_epoch, max_steps, smoke=smoke, family='qwen35', case=self.case)
        if (objective not in ('sft', 'grpo') or
                max_steps < 1 or batch_size < 1 or learning_rate <= 0 or response_cap < 1 or
                validation_samples < 1 or not {tuple(r['prompt_token_ids']) for r in train}.isdisjoint(
                    {tuple(r['prompt_token_ids']) for r in heldout})):
            raise ValueError('Qwen3.5 independent TRAIN/VAL or schedule differs')
        if objective == 'sft' and any(not row.get('response_token_ids') for row in train + heldout):
            raise ValueError('Custom Qwen3.5 SFT needs TRAIN/VAL supervised answer IDs')
        judge_environment = self._verify_judge(train + heldout)
        config = {'model_identity': self.identity, 'method': self.case.method,
                  'task': self.case.task, 'objective': objective, 'train_sha256': _digest(train),
                  'validation_sha256': _digest(heldout), 'max_steps': max_steps,
                  'batch_size': batch_size, 'learning_rate': learning_rate,
                  'selection_metric': 'heldout_nll' if objective == 'sft' else 'accuracy',
                  'validation_backend': validation_backend, 'validation_samples': validation_samples,
                  'response_cap': response_cap, 'judge_environment': judge_environment,
                  'epochs': epochs, 'steps_per_epoch': steps_per_epoch, 'smoke_only': smoke,
                  'selection_policy': ('final_sft_checkpoint_v1' if objective == 'sft' else 'validation_best_earliest_tie_v1'), 'test_used_for_selection': False,
                  'control': control, 'validation_questions': len(heldout),
                  'rollout_backend': 'vllm' if objective == 'grpo' else None,
                  'rollout_sampling': dict(temperature=1., top_p=1., top_k=-1, presence_penalty=0.)}
        config.update(sft_schedule(objective, max_steps, warmup_steps))
        config.update(validation_schedule=validation_schedule, validation_steps=val_steps)
        sign_topology(config)
        root = Path(output).absolute()
        if objective == 'grpo' and self.case.method in ('bridge', 'random'):
            previous = root / 'protocol.json'
            legacy_restart = (rollout_worker_mode == 'restart' and previous.is_file()
                              and 'rollout_worker_mode' not in json.loads(previous.read_text()))
            if not legacy_restart:
                config['rollout_worker_mode'] = rollout_worker_mode
        protocol_root(root, config)
        sync_parameters(self.actor.trainable())
        if (root / 'complete.json').is_file():
            return self.restore_completed(root, control=control)
        identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        optimizer = torch.optim.AdamW(self.actor.trainable(), lr=learning_rate,
                                      betas=(.9, .999), weight_decay=0.)
        set_sft_learning_rate(optimizer, config, 0)
        previous = latest(root)
        if previous is None:
            if (root / 'metrics.jsonl').is_file() or any((root / 'checkpoints').glob('*')):
                raise RuntimeError('Unsigned Qwen3.5 optimizer cannot be resumed')
            torch.manual_seed(42)
            random.seed(42)
            commit(root, 0, self.actor, optimizer, None, protected={0}, identity=identity)
            start = 0
        else:
            start, record = restore(previous, self.actor, optimizer, expected_identity=identity)
            reconcile_history(root, start, record)
        if start > max_steps:
            raise RuntimeError('Qwen3.5 checkpoint exceeds epoch budget')
        self._current = latest(root) / 'model.safetensors'

        def validate(step):
            destination = root / 'evaluation' / f'step-{step:04d}.json'
            weight_hash = sha256(root / 'checkpoints' / f'step-{step:04d}' / 'model.safetensors')
            if destination.is_file():
                value = json.loads(destination.read_text())
                if (not value.get('complete') or value.get('protocol_sha256') != identity or
                        value.get('checkpoint_sha256') != weight_hash):
                    raise RuntimeError('Committed Qwen3.5 validation differs from checkpoint/protocol')
                return
            saved_rng = rng_state()
            try:
                if objective == 'sft':
                    with torch.inference_mode():
                        tokens = sum(len(row['response_token_ids']) for row in heldout)
                        local_nll = sum(float(-self.actor.response_log_probs(row['prompt_token_ids'],
                            row['response_token_ids']).sum()) for row in dist().local_slice(heldout))
                        nll = sum_values((local_nll,), self.actor.student.device)[0] / tokens
                    report = dict(complete=True, questions=len(heldout), response_tokens=tokens,
                                  heldout_nll=nll, backend='hf')
                else:
                    report = self.evaluate(heldout, backend=validation_backend,
                        samples=validation_samples, cap=response_cap,
                        output=root / 'evaluation' / f'step-{step:04d}-vllm',
                        vllm_python=vllm_python)
            finally:
                restore_rng(saved_rng)
            write_json(destination, {'step': step, 'evaluation_role': 'validation', **report,
                'protocol_sha256': identity, 'checkpoint_sha256': weight_hash})

        if start in val_steps:
            validate(start)
        def choose_through(step):
            return self._selection(root, config, through=step)['best_step']
        metric = config['selection_metric']
        if any(step <= start for step in val_steps):
            best = choose_through(start)
            best_score = json.loads((root / 'evaluation' / f'step-{best:04d}.json').read_text())[metric]
        else:
            # No selection candidate exists before final-only smoke validation.
            best, best_score = 0, float('-inf')
        for step in range(start + 1, max_steps + 1):
            epoch, batch_index = divmod(step - 1, steps_per_epoch)
            offset = batch_index * batch_size
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            batch = [train[index] for index in order[offset:offset + batch_size]]
            if not batch:
                raise RuntimeError('Empty independent Qwen3.5 minibatch')
            if objective == 'grpo':
                metrics = train_step(self.actor, optimizer, self.tokenizer, self.case, batch, step,
                    code_cases=self.code_cases, official_checker=self.official_checker,
                    docker_image=self.docker_image, cap=response_cap, receiver=self.receiver,
                    output=rank_output(root / 'rollouts' / f'step-{step:04d}'), vllm_python=vllm_python)
            else:
                count = sum(len(row['response_token_ids']) for row in batch)
                optimizer.zero_grad(set_to_none=True)
                local_loss = 0.
                for row in dist().local_slice(batch):
                    loss = -self.actor.response_log_probs(row['prompt_token_ids'], row['response_token_ids']).sum() / count
                    loss.backward()
                    local_loss += float(loss.detach())
                dist().all_reduce_gradients(tuple(self.actor.trainable()))
                norm = torch.nn.utils.clip_grad_norm_(self.actor.trainable(), 1., error_if_nonfinite=True)
                set_sft_learning_rate(optimizer, config, step - 1)
                applied_lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                set_sft_learning_rate(optimizer, config, step)
                metrics = {'train/learning_rate': applied_lr,
                           'train/nll': sum_values((local_loss,), self.actor.student.device)[0], 'train/grad_norm': float(norm)}
            record = {'step': step, 'epoch': epoch + 1, 'train_indices': order[offset:offset + batch_size], **metrics}
            commit(root, step, self.actor, optimizer, record, protected={0, best, max_steps, *val_steps}, identity=identity)
            self._current = latest(root) / 'model.safetensors'
            append_metric(root, record)
            if step in val_steps:
                validate(step)
                value = json.loads((root / 'evaluation' / f'step-{step:04d}.json').read_text())
                if objective == 'sft' or value[metric] > best_score:
                    best, best_score = step, value[metric]
        selection = self._selection(root, config)
        write_json(root / 'selection.json', selection)
        write_json(root / 'complete.json', dict(complete=True, steps=max_steps,
            selected_step=selection['best_step'], checkpoint=selection['checkpoint'],
            selection_sha256=sha256(root / 'selection.json')))
        return self.restore_completed(root, control=control)
