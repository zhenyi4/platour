"""Text-only Qwen3.5 reference actors for the four published training methods."""

from contextlib import contextmanager, nullcontext
import inspect
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open


class LoRALinear(nn.Module):
    def __init__(self, base, rank=128, alpha=128):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('Qwen3.5 LoRA target is not an nn.Linear')
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return self.base(x) + delta.to(x.dtype) * (self.alpha / self.rank)


def verify_standard_weight(experiment, path):
    """Bind a generically named standard artifact to its pinned experiment."""
    from platour_exps import verify_weight, weight_record
    path = Path(path)
    expected = {'full': 'model.safetensors', 'lora': 'adapter_model.safetensors'}.get(experiment.method)
    _, filename, size, digest = weight_record(experiment.identifier)
    if expected is None or path.name != expected or filename != expected:
        raise ValueError('Standard weight format does not match the chosen experiment')
    verify_weight(path, size, digest)


class Actor(nn.Module):
    def __init__(self, experiment, student, teacher=None, *, template=None):
        super().__init__()
        self.experiment = experiment
        self.student = student
        self.teacher = teacher
        if experiment.method in ('bridge', 'random'):
            if experiment.method == 'bridge' and teacher is None:
                raise ValueError('A frozen Qwen3.5 sender is required')
            # Published runs require the Qwen3.5-4B text receiver; a caller may declare
            # another (layers, width) explicitly, e.g. tiny CPU-simulation models.
            depth, width = tuple(getattr(experiment, 'receiver_shape', (32, 2560)))
            if student.config.num_hidden_layers != depth or student.config.hidden_size != width:
                raise ValueError('Receiver is not the expected Qwen3.5 text model (default 4B)')
            if len(experiment.layer_map) != depth:
                raise ValueError('Sender layer map must cover every receiver layer')
            self.projections = nn.ModuleList([
                nn.Linear(experiment.teacher_dim, width, bias=False,
                          device=student.device, dtype=torch.float32)
                for _ in range(depth)])
            for layer in self.projections:
                nn.init.zeros_(layer.weight)
            if experiment.method == 'random':
                if template is None:
                    raise ValueError('Random-template actor needs independently derived train-only statistics')
                virtual = template['virtual_hidden']
                tokens = template['template_token_ids']
                if virtual.shape != (63 if experiment.task == 'dapo' else 67, depth, width):
                    raise ValueError('Wrong method/dataset virtual teacher statistics')
                if tokens.shape != virtual.shape[:1]:
                    raise ValueError('Wrong public template token positions')
                self.virtual_hidden = nn.Parameter(virtual.to(student.device, dtype=torch.float32).clone())
                self.register_buffer('template_token_ids', tokens.to(student.device).clone())
                self.prefix_count = 42 if experiment.task == 'dapo' else 45
                self.suffix_count = 22 if experiment.task == 'dapo' else 23
        elif experiment.method == 'lora':
            self.adapters = {}
            targets = [(name, module) for name, module in student.named_modules()
                       if isinstance(module, nn.Linear) and name.startswith('model.layers.')]
            expected = getattr(experiment, 'lora_target_count', 248)
            if not isinstance(expected, int) or expected < 1 or len(targets) != expected:
                raise ValueError(f'Expected {expected} decoder linear LoRA targets, found {len(targets)}')
            for name, module in targets:
                parent, attr = name.rsplit('.', 1)
                adapted = LoRALinear(module, rank=getattr(experiment, 'lora_rank', 128),
                                     alpha=getattr(experiment, 'lora_alpha', 128))
                setattr(student.get_submodule(parent), attr, adapted)
                self.adapters[name] = adapted
        elif experiment.method != 'full':
            raise ValueError('Unknown Qwen3.5 method')
        if self.teacher is not None:
            self.teacher.eval().requires_grad_(False)
        self.student.eval()
        if experiment.method != 'full':
            self.student.requires_grad_(False)
            if experiment.method == 'lora':
                for adapter in self.adapters.values():
                    adapter.lora_A.requires_grad_(True)
                    adapter.lora_B.requires_grad_(True)
        else:
            self.student.requires_grad_(True)

    def trainable(self):
        if self.experiment.method == 'bridge':
            return tuple(self.projections.parameters())
        if self.experiment.method == 'random':
            return (*tuple(self.projections.parameters()), self.virtual_hidden)
        if self.experiment.method == 'lora':
            return tuple(parameter for adapter in self.adapters.values()
                         for parameter in (adapter.lora_A, adapter.lora_B))
        return tuple(self.student.parameters())

    def inference_state(self):
        kind = self.experiment.method
        if kind == 'full':
            return self.student.state_dict()
        if kind == 'lora':
            return {name + '.' + suffix: getattr(adapter, suffix)
                    for name, adapter in self.adapters.items()
                    for suffix in ('lora_A', 'lora_B')}
        state = {'P.' + name: param for name, param in self.projections.state_dict().items()}
        if kind == 'random':
            state.update(virtual_hidden=self.virtual_hidden,
                         template_token_ids=self.template_token_ids)
        return state

    def load_inference_state(self, state):
        expected = self.inference_state()
        if set(expected) != set(state):
            raise ValueError('Published weights do not match the method/architecture')
        with torch.no_grad():
            for name, value in expected.items():
                if value.shape != state[name].shape or value.dtype != state[name].dtype:
                    raise ValueError('Published tensor shape/dtype mismatch: ' + name)
                value.copy_(state[name].to(value.device))

    def load_public(self, path):
        path = Path(path)
        standard = path.name in ('model.safetensors', 'adapter_model.safetensors')
        if standard:
            verify_standard_weight(self.experiment, path)
        elif path.name != self.experiment.tag + '.safetensors':
            raise ValueError('Published weight filename does not match the chosen experiment')
        with safe_open(path, framework='pt', device='cpu') as source:
            metadata = source.metadata() or {}
            if standard and self.experiment.method == 'lora':
                if metadata != {'format': 'pt'}:
                    raise ValueError('Unexpected metadata in the standard PEFT release')
                from .hf_export import custom_lora_name
                state = {custom_lora_name(key): source.get_tensor(key) for key in source.keys()}
                if len(state) != len(source.keys()):
                    raise ValueError('Duplicate PEFT tensor targets')
                self.load_inference_state(state)
                return
            if self.experiment.method != 'full' and (
                    metadata.get('task') != self.experiment.task or
                    metadata.get('method') != self.experiment.method or
                    'selected_step' in metadata or 'selected_checkpoint_step' in metadata):
                raise ValueError('Published weight task/method or private selection metadata differs')
            if self.experiment.method == 'full' and metadata:
                raise ValueError('Full-model release must not contain private metadata')
            if self.experiment.method == 'full':
                embeddings = self.student.get_input_embeddings().weight
                head = self.student.get_output_embeddings()
                if head.weight.data_ptr() == embeddings.data_ptr():
                    head.weight = nn.Parameter(head.weight.detach().clone())
                self.student.config.tie_word_embeddings = False
            self.load_inference_state({key: source.get_tensor(key) for key in source.keys()})

    @contextmanager
    def use_features(self, provider):
        """Scope a PACT-Check sender intervention (DT/IT) to HF and vLLM prefill."""
        if self.experiment.method != 'bridge' or not callable(provider):
            raise ValueError('Sender interventions apply to a real-sender bridge only')
        previous = getattr(self, '_feature_provider', None)
        self._feature_provider = provider
        try:
            yield
        finally:
            self._feature_provider = previous

    def sender_features(self, prompt):
        """Real block-input features of the frozen sender at the audited layer map."""
        inputs = torch.tensor([prompt[:-1]], device=self.teacher.device)
        with torch.no_grad():
            raw = self.teacher.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     use_cache=False, output_hidden_states=True).hidden_states
        return tuple(raw[i] for i in self.experiment.layer_map)

    def _features(self, prompt):
        if len(prompt) < 2:
            raise ValueError('Prompt must contain the final native receiver token')
        n = len(prompt) - 1
        device = self.student.device
        provider = getattr(self, '_feature_provider', None)
        if provider is not None:
            features = [torch.as_tensor(value).to(device) for value in provider(prompt)]
            if len(features) != len(self.projections) or any(
                    value.shape != (1, n, self.experiment.teacher_dim) or
                                          not torch.isfinite(value).all() for value in features):
                raise ValueError('Supplied sender features have the wrong layer/position shape')
            return features, torch.ones(n, device=device, dtype=torch.bool)
        if self.experiment.method == 'random':
            source = self.template_token_ids.tolist()
            template_positions = list(range(self.prefix_count)) + list(
                range(len(prompt) - self.suffix_count, len(prompt) - 1))
            if (len(prompt) - self.suffix_count < self.prefix_count or
                    len(template_positions) != len(source) or
                    [prompt[i] for i in template_positions] != source):
                raise ValueError('The virtual sender needs prompts rendered with the task\'s fixed '
                                 'template (its TRAIN-derived template tokens were not found)')
            layers, width = self.virtual_hidden.shape[1:]
            features = [torch.zeros((1, n, width), device=device, dtype=torch.float32)
                        for _ in range(layers)]
            for slot, position in enumerate(template_positions):
                for depth in range(layers):
                    features[depth][0, position] = self.virtual_hidden[slot, depth]
            mask = torch.zeros(n, device=device, dtype=torch.bool)
            mask[template_positions] = True
            return features, mask
        inputs = torch.tensor([prompt[:-1]], device=self.teacher.device)
        with torch.no_grad():
            raw = self.teacher.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     use_cache=False, output_hidden_states=True).hidden_states
        features = [raw[i].detach().to(device) for i in self.experiment.layer_map]
        return features, torch.ones(n, device=device, dtype=torch.bool)

    @contextmanager
    def _inject(self, features, mask, prompt_length):
        handles = []
        n = prompt_length - 1
        try:
            for depth, layer in enumerate(self.student.model.layers):
                def inject(_layer, args, kwargs, slot=depth):
                    hidden = args[0] if args else kwargs['hidden_states']
                    if hidden.shape[1] < prompt_length:
                        return args, kwargs
                    delta = self.projections[slot](features[slot].float())
                    delta = delta * mask[None, :, None]
                    delta = F.pad(delta, (0, 0, 0, hidden.shape[1] - n)).to(hidden.dtype)
                    if args:
                        return (hidden + delta, *args[1:]), kwargs
                    return args, dict(kwargs, hidden_states=hidden + delta)
                handles.append(layer.register_forward_pre_hook(inject, with_kwargs=True))
            yield
        finally:
            for hook in handles:
                hook.remove()

    @torch.no_grad()
    def prefix_cache(self, prompt):
        """Compute the original HF bridge's full prompt hybrid cache and first logits."""
        if self.experiment.method not in ('bridge', 'random') or len(prompt) < 2:
            raise ValueError('Hybrid memory requires a bridge and a valid prompt')
        features, mask = self._features(prompt)
        ids = torch.tensor([prompt], device=self.student.device)
        attention = torch.ones_like(ids)
        with self._inject(features, mask, len(prompt)):
            output = self.student(input_ids=ids, attention_mask=attention,
                                  position_ids=torch.arange(len(prompt), device=ids.device)[None],
                                  use_cache=True, logits_to_keep=1)
        return output.past_key_values, output.logits[0, -1]

    def response_log_probs(self, prompt, answer):
        if not answer:
            raise ValueError('Empty generated answer')
        tokens = torch.tensor([prompt + answer[:-1]], device=self.student.device)
        kind = self.experiment.method
        if kind in ('bridge', 'random'):
            features, mask = self._features(prompt)
            inject = self._inject(features, mask, len(prompt))
        else:
            inject = nullcontext()
        with inject:
            hidden = self.student.model(input_ids=tokens, use_cache=False).last_hidden_state
        hidden = hidden[0, len(prompt) - 1:]
        if hidden.shape[0] != len(answer):
            raise RuntimeError('Answer token coverage differs')
        labels = torch.tensor(answer, device=hidden.device)
        pieces = []
        for start in range(0, len(answer), 128):
            h, y = hidden[start:start + 128], labels[start:start + 128]
            def score(part, target):
                logits = self.student.lm_head(part).float()
                return logits.gather(1, target[:, None]).squeeze(1) - logits.logsumexp(-1)
            if torch.is_grad_enabled() and h.requires_grad:
                pieces.append(checkpoint(score, h, y, use_reentrant=False))
            else:
                pieces.append(score(h, y))
        return torch.cat(pieces)

    @torch.no_grad()
    def generate(self, prompt, *, seed, cap, temperature=1., top_p=1., top_k=-1,
                 presence_penalty=0.):
        kind = self.experiment.method
        if kind in ('bridge', 'random'):
            features, mask = self._features(prompt)
            inject = self._inject(features, mask, len(prompt))
        else:
            inject = nullcontext()
        inputs = torch.tensor([prompt], device=self.student.device)
        device = self.student.device
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        options = {'input_ids': inputs, 'max_new_tokens': cap,
                   'do_sample': temperature > 0,
                   'pad_token_id': self.student.config.eos_token_id}
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                options['top_k'] = top_k
        if presence_penalty:
            from transformers import LogitsProcessor, LogitsProcessorList

            class Presence(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    for row in range(len(input_ids)):
                        scores[row, input_ids[row].unique()] -= presence_penalty
                    return scores

            options['logits_processor'] = LogitsProcessorList([Presence()])
        with torch.random.fork_rng(devices=rng_devices), inject:
            torch.manual_seed(seed)
            if rng_devices:
                torch.cuda.manual_seed(seed)
            output = self.student.generate(**options)
        return output[0, len(prompt):].tolist()


REFERENCE_KERNELS = ('causal_conv1d_fn', 'causal_conv1d_update',
                     'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')


_ORIGINAL_KERNELS, _REFERENCE_KERNELS = {}, {}


def _remember_kernels(modeling):
    """On first contact, record HF's kernel objects and their pure-torch references."""
    for name in REFERENCE_KERNELS:
        key = (modeling.__name__, name)
        value = getattr(modeling, name, None)
        if key in _ORIGINAL_KERNELS or value is None:
            continue
        _ORIGINAL_KERNELS[key] = value
        forward = getattr(type(value), 'forward', None)
        if getattr(forward, '__closure__', None):
            cells = dict(zip(forward.__code__.co_freevars, forward.__closure__))
            if 'func' in cells:
                value = cells['func'].cell_contents
        value = inspect.unwrap(value)
        _REFERENCE_KERNELS[key] = (value if getattr(value, '__module__', None) == modeling.__name__
                                   else None)


def use_reference_kernels(modules):
    """CPU: causal-conv1d/FLA kernels are CUDA-only; bind HF's own pure-torch references.

    HF wraps each kernel (kernels-hub ``Func`` -> functools ``wrapped`` -> torch function);
    the original torch implementation is recovered, never re-implemented here. The
    references are captured before any CUDA kernel replaces them, so CPU and GPU loads
    can alternate within one process.
    """
    for modeling in modules:
        _remember_kernels(modeling)
        for name in REFERENCE_KERNELS:
            key = (modeling.__name__, name)
            if key not in _ORIGINAL_KERNELS:
                continue
            if _REFERENCE_KERNELS[key] is None:
                raise RuntimeError('Cannot recover the HF reference kernel for CPU: ' + name)
            setattr(modeling, name, _REFERENCE_KERNELS[key])


def use_cuda_kernels(modules):
    """GPU: HF's causal-conv1d kernels and FLA gated-delta kernels (the trained runtime)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    for modeling in modules:
        _remember_kernels(modeling)
        for name in ('causal_conv1d_fn', 'causal_conv1d_update'):
            if (modeling.__name__, name) in _ORIGINAL_KERNELS:
                setattr(modeling, name, _ORIGINAL_KERNELS[(modeling.__name__, name)])
        modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        modeling.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule


def load_text(path, *, device='cuda:0', device_map=None):
    """Load the original text model; leave official FP8 experts quantized.

    On a machine without CUDA (or an explicit ``device='cpu'``) HF's reference
    kernels are used; a text-only ``qwen3_5_text`` checkpoint is also accepted.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    import transformers.models.qwen3_5.modeling_qwen3_5 as dense
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as moe
    on_cpu = not torch.cuda.is_available() or (device_map is None and
                                                 torch.device(device).type == 'cpu')
    if on_cpu:
        use_reference_kernels((dense, moe))
    else:
        use_cuda_kernels((dense, moe))
    configuration = __import__('json').loads((Path(path) / 'config.json').read_text())
    fp8 = configuration.get('quantization_config', {}).get('quant_method') == 'fp8'
    if fp8 and on_cpu:
        raise RuntimeError('FP8 Qwen3.5 experts require CUDA kernels')
    if fp8:
        from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING, allow_all_hub_kernels
        from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
        _HUB_KERNEL_MAPPING['finegrained-fp8'] = {
            'repo_id': 'kernels-community/finegrained-fp8',
            'revision': '8c178950e8710e26a2210e4e909cd02dc16c8715'}
        with allow_all_hub_kernels():  # exact pinned commit; no network publisher lookup
            load_finegrained_fp8_kernel()
    options = {'dtype': torch.bfloat16, 'attn_implementation': 'sdpa', 'local_files_only': True}
    if fp8:
        options['experts_implementation'] = 'grouped_mm'
    if device_map is not None:
        options['device_map'] = device_map
    if configuration.get('model_type') == 'qwen3_5_text':
        text = AutoModelForCausalLM.from_pretrained(str(path), **options)
        if device_map is None:
            text.to(device)
        return text.eval()
    full = AutoModelForImageTextToText.from_pretrained(str(path), **options)
    with torch.device('meta'):
        text = AutoModelForCausalLM.from_config(full.config.text_config)
    text.model = full.model.language_model
    text.lm_head = full.lm_head
    text.generation_config = full.generation_config
    if fp8:
        text.hf_quantizer = full.hf_quantizer
        text.is_quantized = full.is_quantized
        text.set_experts_implementation('grouped_mm')
        for name, parameter in text.named_parameters():
            if name.endswith('_scale_inv'):
                parameter.data = parameter.data.to(torch.float32)
    if any(parameter.is_meta for parameter in text.parameters()):
        raise RuntimeError('Qwen3.5 text weights were not fully materialized')
    if device_map is None:
        text.to(device)
    del full
    return text.eval()
