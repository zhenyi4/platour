"""Prefix residual from teacher/student encodings of the current hidden."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn

from hidden_bridge import HiddenBridge


class SharedSubspaceResidualBridge(HiddenBridge):
    def __init__(self, student, teacher, *, initial_projectors=None, train_projectors=False,
                 latent_dim=None, residual_activation='identity', post_projection_init='zero', **kwargs):
        if latent_dim is None:
            latent_dim = (json.loads(Path(initial_projectors).with_name('subspace_config.json').read_text())['latent_dim']
                          if initial_projectors is not None else 512)
        if latent_dim < 1 or residual_activation not in ('identity', 'relu'):
            raise ValueError('Positive latent dimension and identity/relu activation required')
        if post_projection_init not in ('zero', 'student_decoder'):
            raise ValueError('Post projection initialization must be zero or student_decoder')
        if post_projection_init == 'student_decoder' and initial_projectors is None:
            raise ValueError('Student decoder initialization requires pretrained bridge weights')
        kwargs.pop('gate_mode', None)
        kwargs.pop('layer_mapping', None)
        super().__init__(student, teacher, gate_mode='none', layer_mapping='uniform_endpoints', **kwargs)
        self.latent_dim = latent_dim
        self.train_projectors = train_projectors
        self.residual_activation = residual_activation
        self.post_projection_init = post_projection_init
        self.projectors = self.make_group(teacher.config.hidden_size, latent_dim).requires_grad_(train_projectors)
        self.student_encoders = self.make_group(student.config.hidden_size, latent_dim).requires_grad_(train_projectors)
        if initial_projectors is not None:
            checkpoint = Path(initial_projectors)
            config = json.loads(checkpoint.with_name('subspace_config.json').read_text())
            expected = dict(algorithm='bidirectional_shared_subspace_self_reconstruction',
                            latent_dim=latent_dim, student_dim=student.config.hidden_size,
                            teacher_dim=teacher.config.hidden_size, layer_map=self.layer_map,
                            layer_mapping='uniform_endpoints', bias=False)
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError('Pretrained shared-subspace dimensions or layer matching differ')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.projectors.load_state_dict(state['P_t'])
            self.student_encoders.load_state_dict(state['P_s'])
        self.gate_mode = 'shared_subspace_difference'
        # A bias-free linear makes the residual zero when the two latents match.
        self.gates = self.make_group(latent_dim, student.config.hidden_size)
        for linear in self.gates:
            nn.init.zeros_(linear.weight)
        if post_projection_init == 'student_decoder':
            self.gates.load_state_dict(state['D_s'])
            self.initial_decoder_copy_verified = all(torch.equal(value.cpu(), state['D_s'][key])
                for key, value in self.gates.state_dict().items())
            if not self.initial_decoder_copy_verified:
                raise ValueError('Student decoder initialization must preserve exact FP32 weights')

    def make_group(self, input_dim, output_dim):
        return nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False,
                              device=self.student.device, dtype=torch.float32) for _ in self.layer_map])

    def bridge_parameters(self):
        parameters = tuple(self.gates.parameters())
        if self.train_projectors:
            parameters += tuple(self.projectors.parameters()) + tuple(self.student_encoders.parameters())
        return parameters

    @classmethod
    def from_paths(cls, student_path, teacher_path, checkpoint=None, **options):
        if checkpoint is not None:
            metadata = json.loads(Path(checkpoint).with_name('gate_config.json').read_text())
            options.setdefault('latent_dim', metadata['latent_dim'])
            options.setdefault('residual_activation', metadata.get('residual_activation', 'identity'))
            options.setdefault('train_projectors', metadata.get('encoders_trainable', False))
        return super().from_paths(student_path, teacher_path, checkpoint=checkpoint, **options)

    def residual(self, index, hidden, teacher_hidden, injection_mask):
        n = injection_mask.shape[1]
        if teacher_hidden.shape[:2] != hidden[:, :n].shape[:2]:
            raise ValueError('Teacher prefix and current student prefix must align')
        with torch.autocast(hidden.device.type, enabled=False):
            teacher_latent = self.projectors[index](teacher_hidden.float())
            # Retain gradients to current h_s, including when encoder weights are frozen.
            student_latent = self.student_encoders[index](hidden[:, :n].float())
            difference = teacher_latent - student_latent
            if self.residual_activation == 'relu':
                difference = nn.functional.relu(difference)
            delta = self.gates[index](difference)
            delta = delta * injection_mask.unsqueeze(-1).to(delta.dtype)
        if n < hidden.shape[1]:
            delta = nn.functional.pad(delta, (0, 0, 0, hidden.shape[1] - n))
        if delta.shape != hidden.shape:
            raise ValueError(f'Residual shape {delta.shape} != student hidden {hidden.shape}')
        return delta.to(hidden.dtype)

    @contextmanager
    def inject(self, teacher_hidden, injection_mask):
        handles = []

        def make_hook(index):
            def hook(module, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                updated = hidden + self.residual(index, hidden, teacher_hidden[index], injection_mask)
                if args:
                    return (updated, *args[1:]), kwargs
                return args, {**kwargs, 'hidden_states': updated}
            return hook

        try:
            for index, layer in enumerate(self.student.model.layers):
                handles.append(layer.register_forward_pre_hook(make_hook(index), with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def save_projectors(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.projectors.state_dict(), directory / 'projectors.pt')
        torch.save(self.student_encoders.state_dict(), directory / 'student_encoders.pt')
        torch.save(self.gates.state_dict(), directory / 'gates.pt')
        (directory / 'layer_mapping.json').write_text(json.dumps(dict(
            layer_mapping=self.layer_mapping, layer_map=self.layer_map), indent=2) + '\n')
        (directory / 'projector_config.json').write_text(json.dumps(dict(
            projector_type='linear', teacher_width=self.teacher.config.hidden_size,
            student_width=self.latent_dim, bias=False)) + '\n')
        (directory / 'gate_config.json').write_text(json.dumps(dict(
            gate_mode=self.gate_mode, latent_dim=self.latent_dim, post_projection_bias=False,
            injection='h_s + W_new(activation(P_t(h_t) - P_s(h_s)))', student_hidden_detached=False,
            residual_activation=self.residual_activation, encoders_trainable=self.train_projectors,
            post_projection_init=self.post_projection_init)) + '\n')

    def load_projectors(self, checkpoint):
        checkpoint = Path(checkpoint)
        metadata = json.loads(checkpoint.with_name('gate_config.json').read_text())
        if metadata['gate_mode'] != self.gate_mode or metadata['latent_dim'] != self.latent_dim:
            raise ValueError('Checkpoint residual mode or latent dimension differs')
        if metadata.get('residual_activation', 'identity') != self.residual_activation:
            raise ValueError('Checkpoint residual activation differs')
        if metadata['post_projection_bias'] or metadata['student_hidden_detached']:
            raise ValueError('Checkpoint must use a bias-free residual and attached student hidden')
        mapping = json.loads(checkpoint.with_name('layer_mapping.json').read_text())
        if mapping != dict(layer_mapping=self.layer_mapping, layer_map=self.layer_map):
            raise ValueError('Checkpoint layer mapping differs')
        for modules, filename in [(self.projectors, 'projectors.pt'),
                                  (self.student_encoders, 'student_encoders.pt'),
                                  (self.gates, 'gates.pt')]:
            modules.load_state_dict(torch.load(checkpoint.with_name(filename), map_location='cpu', weights_only=True))
        self.post_projection_init = metadata.get('post_projection_init', 'zero')
