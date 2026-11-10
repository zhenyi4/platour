"""Frozen-model hidden alignment in a shared latent space with self-reconstruction."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from hidden_bridge import bridge_layer_map


def shared_subspace_loss(student_encoders, teacher_encoders, student_decoders, teacher_decoders,
                         student_hidden, teacher_hidden, lambda_align=1., lambda_recon=1.):
    groups = (student_encoders, teacher_encoders, student_decoders, teacher_decoders,
              student_hidden, teacher_hidden)
    if not student_encoders or len({len(group) for group in groups}) != 1:
        raise ValueError('Each matched layer requires two encoders, two decoders and two hiddens')
    values = {}
    for ps, pt, ds, dt, hs, ht in zip(*groups, strict=True):
        with torch.autocast(hs.device.type, enabled=False):
            hs, ht = hs.detach().float(), ht.detach().float()
            zs, zt = ps(hs), pt(ht)
            rs, rt = ds(zs), dt(zt)
            metrics = dict(align_student_mse=(zs - zt.detach()).square().mean(),
                           align_teacher_mse=(zt - zs.detach()).square().mean(),
                           recon_student_mse=(rs - hs).square().mean(),
                           recon_teacher_mse=(rt - ht).square().mean())
            with torch.no_grad():
                metrics.update(align_cosine=nn.functional.cosine_similarity(zs, zt, dim=-1).mean(),
                    recon_student_cosine=nn.functional.cosine_similarity(rs, hs, dim=-1).mean(),
                    recon_teacher_cosine=nn.functional.cosine_similarity(rt, ht, dim=-1).mean(),
                    student_latent_rms=zs.square().mean().sqrt(),
                    teacher_latent_rms=zt.square().mean().sqrt(),
                    student_latent_std=zs.reshape(-1, zs.shape[-1]).std(0, correction=0).mean(),
                    teacher_latent_std=zt.reshape(-1, zt.shape[-1]).std(0, correction=0).mean())
            for key, value in metrics.items():
                values.setdefault(key, []).append(value)
    metrics = {key: torch.stack(value).mean() for key, value in values.items()}
    # Each directional alignment has its own detached target. Neither latent
    # is detached on its reconstruction path.
    total = (lambda_align * (metrics['align_student_mse'] + metrics['align_teacher_mse'])
             + lambda_recon * (metrics['recon_student_mse'] + metrics['recon_teacher_mse']))
    return total, metrics


class SharedSubspaceBridge(nn.Module):
    algorithm = 'bidirectional_shared_subspace_self_reconstruction'

    def __init__(self, student, teacher, *, latent_dim=512, lambda_align=1., lambda_recon=1.):
        super().__init__()
        if latent_dim < 1:
            raise ValueError('Latent dimension must be positive')
        self.student = student.requires_grad_(False).eval()
        self.teacher = teacher.requires_grad_(False).eval()
        self.latent_dim = latent_dim
        self.lambda_align, self.lambda_recon = lambda_align, lambda_recon
        self.layer_map = bridge_layer_map(student.config.num_hidden_layers, teacher.config.num_hidden_layers,
                                          'uniform_endpoints')
        self.student_encoders = self.make_group(student.config.hidden_size, latent_dim)
        self.teacher_encoders = self.make_group(teacher.config.hidden_size, latent_dim)
        self.student_decoders = self.make_group(latent_dim, student.config.hidden_size)
        self.teacher_decoders = self.make_group(latent_dim, teacher.config.hidden_size)

    def make_group(self, input_dim, output_dim):
        return nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False,
                              device=self.student.device, dtype=torch.float32) for _ in self.layer_map])

    def projection_groups(self):
        return dict(P_s=self.student_encoders, P_t=self.teacher_encoders,
                    D_s=self.student_decoders, D_t=self.teacher_decoders)

    def bridge_parameters(self):
        return tuple(p for group in self.projection_groups().values() for p in group.parameters())

    @classmethod
    def from_paths(cls, student_path, teacher_path, **options):
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        student = AutoModelForCausalLM.from_pretrained(student_path, **kwargs).to('cuda')
        teacher = AutoModelForCausalLM.from_pretrained(teacher_path, **kwargs).to('cuda')
        return cls(student, teacher, **options)

    def representation_loss(self, ids):
        with torch.no_grad():
            teacher = self.teacher.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
            student = self.student.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
        return shared_subspace_loss(self.student_encoders, self.teacher_encoders,
                                   self.student_decoders, self.teacher_decoders,
                                   student[:-1], [teacher[i] for i in self.layer_map],
                                   self.lambda_align, self.lambda_recon)

    def metadata(self):
        return dict(algorithm=self.algorithm, latent_dim=self.latent_dim,
                    student_dim=self.student.config.hidden_size, teacher_dim=self.teacher.config.hidden_size,
                    layer_mapping='uniform_endpoints', layer_map=self.layer_map,
                    lambda_align=self.lambda_align, lambda_recon=self.lambda_recon,
                    initialization='PyTorch Linear default random', bias=False, projection_dtype='float32')

    def save_projectors(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({name: group.state_dict() for name, group in self.projection_groups().items()}, directory / 'bridge.pt')
        (directory / 'subspace_config.json').write_text(json.dumps(self.metadata(), indent=2) + '\n')

    def load_projectors(self, checkpoint):
        checkpoint = Path(checkpoint)
        if json.loads(checkpoint.with_name('subspace_config.json').read_text()) != self.metadata():
            raise ValueError('Checkpoint shared-subspace configuration does not match model')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.keys() != self.projection_groups().keys():
            raise ValueError('Checkpoint must contain all four projection groups')
        for name, group in self.projection_groups().items():
            group.load_state_dict(state[name])
