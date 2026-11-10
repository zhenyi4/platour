"""Initialization and projection-depth ablations of the Qwen3 hidden GRPO bridge."""
import math

import torch
from torch import nn

from direct_residual_chat_bridge import DirectResidualChatBridge


class TwoLayerHiddenProjection(nn.Module):
    def __init__(self,input_dim,output_dim,device):
        super().__init__()
        self.in_features=self.hidden_features=input_dim
        self.out_features=output_dim
        options=dict(device=device,dtype=torch.float32,bias=False)
        self.first=nn.Linear(input_dim,input_dim,**options)
        self.activation=nn.GELU()
        self.second=nn.Linear(input_dim,output_dim,**options)

    def forward(self,features):
        return self.second(self.activation(self.first(features)))


class Qwen3HiddenGRPOAblation(DirectResidualChatBridge):
    def __init__(self,student,teacher,transfer_mode):
        if transfer_mode not in ('hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard'):
            raise ValueError('Expected a hidden GRPO ablation mode')
        super().__init__(student,teacher,variant='teacher_residual',initialization='zero')
        self.transfer_mode=transfer_mode
        if transfer_mode=='hidden_random_standard':
            # Already nontrivial: parity should test the actual initialization,
            # not amplify standard-size weights by another factor of 1000.
            self.handoff_probe_scale=1.
        if transfer_mode in ('hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero'):
            self.maps=nn.ModuleList([TwoLayerHiddenProjection(teacher.config.hidden_size,
                student.config.hidden_size,student.device) for _ in self.layer_map])
            if transfer_mode=='hidden_factorized_w2zero':
                for projection in self.maps:projection.activation=nn.Identity()
            self.initialization='random'

    def initialize_for_grpo(self,seed=42):
        if self.transfer_mode in ('hidden_mlp_w2zero','hidden_factorized_w2zero'):
            generator=torch.Generator(device=self.student.device).manual_seed(seed)
            with torch.no_grad():
                for projection in self.maps:
                    nn.init.kaiming_uniform_(projection.first.weight,a=math.sqrt(5),generator=generator)
                    nn.init.zeros_(projection.second.weight)
            self.initialization='W1 random; W2 zero'
            return dict(kind='W1 Kaiming uniform a=sqrt(5); W2 zero',seed=seed,
                constant_gate=1.,sft_checkpoint_loaded=False,transfer_mode=self.transfer_mode)
        scale=1. if self.transfer_mode=='hidden_random_standard' else 1e-4
        bound=scale/math.sqrt(self.teacher.config.hidden_size)
        generator=torch.Generator(device=self.student.device).manual_seed(seed)
        zero=self.transfer_mode=='hidden_zero'
        with torch.no_grad():
            for parameter in self.bridge_parameters():
                if zero:parameter.zero_()
                else:parameter.uniform_(-bound,bound,generator=generator)
        self.initialization='zero' if zero else 'random'
        kind=('zero' if zero else 'standard random uniform linear' if scale==1. else
              'small random uniform on both MLP matrices')
        return dict(kind=kind,
            seed=seed,bound=0. if zero else bound,scale=0. if zero else scale,
            constant_gate=1.,sft_checkpoint_loaded=False,transfer_mode=self.transfer_mode)

    def handoff_probe_parameters(self):
        if self.transfer_mode in ('hidden_mlp_w2zero','hidden_factorized_w2zero'):
            return [m.second.weight for m in self.maps]
        return self.bridge_parameters()

    def metadata(self):
        metadata=super().metadata()
        zero=self.transfer_mode=='hidden_zero'
        linear=self.transfer_mode not in ('hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero')
        teacher_width=self.teacher.config.hidden_size
        initialization=('all-zero linear projection matrices' if zero else
            f'linear uniform +/-1/sqrt({teacher_width}), seed42' if linear else
            f'both MLP matrices uniform +/-1e-4/sqrt({teacher_width}), seed42')
        metadata.update(transfer_mode=self.transfer_mode,constant_gate=1.,
            algorithm='direct_hidden_linear_residual' if linear else 'direct_hidden_mlp_residual',
            injection='h_s + P(h_t)' if linear else 'h_s + W2(GELU(W1(h_t)))',
            initialization=initialization,
            projection=dict(type='linear' if linear else 'two_layer_mlp',
                input_dim=teacher_width,output_dim=self.student.config.hidden_size,
                intermediate_dim=None if linear else teacher_width,
                activation=None if linear else 'GELU exact',bias=False),
            trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))
        if self.transfer_mode in ('hidden_mlp_w2zero','hidden_factorized_w2zero'):
            nonlinear=self.transfer_mode=='hidden_mlp_w2zero'
            metadata['initialization']='W1 Kaiming uniform a=sqrt(5), W2 all-zero, seed42'
            metadata['projection']['activation']='GELU exact' if nonlinear else None
            metadata['injection']='h_s + W2(GELU(W1(h_t)))' if nonlinear else 'h_s + W2(W1(h_t))'
        return metadata
