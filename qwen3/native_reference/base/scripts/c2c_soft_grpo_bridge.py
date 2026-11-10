"""C2C fuser with deterministic soft gates for on-policy GRPO.

Topology follows C2CProjector / StandardFFNLayer / RegularMLP from
thu-nics/C2C commit 113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a:
https://github.com/thu-nics/C2C/blob/113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a/rosetta/model/projector.py
The local implementation avoids importing the official SFT wrapper/runtime.
Deliberate GRPO adaptations: no dropout/Gumbel; soft gates during GRPO,
small-random output projections, and the existing equal-gap layer mapping.
"""
import math
from contextlib import contextmanager

import torch
from torch import nn

from qwen3_kv_residual_bridge import Qwen3KVResidualBridge


C2C_REVISION = '113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a'


def fuser_parameter_count(source_width, target_width, target_heads,
                          hidden_dim=1024, intermediate_dim=1024, num_layers=3):
    if num_layers < 3:
        raise ValueError('C2C requires at least three projection stages')
    h, m = hidden_dim, intermediate_dim
    branch = ((source_width + target_width) * h + h
              + (num_layers - 1) * (h + 2 * h * m)
              + h + 2 * h * h
              + h * target_width + target_width
              + h * target_heads + target_heads + 1)
    return 2 * branch


class StandardFFN(nn.Module):
    """Official pre-RMSNorm, bias-free GELU FFN with an internal residual."""
    def __init__(self, hidden, intermediate, device=None):
        super().__init__()
        options=dict(device=device, dtype=torch.float32)
        self.norm=nn.RMSNorm(hidden, eps=1e-6, **options)
        self.w1=nn.Linear(hidden, intermediate, bias=False, **options)
        self.w2=nn.Linear(intermediate, hidden, bias=False, **options)
        self.act=nn.GELU()

    def forward(self, x):
        return x + self.w2(self.act(self.w1(self.norm(x))))


class RegularMLP(nn.Module):
    def __init__(self, hidden, intermediate, layers, device=None):
        super().__init__()
        self.blocks=nn.ModuleList([StandardFFN(hidden, intermediate, device) for _ in range(layers)])

    def forward(self, x):
        for block in self.blocks:
            x=block(x)
        return x


class C2CSoftProjector(nn.Module):
    def __init__(self, source_width, target_heads, target_head_dim,
                 hidden_dim=1024, intermediate_dim=1024, num_layers=3, device=None):
        super().__init__()
        if num_layers < 3:
            raise ValueError('C2C requires at least three projection stages')
        self.source_width=source_width
        self.target_heads, self.target_head_dim=target_heads, target_head_dim
        self.hidden_dim, self.intermediate_dim, self.num_layers=hidden_dim, intermediate_dim, num_layers
        self.gate_evaluation_mode='soft'
        target_width=target_heads * target_head_dim
        options=dict(device=device, dtype=torch.float32)
        for kind in ('key','value'):
            setattr(self,kind+'_in',nn.Linear(source_width+target_width,hidden_dim,**options))
            setattr(self,kind+'_mlp1',RegularMLP(hidden_dim,intermediate_dim,1,device))
            setattr(self,kind+'_scalar_mlp2',RegularMLP(hidden_dim,hidden_dim,1,device))
            setattr(self,kind+'_scalar_head',nn.Linear(hidden_dim,target_heads,**options))
            setattr(self,kind+'_proj_mlp2',RegularMLP(hidden_dim,intermediate_dim,num_layers-2,device))
            setattr(self,kind+'_proj_out',nn.Linear(hidden_dim,target_width,**options))
            setattr(self,kind+'_gate_logit',nn.Parameter(torch.zeros((),**options)))

    def forward(self, source_kv, target_kv):
        result=[]
        for kind, source, target in zip(('key','value'),source_kv,target_kv,strict=True):
            batch, heads, length, dim=target.shape
            if (heads,dim)!=(self.target_heads,self.target_head_dim):
                raise ValueError('Target KV layout differs from projector configuration')
            source_flat=source.transpose(1,2).reshape(batch,length,self.source_width).float()
            target_flat=target.transpose(1,2).reshape(batch,length,heads*dim).float()
            hidden=getattr(self,kind+'_mlp1')(getattr(self,kind+'_in')(
                torch.cat((source_flat,target_flat),-1)))
            delta=getattr(self,kind+'_proj_out')(getattr(self,kind+'_proj_mlp2')(hidden))
            delta=delta.view(batch,length,heads,dim).transpose(1,2)
            weight=getattr(self,kind+'_scalar_head')(getattr(self,kind+'_scalar_mlp2')(hidden))
            weight=weight.sigmoid().transpose(1,2).unsqueeze(-1)
            logit=getattr(self,kind+'_gate_logit')
            gate=(logit>0).to(logit.dtype) if self.gate_evaluation_mode=='hard' else logit.sigmoid()
            result.append(target.float()+gate*weight*delta)
        return tuple(result)

    def initialize_output(self):
        bound=1e-4/math.sqrt(self.hidden_dim)
        with torch.no_grad():
            for kind in ('key','value'):
                output=getattr(self,kind+'_proj_out')
                output.weight.uniform_(-bound,bound)
                output.bias.zero_()
                head=getattr(self,kind+'_scalar_head')
                head.weight.zero_()
                head.bias.zero_()
                getattr(self,kind+'_gate_logit').zero_()


class Qwen3C2CSoftBridge(Qwen3KVResidualBridge):
    def __init__(self, student, teacher, hidden_dim=1024, intermediate_dim=1024, num_layers=3):
        # Reuse only the full-attention prefix/cache machinery. The small linear
        # maps constructed by the parent are replaced with the C2C fusers.
        super().__init__(student,teacher,mode='post_prefill')
        self.hidden_dim,self.intermediate_dim,self.num_layers=hidden_dim,intermediate_dim,num_layers
        self.maps=nn.ModuleList([C2CSoftProjector(self.teacher_kv_width,
            student.config.num_key_value_heads,student.config.head_dim,
            hidden_dim,intermediate_dim,num_layers,student.device) for _ in self.layer_map])
        self.initialization='uninitialized C2C; call shared GRPO initialize() or load checkpoint'

    def initialize_for_grpo(self, seed=42):
        devices=[self.student.device.index] if self.student.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.student.device):
                    torch.cuda.manual_seed(seed)
            # Normal PyTorch linear initialization and RMSNorm gain=1 inside.
            for module in self.maps.modules():
                if isinstance(module,(nn.Linear,nn.RMSNorm)):
                    module.reset_parameters()
            for fuser in self.maps:
                fuser.initialize_output()
        self.initialization='standard internal MLP; output uniform +/-1e-4/sqrt(hidden_dim); output biases/head logits/gate logits zero'
        return dict(kind='c2c_soft_small_output',seed=seed,scale=1e-4,
            bound=1e-4/math.sqrt(self.hidden_dim),output_fan_in=self.hidden_dim,
            internal_initialization='PyTorch default Linear; RMSNorm gain=1',
            gate_initial_value=.5,dynamic_head_initial_value=.5,gate_mode='deterministic sigmoid',
            dropout=0.,gumbel=False,sft_checkpoint_loaded=False)

    def fuse(self,index,key,value,features,mask):
        fused=self.maps[index](features,(key,value))
        valid=mask[:,None,:,None].bool()
        return tuple(torch.where(valid,new.to(native.dtype),native)
                     for native,new in zip((key,value),fused,strict=True))

    def handoff_probe_parameters(self):
        return tuple(getattr(fuser,kind+'_proj_out').weight
                     for fuser in self.maps for kind in ('key','value'))

    @contextmanager
    def evaluation_gates(self, mode):
        """Temporarily switch only scalar gates; always restore rollout behavior."""
        if mode not in ('soft','hard'):
            raise ValueError(f'Unknown C2C evaluation gate mode: {mode}')
        previous=[fuser.gate_evaluation_mode for fuser in self.maps]
        try:
            for fuser in self.maps:fuser.gate_evaluation_mode=mode
            yield
        finally:
            for fuser,old in zip(self.maps,previous,strict=True):fuser.gate_evaluation_mode=old

    def evaluation_gate_statistics(self):
        logits=torch.stack([getattr(fuser,kind+'_gate_logit').detach()
                            for fuser in self.maps for kind in ('key','value')])
        return dict(scalar_gates=logits.numel(),hard_active=int((logits>0).sum()),
                    hard_rule='gate_logit > 0; zero is off',
                    soft_gate_mean=float(logits.sigmoid().mean()),
                    head_weight_mode='continuous sigmoid in both evaluations')

    def metadata(self):
        value=super().metadata()
        value.update(bridge_type='qwen3_c2c_soft_grpo',transfer_mode='c2c_soft',
            formula='KV_s + sigmoid(gate_logit) * sigmoid(head_logits(KV_s, KV_t)) * F(KV_s, KV_t)',
            initialization=self.initialization,constant_gate=None,
            gate_mode='deterministic sigmoid for rollout/update/eval',gumbel=False,dropout=0.,
            eval_gate_modes=['soft','hard'],
            hard_eval_gate_rule='scalar gate_logit > 0; zero off; dynamic head weights remain sigmoid',
            fuser=dict(hidden_dim=self.hidden_dim,intermediate_dim=self.intermediate_dim,num_layers=self.num_layers,
                       activation='GELU',normalization='RMSNorm eps=1e-6',
                       initial_output_scale=1e-4,initial_gate=.5,initial_head_weight=.5),
            c2c_source_revision=C2C_REVISION,
            c2c_adaptations=['equal-gap layer mapping','deterministic soft gates','dropout disabled',
                             'small-random output initialization','GRPO instead of SFT'])
        return value
