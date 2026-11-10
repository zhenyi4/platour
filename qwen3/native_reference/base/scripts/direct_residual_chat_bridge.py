"""Linear residual bridges from teacher, student+teacher, or neighboring teacher layers."""
import torch
from torch import nn
from torch.nn import functional as F

from hidden_bridge import bridge_layer_map
from shared_subspace_chat_bridge import SharedSubspaceChatBridge

VARIANTS=('teacher_residual','student_teacher_residual','teacher_window_residual','teacher_window5_residual')


class DirectResidualChatBridge(SharedSubspaceChatBridge):
    """Reuse the exact CE loss and prefix injection contract of the existing bridge."""
    def __init__(self,student,teacher,*,variant,initialization='random',lambda_lm=1.,lm_chunk_tokens=256,layer_mapping='uniform_endpoints'):
        nn.Module.__init__(self)
        if variant not in VARIANTS or lambda_lm<=0 or lm_chunk_tokens<1:
            raise ValueError('Need a supported residual variant and positive CE settings')
        if initialization not in ('random','zero'):
            raise ValueError('Use random or zero projection initialization')
        self.student=student.requires_grad_(False).eval()
        self.teacher=teacher.requires_grad_(False).eval()
        self.variant=variant;self.initialization=initialization;self.lambda_lm=lambda_lm;self.lm_chunk_tokens=lm_chunk_tokens
        self.compute_mse=False
        self.layer_mapping=layer_mapping
        self.layer_map=bridge_layer_map(student.config.num_hidden_layers,teacher.config.num_hidden_layers,layer_mapping)
        depth=teacher.config.num_hidden_layers
        radius={'teacher_window_residual':1,'teacher_window5_residual':2}.get(variant,0)
        self.assignment=[[min(depth-1,max(0,j+offset)) for offset in range(-radius,radius+1)] for j in self.layer_map]
        width=teacher.config.hidden_size*(2*radius+1)
        if variant=='student_teacher_residual':width+=student.config.hidden_size
        self.maps=nn.ModuleList([nn.Linear(width,student.config.hidden_size,bias=False,
            device=student.device,dtype=torch.float32) for _ in self.layer_map])
        if initialization=='zero':
            for projection in self.maps:nn.init.zeros_(projection.weight)

    def projection_groups(self):return dict(P=self.maps)

    def teacher_features(self,hidden):
        return tuple(torch.cat([hidden[j] for j in window],dim=-1) for window in self.assignment)

    def residual(self,index,hidden,teacher_hidden,injection_mask):
        n=injection_mask.shape[1]
        if teacher_hidden.shape[:2]!=hidden[:,:n].shape[:2]:
            raise ValueError('Teacher and student prefixes must align')
        with torch.autocast(hidden.device.type,enabled=False):
            features=teacher_hidden.float()
            if self.variant=='student_teacher_residual':
                # Preserve gradients through the current student state and all
                # residual bridges applied in preceding layers.
                features=torch.cat([hidden[:,:n].float(),features],dim=-1)
            delta=self.maps[index](features)*injection_mask.unsqueeze(-1)
        if n<hidden.shape[1]:delta=F.pad(delta,(0,0,0,hidden.shape[1]-n))
        assert delta.shape==hidden.shape
        return delta.to(hidden.dtype)

    def metadata(self):
        formulas={
            'teacher_residual':'h_s + P(h_t)',
            'student_teacher_residual':'h_s + P(concat(h_s, h_t))',
            'teacher_window_residual':'h_s + P(concat(h_t[j-1], h_t[j], h_t[j+1]))',
            'teacher_window5_residual':'h_s + P(concat(h_t[j-2], h_t[j-1], h_t[j], h_t[j+1], h_t[j+2]))',
        }
        return dict(algorithm='direct_hidden_linear_residual',variant=self.variant,injection=formulas[self.variant],
            student_dim=self.student.config.hidden_size,teacher_dim=self.teacher.config.hidden_size,
            input_dim=self.maps[0].in_features,output_dim=self.maps[0].out_features,
            layer_mapping=self.layer_mapping,layer_map=self.layer_map,teacher_assignment=self.assignment,
            boundary='repeat boundary layers',hidden_space='decoder block input before RMSNorm',
            bias=False,projection_dtype='float32',initialization=('all-zero projection matrices'
                if self.initialization=='zero' else 'PyTorch Linear default random'),
            student_hidden_detached=False,prefix_only=True,last_prefix_token_recomputed=True)
