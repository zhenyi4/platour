"""Prefix-only four-term MSE, optional reply CE, for padded mixed-mode chat."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from shared_subspace_bridge import SharedSubspaceBridge,shared_subspace_loss
from shared_subspace_residual_bridge import SharedSubspaceResidualBridge


class SharedSubspaceChatBridge(SharedSubspaceBridge):
    residual_activation='identity'
    residual=SharedSubspaceResidualBridge.residual
    inject=SharedSubspaceResidualBridge.inject

    def __init__(self,*args,lambda_lm=0.,lm_chunk_tokens=256,compute_mse=True,**kwargs):
        super().__init__(*args,**kwargs)
        if lambda_lm<0 or lm_chunk_tokens<1:raise ValueError('Invalid CE weight/chunk')
        self.lambda_lm=lambda_lm;self.lm_chunk_tokens=lm_chunk_tokens
        self.compute_mse=compute_mse
        if not compute_mse:
            if not lambda_lm:raise ValueError('CE-only requires a positive CE weight')
            self.teacher_decoders.requires_grad_(False)

    def bridge_parameters(self):
        return tuple(p for p in super().bridge_parameters() if p.requires_grad)

    @property
    def projectors(self):return self.teacher_encoders

    @property
    def gates(self):return self.student_decoders

    def teacher_features(self,hidden):
        return tuple(hidden[i] for i in self.layer_map)

    def chat_loss(self,batch,evaluate_lm=False):
        ids,mask=batch['prefix_ids'],batch['prefix_mask']
        with torch.no_grad():
            teacher=self.teacher.model(input_ids=ids,attention_mask=mask.long(),use_cache=False,
                output_hidden_states=True).hidden_states
            if self.compute_mse:
                student=self.student.model(input_ids=ids,attention_mask=mask.long(),use_cache=False,
                    output_hidden_states=True).hidden_states
        teacher=self.teacher_features(teacher)
        # Remove padding before any MSE/cosine computation. Replies never enter
        # these forwards; each example retains its true assistant-start boundary.
        representation=teacher[0].new_zeros((),dtype=torch.float32);metrics={}
        if self.compute_mse:
            representation,metrics=shared_subspace_loss(self.student_encoders,self.teacher_encoders,
                self.student_decoders,self.teacher_decoders,[h[mask][None] for h in student[:-1]],
                [h[mask][None] for h in teacher],self.lambda_align,self.lambda_recon)
            metrics['representation_loss']=representation
        lm=representation.new_zeros(())
        if self.lambda_lm or evaluate_lm:
            # Match inference: import/inject prompt[:-1], then let the frozen
            # target recompute the final prompt token and generate the reply.
            injection_mask=mask.clone()
            last_positions=mask.long().sum(1)-1
            injection_mask[torch.arange(len(mask),device=mask.device),last_positions]=False
            with self.inject(teacher,injection_mask):
                hidden=self.student.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                    use_cache=False).last_hidden_state
            labels=batch['labels'][:,1:]
            active=labels!=-100
            predictors=hidden[:,:-1][active];targets=labels[active]
            assert targets.numel()>0
            def chunk(h,y):return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
            pieces=[]
            for offset in range(0,len(targets),self.lm_chunk_tokens):
                h=predictors[offset:offset+self.lm_chunk_tokens];y=targets[offset:offset+self.lm_chunk_tokens]
                pieces.append(checkpoint(chunk,h,y,use_reentrant=False) if torch.is_grad_enabled() else chunk(h,y))
            lm=torch.stack(pieces).sum()/len(targets)
            metrics.update(lm_nll=lm,weighted_lm_loss=self.lambda_lm*lm)
        return representation,lm,metrics
