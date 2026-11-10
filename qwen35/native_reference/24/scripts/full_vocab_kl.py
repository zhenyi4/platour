"""Exact forward or reverse KL over every LM-head output, with bounded logit memory."""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def full_vocab_kl(student_hidden, teacher_hidden, student_head, teacher_head,
                  response_mask, *, temperature=1.0, chunk_tokens=64, direction="forward"):
    """Mean T² KL on selected next-token positions.

    Forward is KL(teacher || student); reverse is KL(student || teacher).

    Inputs are final normalized decoder states aligned to the SAME prediction
    positions. The caller shifts the response mask by one token. Each chunk
    contains complete vocabularies: there is no top-k selection or sampled KL.
    Frozen LM heads are checkpointed during training so backward recomputes
    logits per token chunk instead of retaining an N_tokens x V tensor.
    """
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be forward or reverse")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not isinstance(chunk_tokens, int) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if student_hidden.shape[:-1] != teacher_hidden.shape[:-1] or response_mask.shape != student_hidden.shape[:-1]:
        raise ValueError("Teacher, student and response mask positions must agree")
    if response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean")
    if student_head.weight.shape[0] != teacher_head.weight.shape[0]:
        raise ValueError("Full-vocabulary KL requires identical output vocabularies")
    if any(p.requires_grad for head in (student_head, teacher_head) for p in head.parameters()):
        raise ValueError("Hidden bridge KL expects both LM heads to be frozen")
    student = student_hidden[response_mask]
    teacher = teacher_hidden.detach()[response_mask]
    count = student.shape[0]
    if count == 0:
        raise ValueError("No response prediction positions selected")

    def chunk_loss(s, t):
        # Keep each model's head matmul dtype; normalize/reduce in FP32.
        log_student = F.log_softmax(student_head(s).float() / temperature, dim=-1)
        with torch.no_grad():
            log_teacher = F.log_softmax(teacher_head(t).float() / temperature, dim=-1)
        # PyTorch's input is log Q and target is log P: this is KL(P_T || P_S).
        if direction == "reverse":
            # Keep both the student probabilities and log probabilities attached.
            return (log_student.exp() * (log_student - log_teacher)).sum()
        return F.kl_div(log_student, log_teacher, reduction="sum", log_target=True)

    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, count, chunk_tokens):
        s, t = student[start:start+chunk_tokens], teacher[start:start+chunk_tokens]
        if torch.is_grad_enabled() and s.requires_grad:
            value = checkpoint(chunk_loss, s, t, use_reentrant=False, preserve_rng_state=False)
        else:
            value = chunk_loss(s, t)
        total = total + value
    return total * (temperature**2 / count)
