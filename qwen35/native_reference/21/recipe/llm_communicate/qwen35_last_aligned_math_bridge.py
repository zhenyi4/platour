"""4B encoder -> 0.8B decoder using the teacher's last 24 block inputs."""
import torch

from direct_residual_chat_bridge import DirectResidualChatBridge
from hidden_bridge import bridge_layer_map
from qwen35_direct_math_bridge import accelerated_models
from qwen35_student import load_student


class LastAlignedResidualBridge(DirectResidualChatBridge):
    def __init__(self, student, teacher, *, lm_chunk_tokens=128):
        super().__init__(student, teacher, variant='teacher_residual',
                         initialization='zero', lm_chunk_tokens=lm_chunk_tokens)
        self.layer_map = bridge_layer_map(student.config.num_hidden_layers,
                                         teacher.config.num_hidden_layers, 'last_aligned')
        self.assignment = [[layer] for layer in self.layer_map]

    def metadata(self):
        return dict(super().metadata(), layer_mapping='last_aligned',
                    boundary='Not applicable: one teacher layer per student layer')


def load_bridge(root):
    runtime = accelerated_models()
    student = load_student(root / 'data/models/Qwen3.5-0.8B')
    teacher = load_student(root / 'data/models/Qwen3.5-4B')
    assert student.config.num_hidden_layers == 24
    assert teacher.config.num_hidden_layers == 32
    assert student.config.layer_types == teacher.config.layer_types[-24:]
    assert student.config.hidden_size == 1024 and teacher.config.hidden_size == 2560
    bridge = LastAlignedResidualBridge(student, teacher).eval()
    assert bridge.layer_map == list(range(8, 32))
    assert bridge.assignment == [[i] for i in range(8, 32)]
    assert sum(p.numel() for p in bridge.bridge_parameters()) == 62914560
    assert all(torch.count_nonzero(p) == 0 for p in bridge.bridge_parameters())
    runtime.update(teacher_layers=32, student_layers=24,
                   layer_mapping='last_aligned', layer_map=bridge.layer_map)
    return bridge, runtime


@torch.no_grad()
def cache_check(bridge, prompt):
    """Check trained bridge prefill versus teacher forcing and native generation."""
    ids = torch.tensor([prompt], device=bridge.student.device)
    mask = torch.ones_like(ids)
    teacher = bridge.teacher.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                  output_hidden_states=True).hidden_states
    features = bridge.teacher_features(teacher)
    inject = mask.bool()
    inject[:, -1] = False

    def prefill(tokens):
        with bridge.inject(features, inject):
            return bridge.student(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                  use_cache=True, logits_to_keep=1)

    a = prefill(ids)
    nxt = a.logits[:, -1].argmax(-1, keepdim=True)
    extended = torch.cat([ids, nxt], dim=1)
    cached = bridge.student(input_ids=nxt, attention_mask=torch.ones_like(extended),
                            past_key_values=a.past_key_values, use_cache=True, logits_to_keep=1)
    full = prefill(extended)
    error = (cached.logits.float() - full.logits.float()).abs()
    # Native generate and our decoder must use exactly the same cache semantics.
    fresh = prefill(ids)
    native = bridge.student.generate(input_ids=extended, attention_mask=torch.ones_like(extended),
        past_key_values=fresh.past_key_values, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_logits=True)
    native_error = (native.logits[0].float() - cached.logits[:, -1].float()).abs()
    report = dict(native_hybrid_cache=True, cache_type=type(cached.past_key_values).__name__,
        response_injection=False, last_prompt_token_injected=False,
        mean_abs_logit_error=float(error.mean()), max_abs_logit_error=float(error.max()),
        mean_abs_logit_tolerance=.1, max_abs_logit_tolerance=.5,
        precision_reference='Measured BF16 chunk/recurrent drift; FP32 diagnostic reduces mean error to 0.002–0.005',
        native_generate_max_abs_logit_error=float(native_error.max()),
        native_generate_greedy_match=bool(torch.equal(native.sequences[:, -1], cached.logits[:, -1].argmax(-1))))
    assert report['mean_abs_logit_error'] < .1 and report['max_abs_logit_error'] <= .5, report
    assert report['native_generate_greedy_match'] and report['native_generate_max_abs_logit_error'] < 1e-5, report
    return report
