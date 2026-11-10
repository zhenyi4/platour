"""Existing 397B FP8 smoke + repaired decoder/full-scale small-model proofs, then mandatory live startup checks."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def verify_evidence(original,protocol=None):
    from qwen35_multinode_config import digest
    original=Path(original)
    evidence={
        'fp8_smoke':original/'outputs/multinode_runs/397b_lcb_zero_b_v3/smoke_audit.json',
        'small_model':original/'outputs/multinode_runs/4b_bridge_scheduler_debug_v2/small_model_audit.json',
        'replay':original/'outputs/bridge_scheduler_debug/replay_validation_v2_result.json',
        'decoder_numerics':original/'outputs/bridge_scheduler_debug/restore_logprob_validation/result.json',
    }
    fp8,small,replay,numeric=(read(evidence[key]) for key in ('fp8_smoke','small_model','replay','decoder_numerics'))
    assert fp8['passed'] and fp8['structural']['world_size']==4
    assert fp8['structural']['replica_exact_weights'] and fp8['structural']['nonzero_gradient_updates']>0
    assert len(fp8['worst_cases'])==4 and all(p['passed'] and p['response_tokens']==16384 for p in fp8['worst_cases'])
    assert small['passed'] and small['model']=='4B->4B' and small['world_size']==4
    assert small['structural']['replica_exact_weights'] and small['structural']['singleton_tail_verified']
    assert small['val']['complete'] and small['val']['questions']==100 and small['val']['responses']==1600
    assert small['val']['samples_per_question']==16
    root=evidence['small_model'].parent
    assert all(read(root/f'node-{node}-finished.json')['passed'] for node in (0,1))
    assert replay['passed'] and replay['actual_400_requests'] and replay['restored_requests']>0 and replay['pending_after']==0
    assert numeric['passed'] and numeric['same_history_verified'] and numeric['resumed_tests']==8
    validated=read(original/'outputs/qwen35_multinode_debug/source_manifest.json')['files']
    connector='recipe/llm_communicate/qwen35_hybrid_memory_connector.py'
    assert digest(Path(__file__).with_name('qwen35_hybrid_memory_connector.py'))==validated[connector]
    small_protocol=read(root/'protocol.json')
    assert small_protocol['vllm_max_num_seqs']==16 and small_protocol['vllm_decode_batch_size']==0
    value=dict(paths={key:str(path) for key,path in evidence.items()},
        sha256={key:digest(path) for key,path in evidence.items()},
        components_passed=True,live_fp8_and_full_validation_required=True,
        previous_launcher_exit_codes_observed=False,
        procedure='Fresh current-code 16K FP8 resident stress on all ranks, then complete100x16 step0 validation before first optimizer update')
    if protocol is not None:
        assert protocol['teacher_key']=='397b' and protocol['dataset']=='lcb' and protocol['world_size']==4
        assert protocol['initialization']=='zero' and protocol['eval_samples']==16 and protocol['validation_before_training']
        assert protocol['vllm_max_num_seqs']==16 and protocol['vllm_decode_batch_size']==0
        assert protocol['qualification_evidence']==value,'Component evidence changed since preparation'
    return value


def verify_live(root,protocol,rank,initial_sha):
    from qwen35_multinode_config import ORIGINAL
    verify_evidence(ORIGINAL,protocol)
    root=Path(root)
    profiles=[read(root/f'worst_case_profile.rank-{i:02d}.json') for i in range(protocol['world_size'])]
    assert all(p['passed'] and p['vllm_resident'] and p['response_tokens']==16384 for p in profiles)
    assert all(read(root/f'fp8_hidden_audit.rank-{i:02d}.json')['passed'] for i in range(protocol['world_size']))
    summary=read(root/'evaluation/step-0000/summary.json')
    assert summary['questions']==100 and summary['responses']==1600 and summary['samples_per_question']==16
    assert summary['complete'] and summary['bridge_state_sha256']==initial_sha and summary['gold_missing']==0
    for i in range(protocol['world_size']):
        generated=read(root/'evaluation/step-0000'/f'rank-{i:02d}/responses.json')
        assert generated['complete'] and len(generated['rows'])==400
        assert all(row['bridge_state_sha256']==initial_sha for row in generated['rows'])
    return dict(passed=True,rank=rank,full_validation_responses=1600,initial_bridge_state_sha256=initial_sha,
        startup_stress_ranks=len(profiles),model='397B-FP8->4B',no_optimizer_updates_before_validation=True,
        qualification_evidence=protocol['qualification_evidence'])
