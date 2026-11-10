"""Restore the pinned DAPO split and tokenize its unchanged Answer: prompts."""
import hashlib
import json
from pathlib import Path

from qwen35_bridge_opd_core import ROOT, save


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_splits(directory=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from prepare_dapo_math_dedup import key_for

    directory = Path(directory or ROOT / 'data/dapo_math_17k')
    metadata = json.loads((directory / 'summary.json').read_text())
    manifest_path = directory / 'split_manifest.jsonl'
    assert digest(manifest_path) == metadata['files_sha256'][manifest_path.name]
    with manifest_path.open() as stream:
        manifest = [json.loads(line) for line in stream if line.strip()]
    selected = {split: [r for r in manifest if r['split'] == split]
                for split in ('train', 'validation')}
    assert [len(selected[s]) for s in selected] == [metadata['train_rows'], metadata['validation_rows']]
    hashes = {s: {r['question_sha256'] for r in rows} for s, rows in selected.items()}
    assert all(len(hashes[s]) == len(selected[s]) for s in selected)
    assert not hashes['train'] & hashes['validation']
    missing = [s for s in selected if not (directory / f'{s}.parquet').exists()]
    if missing:
        source = directory / 'raw/data/dapo-math-17k.parquet'
        assert digest(source) == metadata['source_sha256']
        parquet = pq.ParquetFile(source)
        assert parquet.metadata.num_rows == metadata['source_rows']
        wanted = {r['first_source_row']: r for s in missing for r in selected[s]}
        assert len(wanted) == sum(len(selected[s]) for s in missing)
        found, offset = {}, 0
        for batch in parquet.iter_batches(batch_size=8192):
            for index, row in enumerate(batch.to_pylist(), offset):
                if index in wanted:
                    item = wanted[index]
                    assert hashlib.sha256(key_for(row).encode()).hexdigest() == item['question_sha256']
                    assert row['extra_info']['index'] == item['source_id']
                    found[index] = row
            offset += batch.num_rows
            if len(found) == len(wanted):
                break
        assert len(found) == len(wanted)
        for split in missing:
            rows = [found[r['first_source_row']] for r in selected[split]]
            target = directory / f'{split}.parquet'
            temporary = target.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow.remove_metadata()), temporary)
            assert digest(temporary) == metadata['files_sha256'][target.name]
            temporary.replace(target)
    result = {}
    for split in selected:
        path = directory / f'{split}.parquet'
        assert digest(path) == metadata['files_sha256'][path.name]
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == len(selected[split])
        for row, item in zip(rows, selected[split], strict=True):
            assert hashlib.sha256(key_for(row).encode()).hexdigest() == item['question_sha256']
            assert row['extra_info']['index'] == item['source_id']
            assert str(row['reward_model']['ground_truth']).strip()
        result[split] = rows
    return result, metadata


def tokenized_splits(base, tokenizer, teacher_tokenizer):
    splits, metadata = restore_splits()
    result = {}
    for split, rows in splits.items():
        encoded = []
        for index, row in enumerate(rows):
            messages = row['prompt']
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            teacher_ids = teacher_tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if hasattr(ids, 'keys'):
                ids = ids['input_ids']
            if hasattr(teacher_ids, 'keys'):
                teacher_ids = teacher_ids['input_ids']
            assert ids == teacher_ids and 2 <= len(ids) <= 2048
            assert all(isinstance(token, int) for token in ids)
            encoded.append(dict(index=index, source_id=row['extra_info']['index'],
                messages=messages, question=messages[-1]['content'],
                input_text=messages[-1]['content'], gold=str(row['reward_model']['ground_truth']),
                prompt_token_ids=ids))
        result[split] = encoded
    assert not {tuple(r['prompt_token_ids']) for r in result['train']} & {
        tuple(r['prompt_token_ids']) for r in result['validation']}
    save(base / 'dataset.json', dict(source=metadata['source'],
        split='Existing deduplicated seed-42 train/validation; validation used as DAPO test set',
        train_questions=len(result['train']), eval_questions=len(result['validation']),
        train_validation_overlap=0, source_files_sha256=metadata['files_sha256'],
        prompt_modified=False, prompt_format='Original DAPO Answer: instruction',
        maximum_train_prompt_tokens=max(len(r['prompt_token_ids']) for r in result['train']),
        maximum_eval_prompt_tokens=max(len(r['prompt_token_ids']) for r in result['validation'])))
    return result['train'], result['validation']
