"""Deduplicate the pinned official DAPO parquet and create a seeded holdout."""
import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key_for(row):
    # Preserve internal whitespace and math notation; normalize line endings only.
    prompt = [dict(role=m['role'], content=unicodedata.normalize(
        'NFC', m['content'].replace('\r\n', '\n').replace('\r', '\n')).strip())
        for m in row['prompt']]
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/dapo_math_17k'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-size', type=int, default=500)
    args = parser.parse_args()
    source = args.root / 'raw/data/dapo-math-17k.parquet'
    parquet = pq.ParquetFile(source)
    records, counts, first_rows, conflicts = {}, {}, {}, {}
    row_number = 0
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            key = key_for(row)
            if key not in records:
                records[key] = row
                counts[key] = 0
                first_rows[key] = row_number
            elif records[key]['reward_model'] != row['reward_model']:
                variants = conflicts.setdefault(key, {})
                for variant in (records[key], row):
                    variants[json.dumps(variant['reward_model'], sort_keys=True)] = variant
            counts[key] += 1
            row_number += 1
    conflict_path = args.root / 'excluded_conflicts.jsonl'
    conflict_path.write_text(''.join(json.dumps(dict(
        question_sha256=hashlib.sha256(k.encode()).hexdigest(),
        source_occurrences=counts[k], variants=list(variants.values())), ensure_ascii=False) + '\n'
        for k, variants in sorted(conflicts.items())))
    keys = sorted(set(records) - set(conflicts))
    assert 0 < args.validation_size < len(keys)
    validation = set(random.Random(args.seed).sample(keys, args.validation_size))
    groups = {'train': [k for k in keys if k not in validation],
              'validation': [k for k in keys if k in validation]}
    manifest = []
    files = {conflict_path.name: digest(conflict_path)}
    for split, selected in groups.items():
        rows = [records[k] for k in selected]
        path = args.root / f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow.remove_metadata()), path)
        files[path.name] = digest(path)
        for key in selected:
            manifest.append(dict(question_sha256=hashlib.sha256(key.encode()).hexdigest(),
                                 source_id=records[key]['extra_info']['index'],
                                 first_source_row=first_rows[key],
                                 source_occurrences=counts[key], split=split))
    path = args.root / 'split_manifest.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest))
    files[path.name] = digest(path)
    # Validate the saved artifacts, not just the in-memory split.
    saved = {s: pq.read_table(args.root / f'{s}.parquet').to_pylist() for s in groups}
    saved_keys = {s: {key_for(r) for r in rows} for s, rows in saved.items()}
    assert len(saved['validation']) == args.validation_size
    assert not saved_keys['train'] & saved_keys['validation']
    assert saved_keys['train'] | saved_keys['validation'] == set(keys)
    assert all(len(saved_keys[s]) == len(saved[s]) for s in groups)
    assert sum(r['source_occurrences'] for r in manifest) + sum(counts[k] for k in conflicts) == row_number
    summary = dict(source=json.loads((args.root / 'raw/source.json').read_text()),
                   source_file='data/dapo-math-17k.parquet', source_sha256=digest(source),
                   source_rows=row_number, unique_questions_before_conflict_filter=len(records),
                   unique_questions=len(keys), removed_duplicate_rows=row_number - len(records),
                   excluded_conflicting_questions=len(conflicts),
                   excluded_conflicting_source_rows=sum(counts[k] for k in conflicts),
                   seed=args.seed, train_rows=len(saved['train']),
                   validation_rows=len(saved['validation']), train_validation_overlap=0,
                   deduplication='Full prompt messages (role + content), NFC, LF line endings, outer strip; no semantic deduplication',
                   split_algorithm='Sort canonical prompt JSON strings; Python random.Random(seed).sample selects validation; retain sorted order within splits',
                   original_fields_preserved=True, files_sha256=files)
    (args.root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
