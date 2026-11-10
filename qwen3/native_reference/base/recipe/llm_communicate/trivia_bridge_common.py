"""TriviaQA short-answer prompt, official alias EM/F1, and run paths."""
from collections import Counter
import re
import string

from arc_bridge_sft_common import ROOT, HERE, PYTHON, GREEDY, environment, save, immutable, read, jsonl, sha

BASE = ROOT / 'outputs/llm_communicate/triviaqa_answer_sentence_short10k_sft_ep3_bs64'
MODELS = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B']
INSTRUCTION = ('Reply with exactly "The answer is \\boxed{ANSWER}", replacing ANSWER with '
               'a short answer (a word or brief phrase, such as a name, place, or date). '
               'Do not include any explanation or any other text.')


def normalize_answer(s):
    # Verbatim algorithm from pinned TriviaQA official evaluation.py.
    s = s.replace('_', ' ').lower()
    exclude = set(string.punctuation + '\u2018\u2019\u00b4`')
    s = ''.join(ch if ch not in exclude else ' ' for ch in s)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split()).strip()


def extract_answer(text):
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None, 'missing_boxed'
    start = matches[-1].end()
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            value = text[start:end].strip()
            while True:
                wrapper = re.fullmatch(r'\\(?:text|mathrm|mathbf|textbf)\s*\{\s*(.*?)\s*\}', value, re.DOTALL)
                if wrapper is None:
                    break
                value = wrapper.group(1).strip()
            return (value, 'valid') if value else (None, 'invalid_boxed')
    return None, 'invalid_boxed'


def score(row):
    answer, status = extract_answer(row['response'])
    golds = row['normalized_aliases']
    assert golds
    pred = normalize_answer(answer) if answer is not None else ''
    em = int(status == 'valid' and pred in golds)
    def f1(gold):
        common = sum((Counter(pred.split()) & Counter(normalize_answer(gold).split())).values())
        return 2 * common / (len(pred.split()) + len(normalize_answer(gold).split())) if common else 0.
    f1_value = max(map(f1, golds)) if status == 'valid' else 0.
    exact = re.fullmatch(r'The answer is \\boxed\{[^{}\n]+\}', row['response'].strip())
    return dict(row, extracted_answer=answer, extraction_status=status, acc=em, f1=f1_value,
                response_format='answer_sentence_short' if exact else ('prose_and_boxed' if status == 'valid' else status))


def summarize(rows):
    n = len(rows)
    return dict(questions=n, responses=n, correct=sum(r['acc'] for r in rows),
        accuracy=sum(r['acc'] for r in rows)/n, f1=sum(r['f1'] for r in rows)/n,
        mean_tokens=sum(r['tokens'] for r in rows)/n,
        clipped=sum(r['finish_reason']=='length' for r in rows),
        extraction_status_counts=dict(Counter(r['extraction_status'] for r in rows)),
        response_formats=dict(Counter(r['response_format'] for r in rows)))
