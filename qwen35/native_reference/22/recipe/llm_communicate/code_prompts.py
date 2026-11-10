"""Prompt rows for the code tasks (LCB train/validation/test, HumanEval, MBPP full test), non-thinking chat format.

LCB: official LiveCodeBench generic prompt (SYSTEM_MESSAGE_GENERIC + get_generic_question_template_answer), as in
     outputs/lcb_eval (pinned LiveCodeBench commit).
HumanEval / MBPP: 0-shot instruction prompt (EvalPlus instruction wording); MBPP shows the problem text and its three
     test asserts (MBPP full-test convention), HumanEval shows the function signature + docstring.
Every row carries a `judge` spec consumed by code_judge.py; no test data is embedded for LCB (it is mounted).
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
LCB = ROOT/'data/livecodebench'
BENCH = ROOT/'data/code_bench'
INSTRUCTION = 'Please provide a self-contained Python script that solves the following problem in a markdown code block:'
MAX_PROMPT_TOKENS = 2048


def encode(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    assert tokenizer.decode(ids).endswith('<think>\n\n</think>\n\n')
    return list(ids)


def lcb_problems():
    return {p['key']: p for p in map(json.loads, (LCB/'problems.jsonl').open())}


_OFFICIAL = {}


def official_prompt_module():
    """lcb_runner/prompts/code_generation.py of the pinned commit, loaded by file path: the package __init__ imports
    unrelated modules (anthropic, ...) and the module reads few-shot files relative to the working directory."""
    if 'module' not in _OFFICIAL:
        import importlib.util
        import os
        sys.path.insert(0, str(ROOT/'third_party/livecodebench'))
        path = ROOT/'third_party/livecodebench/lcb_runner/prompts/code_generation.py'
        spec = importlib.util.spec_from_file_location('lcb_official_code_generation_prompts', path)
        module = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        os.chdir(ROOT/'third_party/livecodebench')
        try:
            spec.loader.exec_module(module)
        finally:
            os.chdir(cwd)
        _OFFICIAL['module'] = module
    return _OFFICIAL['module']


def prompt_constants():
    return official_prompt_module().PromptConstants


def lcb_messages(problem):
    PromptConstants = prompt_constants()
    # Same text as lcb_runner.prompts.code_generation.get_generic_question_template_answer (checked in tests).
    prompt = f"### Question:\n{problem['question_content']}\n\n"
    if problem['starter_code']:
        prompt += f'### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'
        prompt += f"```python\n{problem['starter_code']}\n```\n\n"
    else:
        prompt += f'### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'
        prompt += '```python\n# YOUR CODE HERE\n```\n\n'
    prompt += '### Answer: (use the provided format with backticks)\n\n'
    return [dict(role='system', content=PromptConstants.SYSTEM_MESSAGE_GENERIC), dict(role='user', content=prompt)]


def lcb_rows(keys, tokenizer, problems=None):
    problems = problems or lcb_problems()
    rows = []
    for index, key in enumerate(keys):
        p = problems[key]
        messages = lcb_messages(p)
        rows.append(dict(index=index, key=key, platform=p['platform'], question_id=p['question_id'],
            difficulty=p['difficulty'], contest_date=p['contest_date'], messages=messages, gold='',
            judge=dict(kind='lcb', key=key, tests=p['tests']), prompt_token_ids=encode(tokenizer, messages)))
    return rows


def humaneval_rows(tokenizer):
    rows = []
    for index, r in enumerate(map(json.loads, (BENCH/'humaneval.jsonl').open())):
        messages = [dict(role='user', content=f"{INSTRUCTION}\n```python\n{r['prompt'].strip()}\n```")]
        imports = [line for line in r['prompt'].splitlines() if re.match(r'^(import |from \S+ import )', line)]
        rows.append(dict(index=index, task_id=r['task_id'], messages=messages, gold='',
            judge=dict(kind='humaneval', prompt_imports=imports, test=r['test'], entry_point=r['entry_point']),
            prompt_token_ids=encode(tokenizer, messages)))
    return rows


def mbpp_rows(tokenizer):
    rows = []
    for index, r in enumerate(map(json.loads, (BENCH/'mbpp_full_test.jsonl').open())):
        tests = '\n'.join(r['test_list'])
        messages = [dict(role='user', content=f'{INSTRUCTION}\n```python\n"""\n{r["text"].strip()}\n{tests}\n"""\n```')]
        rows.append(dict(index=index, task_id=r['task_id'], messages=messages, gold='',
            judge=dict(kind='mbpp', test_setup_code=r['test_setup_code'], test_list=r['test_list']),
            prompt_token_ids=encode(tokenizer, messages)))
    return rows
