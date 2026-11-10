"""Versioned whole-final-answer verification; never search the reasoning for a matching number."""
import os
import re

SCORING_VERSION = 'answer_math_v2'
TIMEOUT = int(os.environ.get('MATH_ANSWER_TIMEOUT_S', '5'))


def normalize_math_answer(answer):
    value = str(answer).strip()
    value = re.sub(r'^Answer\s*:\s*', '', value, count=1, flags=re.IGNORECASE).strip()
    if value.endswith('.') and not value.endswith(r'\.'):
        value = value[:-1].rstrip()
    for left, right in [('$$', '$$'), (r'\[', r'\]'), (r'\(', r'\)'), ('$', '$')]:
        interior = value[len(left):-len(right)]
        if (value.startswith(left) and value.endswith(right) and len(value) > len(left)+len(right)
                and left not in interior and right not in interior):
            value = interior.strip()
            break
    for command in (r'\boxed{', r'\fbox{'):
        if value.startswith(command):
            depth = 1
            closing = None
            for position in range(len(command), len(value)):
                if value[position] == '{':
                    depth += 1
                elif value[position] == '}':
                    depth -= 1
                    if depth == 0:
                        closing = position
                        break
            if closing == len(value)-1:
                value = value[len(command):closing].strip()
    value = value.replace('<=', r'\le ').replace('>=', r'\ge ').replace('!=', r'\ne ')
    return value


def score_final_answer(gold, answer, timeout_seconds=None):
    """One canonical parse per gold/answer, with a timeout and whole-expression comparison."""
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from math_verify.parser import LatexExtractionConfig
    timeout = TIMEOUT if timeout_seconds is None else int(timeout_seconds)
    if timeout <= 0:
        raise ValueError('Final-answer timeout must be positive')
    normalized = normalize_math_answer(answer)
    reference = normalize_math_answer(gold)
    result = dict(acc=0, score=0., scoring_timed_out=False, answer_parse_failed=False,
                  gold_parse_failed=False, normalized_answer=normalized, scoring_version=SCORING_VERSION)
    if not normalized or not reference:
        result['answer_parse_failed'] = not normalized
        result['gold_parse_failed'] = not reference
        return result
    config = [LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0)]
    try:
        expected = parse(r'\boxed{'+reference+'}', extraction_config=config,
                         fallback_mode='no_fallback', extraction_mode='first_match',
                         parsing_timeout=timeout, raise_on_error=True)
        actual = parse(r'\boxed{'+normalized+'}', extraction_config=config,
                       fallback_mode='no_fallback', extraction_mode='first_match',
                       parsing_timeout=timeout, raise_on_error=True)
        result.update(answer_parse_failed=not bool(actual), gold_parse_failed=not bool(expected))
        if actual and expected:
            result['acc'] = int(verify(expected, actual, strict=True, timeout_seconds=timeout,
                                       raise_on_error=True))
            result['verification_mode'] = 'symbolic'
        else:
            result['acc'] = int(re.sub(r'\s+', '', normalized) == re.sub(r'\s+', '', reference))
            result['verification_mode'] = 'whole_answer_exact_only'
        result['score'] = float(result['acc'])
    except (TimeoutError, TimeoutException) as error:
        result.update(scoring_timed_out=True,timeout_error=type(error).__name__,acc=0,score=0.)
    return result
