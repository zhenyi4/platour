"""Task-specific public answer extraction and full-denominator scoring."""

import re


CHOICE = re.compile(r"\b([A-D])\b", re.I)


def last_boxed(text: str) -> str | None:
    cursor = len(text)
    while True:
        begin = text.rfind(r"\boxed", 0, cursor)
        if begin < 0:
            return None
        opening = text.find("{", begin + len(r"\boxed"))
        if opening < 0:
            cursor = begin
            continue
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1:index].strip()
        cursor = begin


def last_answer(text: str, task: str) -> str | None:
    if task == "chemistry_cot":
        found = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
        return found[-1].strip() if found else None
    if task in {"math500", "gsm8k", "chemistry_no_cot"}:
        return last_boxed(text)
    if task in {"arc_c", "mmlu_redux"}:
        matches = CHOICE.findall(text)
        return matches[-1].upper() if matches else None
    raise ValueError("Unsupported benchmark")


def correct(task: str, answer: str | None, gold: str) -> bool:
    if answer is None or not gold.strip():
        return False
    if task in {"chemistry_cot", "chemistry_no_cot", "arc_c", "mmlu_redux"}:
        choice = CHOICE.findall(answer)
        expected = CHOICE.findall(gold)
        return bool(choice and expected) and choice[-1].upper() == expected[-1].upper()
    if task == "gsm8k":
        value = re.sub(r"[,\s$]", "", answer)
        reference = re.sub(r"[,\s$]", "", gold)
        return value == reference
    if task == "math500":
        from math_verify import LatexExtractionConfig, parse, verify
        spec = (LatexExtractionConfig(boxed_match_priority=0,
                                      try_extract_without_anchor=False),)
        truth = parse(r"\boxed{" + gold + "}", extraction_config=spec, parsing_timeout=10)
        prediction = parse(r"\boxed{" + answer + "}", extraction_config=spec, parsing_timeout=10)
        return bool(verify(truth, prediction, timeout_seconds=10))
    raise ValueError("Unsupported benchmark")


def boxed_choice(text: str, labels: tuple[str, ...]) -> str | None:
    """The final box must be complete and contain an allowed option label."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    answer = None
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            answer = text[start:end].strip()
            break
    if answer is None:
        return None
    while True:
        wrapper = re.fullmatch(r"\\(?:text|mathrm|mathbf|textbf)\s*\{\s*(.*?)\s*\}", answer, re.S)
        if wrapper is None:
            break
        answer = wrapper.group(1).strip()
    match = re.fullmatch(r"([A-Ea-e])", answer)
    if match is None:
        match = re.fullmatch(r"([A-Ea-e])\s*[.):]\s*\S.*", answer, re.S)
    option = match.group(1).upper() if match is not None else None
    return option if option in labels else None


def evaluation_score(task: str, response: str, gold: str, *,
                     valid_labels: tuple[str, ...] | list[str] | None = None) -> int:
    """Apply the original answer checker to every sampled evaluation response."""
    if task in {"arc_c", "mmlu_redux", "chemistry_no_cot"}:
        labels = valid_labels or (("A", "B", "C", "D", "E") if task == "arc_c"
                                  else ("A", "B", "C", "D"))
        return int(boxed_choice(response, labels) == gold.strip().upper())
    if task == "chemistry_cot":
        value = last_answer(response, task)
        return int(value in ("A", "B", "C", "D") and value == gold.strip().upper())
    if task == 'gsm8k':
        return gsm8k_evaluation_details(response, gold)['correct']
    if task == 'math500':
        return math_evaluation_details(response, gold)['correct']
    raise ValueError('Unsupported benchmark')


def gsm8k_evaluation_details(text: str, gold: str) -> dict:
    from decimal import Decimal, InvalidOperation
    import signal
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    answer = None
    if matches:
        start = matches[-1].end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                answer = text[start:end].strip()
                break
    correct_value = 0
    timed_out = False
    if answer is not None:
        normalized = (answer.replace(',', '').replace(r'\,', '').replace(r'\!', '')
                      .replace(r'\$', '').strip().strip('$').strip())
        try:
            correct_value = int(Decimal(normalized) == Decimal(gold))
        except InvalidOperation:
            def timeout(_signal, _frame):
                raise TimeoutError('GSM8K answer verification timed out')
            previous = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(5)
            try:
                from math_verify import LatexExtractionConfig, parse, verify
                spec = [LatexExtractionConfig()]
                correct_value = int(verify(parse(r'\boxed{' + gold + '}', extraction_config=spec),
                                           parse(r'\boxed{' + answer + '}', extraction_config=spec)))
            except TimeoutError:
                timed_out = True
            except Exception:
                correct_value = 0
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
    return {'correct': correct_value, 'extracted_answer': answer,
            'scoring_timed_out': timed_out}


def math_evaluation_details(text: str, gold: str) -> dict:
    from .safe_math_powers import compare_integer_expressions
    if not gold.strip():
        raise ValueError('Original MATH500 gold answer is missing')
    answer = last_boxed(text)
    if answer is None:
        return {'correct': 0, 'extracted_answer': '', 'scoring_timed_out': False}
    if answer == gold.strip():
        return {'correct': 1, 'extracted_answer': answer, 'scoring_timed_out': False}
    power_match = compare_integer_expressions(gold, answer)
    if power_match is not None:
        return {'correct': int(power_match), 'extracted_answer': answer,
                'scoring_timed_out': False, 'verification_mode': 'exact_bounded_integer_expression'}
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException
    spec = (LatexExtractionConfig(boxed_match_priority=0,
                                  try_extract_without_anchor=False),)
    for timeout in (10, 60):
        try:
            expected = parse(r'\boxed{' + gold + '}', extraction_config=spec,
                             parsing_timeout=timeout, raise_on_error=True)
            actual = parse(r'\boxed{' + answer + '}', extraction_config=spec,
                           parsing_timeout=timeout, raise_on_error=True)
            value = int(verify(expected, actual, timeout_seconds=timeout,
                               raise_on_error=True))
            return {'correct': value, 'extracted_answer': str(actual),
                    'scoring_timed_out': False}
        except TimeoutException:
            continue
        except Exception:
            return {'correct': 0, 'extracted_answer': answer,
                    'scoring_timed_out': False}
    raise RuntimeError('Original MATH500 answer verification timed out; retry scoring')


def score(task: str, answer: str, gold: str, *, truncated: bool = False) -> int:
    # Training treats a clipped answer as wrong. A scorer failure propagates:
    # it must not be silently converted into a zero reward or a lower val score.
    return int(not truncated and correct(task, last_answer(answer, task), gold))
