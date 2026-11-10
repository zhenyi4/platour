"""Bounded exact evaluation of closed integer/factorial expressions, never partial answer matching."""
from contextlib import contextmanager
from fractions import Fraction
import math
import re

MAX_BITS = 32768
MAX_FACTORIAL = 4096
MAX_TERMS = 4096
SUM = re.compile(r'\\sum_\{([a-zA-Z])=(\d{1,4})\}\^\{(\d{1,4})\}(?:\{([a-zA-Z])!\}|([a-zA-Z])!)')


class Unsupported(ValueError):
    pass


class Parser:
    def __init__(self, source):
        self.source = source
        self.position = 0
        self.operations = 0

    def checked(self, value):
        self.operations += 1
        if self.operations > 16384 or max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
            raise Unsupported('Expression exceeds exact evaluator resource bounds')
        return value

    def group(self, opener='{', closer='}'):
        if self.source[self.position:self.position+1] != opener:
            raise Unsupported('Missing full arithmetic group')
        self.position += 1
        value = self.expression()
        if self.source[self.position:self.position+1] != closer:
            raise Unsupported('Unclosed arithmetic group')
        self.position += 1
        return value

    def atom(self):
        rest = self.source[self.position:]
        if rest.startswith('\\sum'):
            match = SUM.match(self.source, self.position)
            if match is None or match[1] != (match[4] or match[5]):
                raise Unsupported('Only closed finite factorial sums are supported')
            low, high = int(match[2]), int(match[3])
            if not 0 <= low <= high <= MAX_FACTORIAL or high - low + 1 > MAX_TERMS:
                raise Unsupported('Finite sum outside resource bounds')
            factorial = math.factorial(low)
            total = factorial
            for k in range(low + 1, high + 1):
                factorial *= k
                total += factorial
            self.position = match.end()
            if self.position < len(self.source) and self.source[self.position] not in (')', '}'):
                raise Unsupported('Ambiguous ungrouped summand boundary; use original parser')
            return self.checked(Fraction(total))
        if rest.startswith('\\frac'):
            self.position += len('\\frac')
            numerator = self.group()
            denominator = self.group()
            return self.checked(numerator / denominator)
        if rest[:1] == '(':
            return self.group('(', ')')
        if rest[:1] == '{':
            return self.group()
        match = re.match(r'\d{1,6}', rest)
        if match is None:
            raise Unsupported('Symbols, prose and unsupported operators are not accepted')
        self.position += len(match[0])
        return self.checked(Fraction(int(match[0])))

    def expression(self, minimum=0):
        rest = self.source[self.position:]
        if rest[:1] in ('+', '-'):
            sign = rest[0]
            self.position += 1
            value = self.expression(30)
            if sign == '-':
                value = -value
        else:
            value = self.atom()
        while self.position < len(self.source):
            operator = self.source[self.position]
            if operator == '!':
                if 50 < minimum:
                    break
                self.position += 1
                if value.denominator != 1 or not 0 <= value <= MAX_FACTORIAL:
                    raise Unsupported('Only bounded nonnegative integer factorials are supported')
                value = self.checked(Fraction(math.factorial(value.numerator)))
                continue
            precedence = {'+': 10, '-': 10, '*': 20, '/': 20, '^': 40}.get(operator)
            if precedence is None or precedence < minimum:
                break
            self.position += 1
            right = self.expression(precedence if operator == '^' else precedence + 1)
            if operator == '+':
                value += right
            elif operator == '-':
                value -= right
            elif operator == '*':
                value *= right
            elif operator == '/':
                value /= right
            else:
                if right.denominator != 1 or abs(right.numerator) > 16:
                    raise Unsupported('Unbounded exponent')
                value **= right.numerator
            value = self.checked(value)
        return value


def exact_factorial_value(latex):
    if not isinstance(latex, str) or '!' not in latex or len(latex) > 2000:
        return None
    source = re.sub(r'\s+', '', latex)
    source = source.replace(r'\left', '').replace(r'\right', '')
    source = source.replace(r'\,', '').replace(r'\;', '').replace(r'\cdot', '*').replace(r'\times', '*')
    try:
        parser = Parser(source)
        result = parser.expression()
        if parser.position != len(source):
            return None
        return result
    except (Unsupported, ZeroDivisionError, OverflowError, RecursionError):
        return None


@contextmanager
def exact_factorial_conversion():
    import math_verify.parser as parser
    import sympy
    original = parser.latex2sympy

    def bounded_exact(latex, *args, **kwargs):
        result = exact_factorial_value(latex)
        if result is not None:
            return sympy.Rational(result.numerator, result.denominator)
        return original(latex, *args, **kwargs)

    parser.parse_latex_cached.cache_clear()
    parser.extract_latex.cache_clear()
    parser.latex2sympy = bounded_exact
    try:
        yield
    finally:
        parser.latex2sympy = original
        parser.parse_latex_cached.cache_clear()
        parser.extract_latex.cache_clear()
