"""Bounded same-base integer-power comparison from the fixed reproduction scorer."""

import ast
import re

MAX_BITS = 4096


def _tree(text):
    if len(text) > 2048:
        raise ValueError('Expression too long')
    text = text.replace(r'\left', '').replace(r'\right', '')
    if not re.fullmatch(r'[0-9\s{}()+*^\-]+', text):
        raise ValueError('Unsupported syntax')
    text = text.replace('{', '(').replace('}', ')').replace('^', '**').strip()
    text = re.sub(r'(?<=[0-9)])\s*(?=\()', '*', text)
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError('Expression too complex')
    return tree.body


def _integer(node, depth=0):
    if depth > 24:
        raise ValueError('Expression too deep')
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _integer(node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            value = -value
    elif isinstance(node, ast.BinOp):
        a, b = _integer(node.left, depth + 1), _integer(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            value = a + b
        elif isinstance(node.op, ast.Sub):
            value = a - b
        elif isinstance(node.op, ast.Mult):
            if a.bit_length() + b.bit_length() > MAX_BITS:
                raise ValueError('Integer multiplication too large')
            value = a * b
        elif isinstance(node.op, ast.Pow):
            if b < 0 or b > MAX_BITS or max(1, a.bit_length()) * b > MAX_BITS:
                raise ValueError('Integer power too large')
            value = a ** b
        else:
            raise ValueError('Unsupported operator')
    else:
        raise ValueError('Unsupported expression')
    if value.bit_length() > MAX_BITS:
        raise ValueError('Integer too large')
    return value


def compare_integer_powers(gold, predicted):
    """Return a proven bool for b**m versus b**n, b>=2 and m,n>=0."""
    try:
        a, b = _tree(gold), _tree(predicted)
        if not all(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in (a, b)):
            return None
        base_a, base_b = _integer(a.left), _integer(b.left)
        if base_a != base_b or base_a < 2:
            return None
        exponent_a, exponent_b = _integer(a.right), _integer(b.right)
        if min(exponent_a, exponent_b) < 0:
            return None
        return exponent_a == exponent_b
    except (ValueError, SyntaxError, RecursionError, OverflowError):
        return None


def _bounded_nonnegative(node, cap, depth=0):
    if depth > 24:
        raise ValueError('Expression too deep')
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
        return min(node.value, cap)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _bounded_nonnegative(node.operand, cap, depth + 1)
    if not isinstance(node, ast.BinOp):
        raise ValueError('Unsupported expression')
    a = _bounded_nonnegative(node.left, cap, depth + 1)
    b = _bounded_nonnegative(node.right, cap, depth + 1)
    if isinstance(node.op, ast.Add):
        return min(cap, a + b)
    if isinstance(node.op, ast.Mult):
        return min(cap, a * b)
    if isinstance(node.op, ast.Sub):
        if a == cap or b == cap or a < b:
            raise ValueError('Subtraction cannot be proved from saturated bounds')
        return a - b
    if isinstance(node.op, ast.Pow):
        if b == 0:
            if a == 0:
                raise ValueError('Indeterminate zero power')
            return 1
        if a <= 1:
            return a
        if a == cap or b == cap:
            return cap
        result = 1
        while b:
            if b & 1:
                result = min(cap, result * a)
                if result == cap:
                    return cap
            b >>= 1
            if b:
                a = min(cap, a * a)
        return result
    raise ValueError('Unsupported operator')


def compare_integer_expressions(gold, predicted):
    match = compare_integer_powers(gold, predicted)
    if match is not None:
        return match
    try:
        expected = _integer(_tree(gold))
        if expected < 0 or expected.bit_length() > 256:
            return None
        cap = max(4, expected + 1)
        actual = _bounded_nonnegative(_tree(predicted), cap)
        return actual == expected
    except (ValueError, SyntaxError, RecursionError, OverflowError):
        return None
