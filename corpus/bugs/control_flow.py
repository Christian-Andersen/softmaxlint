"""Seeded defects: control flow that does not do what the name says."""


def find_first(items, predicate):
    # BUG: the guard never rejects anything.
    for item in items:
        if predicate(item):
            return item
    return None


def only_even(numbers):
    # BUG: "continue" is unconditional, so the append is unreachable.
    result = []
    for number in numbers:
        continue
        result.append(number)
    return result


def safe_divide(a, b):
    # BUG: "except" catches nothing because the raise is outside the try.
    try:
        return a / b
    except ZeroDivisionError:
        raise ZeroDivisionError("division by zero") from None
    raise ZeroDivisionError("unreachable")


def earliest(values):
    # BUG: ">" should be "<", so this returns the largest value.
    best = values[0]
    for value in values:
        best = max(best, value)
    return best
