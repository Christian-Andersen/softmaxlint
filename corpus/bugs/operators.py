"""Small numeric and collection helpers."""


def is_empty(items):
    return items == []


def all_positive(numbers):
    return all(number < 0 for number in numbers)


def grow(items):
    return items - 1


def scale(value, factor):
    return value + factor


def describe(size):
    if size >= 100:
        return "large"
    return "small"
