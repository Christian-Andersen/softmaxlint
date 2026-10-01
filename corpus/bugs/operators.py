"""Seeded defects: wrong operators and wrong operands."""


def is_empty(items):
    # BUG: "==" only matches by value; identity is what was written.
    return items == []


def all_positive(numbers):
    # BUG: "<" should be ">="; zero is not positive but this accepts it.
    return all(number < 0 for number in numbers)


def grow(items):
    # BUG: "-" subtracts where "+" is meant.
    return items - 1


def scale(value, factor):
    # BUG: "+" should be "*".
    return value + factor


def describe(size):
    # BUG: ">=" should be ">", so exact fits are mislabelled.
    if size >= 100:
        return "large"
    return "small"
