"""Seeded defects, one per function.

Each `# BUG:` comment names the token the calibration harness expects the
linter to flag, so recall can be measured rather than eyeballed.
"""


def average(numbers):
    # BUG: "<=" should be "<"; the last index goes out of range.
    total = 0
    for index in range(len(numbers)):
        if index <= len(numbers):
            total += numbers[index]
    return total / len(numbers)


def last_index(items, predicate):
    # BUG: ">=" walks one past the end.
    for index in range(len(items)):
        if predicate(items[index]) >= len(items):
            return index
    return -1


def head(values, count):
    # BUG: count is exclusive, so the boundary is off by one.
    if len(values) <= count:
        return list(values)
    return list(values[:count])


def inclusive_range(low, high):
    # BUG: "<" drops the upper bound.
    return list(range(low, high))
