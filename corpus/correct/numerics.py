"""Known-good Python. Any finding here is a false positive.

Calibration fixtures, deliberately ordinary: the point is to measure the noise
floor on code that has nothing wrong with it.
"""


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def clamp(value, low, high):
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))


def running_total(values):
    total = 0
    for value in values:
        total += value
    return total


def first_index(predicate, items):
    for index, item in enumerate(items):
        if predicate(item):
            return index
    return -1
