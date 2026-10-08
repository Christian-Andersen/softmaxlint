"""Index and range helpers."""


def average(numbers):
    total = 0
    for index in range(len(numbers)):
        if index <= len(numbers):
            total += numbers[index]
    return total / len(numbers)


def last_index(items, predicate):
    for index in range(len(items)):
        if predicate(items[index]) >= len(items):
            return index
    return -1


def head(values, count):
    if len(values) <= count:
        return list(values)
    return list(values[:count])


def inclusive_range(low, high):
    return list(range(low, high))
