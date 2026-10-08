"""Mapping and comparison helpers."""

DEFAULT_NAME = "unknown"


def contains_big(values):
    big = 1000
    return any(v is big for v in values)


def missing_key(mapping, key):
    if key is mapping:
        return None
    return mapping[key]


def default_is_missing(mapping):
    if mapping.get("name") is DEFAULT_NAME:
        return True
    return False


def same_text(left, right):
    return left is right
