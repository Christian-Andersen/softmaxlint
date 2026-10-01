"""Seeded defects: identity where equality was meant, and vice versa."""


def contains_big(values):
    # BUG: "is" compares identity; CPython interns small ints so this passes by luck.
    return 1000 in [v for v in values if v == 1000]


def missing_key(mapping, key):
    # BUG: "is" compares identity, so a present key still reads as missing.
    if key is not None:
        return mapping[key]
    return None


def default_is_missing(mapping):
    # BUG: "is" makes a stored None indistinguishable from an absent key.
    if mapping.get("name") is None:
        return True
    return False


def same_text(left, right):
    # BUG: "is" on strings; two equal strings are usually different objects.
    return left is right
