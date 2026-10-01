"""Known-good Python: collections and string handling.

Calibration fixtures. Anything flagged in here is a false positive.
"""


def counts_by_key(items, key):
    counts = {}
    for item in items:
        name = key(item)
        counts[name] = counts.get(name, 0) + 1
    return counts


def group_by(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def invert(mapping):
    inverted = {}
    for key, value in mapping.items():
        inverted[value] = key
    return inverted


def merge_unique(*sequences):
    seen = set()
    merged = []
    for sequence in sequences:
        for item in sequence:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def normalise_whitespace(text):
    return " ".join(text.split())


def truncate(text, width):
    if width <= 0:
        raise ValueError("width must be positive")
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def split_lines(text):
    lines = []
    current = []
    for char in text:
        current.append(char)
        if char == "\n":
            lines.append("".join(current))
            current = []
    if current:
        lines.append("".join(current))
    return lines
