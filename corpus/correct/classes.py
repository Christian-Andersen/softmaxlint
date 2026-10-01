"""Known-good Python: classes and errors.

Calibration fixtures. Anything flagged in here is a false positive.
"""


class Counter:
    def __init__(self, start=0):
        self._counts = {}
        self._total = start

    def add(self, amount=1):
        self._total += amount
        return self

    @property
    def total(self):
        return self._total

    def most_common(self, limit=None):
        pairs = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))
        if limit is None:
            return pairs
        return pairs[:limit]

    def record(self, key):
        self._counts[key] = self._counts.get(key, 0) + 1
        return self

    def reset(self):
        self._counts.clear()
        self._total = 0
        return self


class ValidationError(Exception):
    def __init__(self, field, message):
        super().__init__(message)
        self.field = field
        self.message = message


def require(condition, field, message):
    if not condition:
        raise ValidationError(field, message)


def parse_port(value):
    require(isinstance(value, str), "port", "port must be a string")
    port = int(value, 10)
    require(1 <= port <= 65535, "port", "port out of range")
    return port
