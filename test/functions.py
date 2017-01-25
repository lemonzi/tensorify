"""Python functions meant to be tensorified by the tests."""

def add(x, y, extra_one=False):
    return x + y + (1 if extra_one else 0)

def replicate(x, how_many):
    return [x] * how_many
