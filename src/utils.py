def to_int(x):
    try:
        return int(x)
    except ValueError:
        return -1