def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two two-element integer tuples elementwise."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

