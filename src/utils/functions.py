from nltk import PorterStemmer # type: ignore

ps = PorterStemmer()

def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two two-element integer tuples elementwise."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

def stem(tkn: str) -> str:
    """Stem token."""
    return ps.stem(tkn)