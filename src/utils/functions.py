import pandas as pd

def add_tuples(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Add two two-element integer tuples elementwise."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

def to_binary(x: str) -> int:
    if x == 'fake':
        return -1
    elif x == 'reliable' or x == 'real':
        return 1
    
def df_type_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the 'type' column of a dataframe to a binary column."""
    df['type_binary'] = df['type'].apply(lambda x: to_binary(x))
    return df

