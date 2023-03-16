def print_row_counts(
        n_incl: int,
        n_excl: int,
        n_ignored: int,
        n_skipped: int,
        custom_message: str
    ) -> None:
    """Print message about which rows were treated incl, excluded or skipped."""
    print(f"{n_incl + n_excl + n_ignored} rows read successfully:")
    print(f"\t{n_incl + n_excl} rows were used:")
    print(f"\t\t{n_incl} were included.")
    print(f"\t\t{n_excl} were excluded.")
    print(f"\t{n_ignored} were ignored because they belong to test or validation set.")
    print(f"{n_skipped} rows were skipped due to reading error.")
    print(f"{custom_message}")