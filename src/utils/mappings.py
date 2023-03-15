# Column headers and indexes for input csv
incl_cols = {
    "id": 1,
    "domain": 2,
    "type": 3,
    "url": 4,
    "content": 5,
    "scraped_at": 6,
    "title": 9,
    "authors": 10,
    "keywords": 11,
    "tags": 14,
    "summary": 15,
}

incl_inds = [ind for ind in incl_cols.values()]
incl_keys = [key for key in incl_cols.keys()]

# Column headers for output csv
out_cols = [
    # Transfered
    "id",
    "domain",
    "type",
    "url",
    "scraped_at",
    "title",
    "authors",
    "keywords",
    "tags",
    "summary",
    # Derived
    "shortened",
    "content_len",
    "mean_word_len",
    "median_word_len"
]

# Label types to disregard
excl_types = {
    "satire",
    "unknown",
    "",
    "unreliable",
    "clickbait",
    "conspiracy",
    "bias",
    "hate",
    "junksci",
    "political",
    "rumor"
}

# Store columns that transfer unchanged from input to output csv's
transfered_cols: list[str] = []
for col_name in incl_keys:
    if col_name in out_cols:
        transfered_cols.append(col_name)
