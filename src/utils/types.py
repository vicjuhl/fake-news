news_info  = tuple[str, str] # type and content
words_info = tuple[str, list[str]] # type and words
doc_total_freqs = tuple[int, int] # n docs with occurence, n total occurrances
words_dict = dict[str, dict[str, doc_total_freqs]] # words and {type: count}