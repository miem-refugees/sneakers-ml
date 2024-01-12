def most_common(lst: list):
    return max(set(lst), key=lst.count)
