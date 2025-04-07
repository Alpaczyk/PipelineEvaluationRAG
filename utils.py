import re
from fuzzywuzzy import process, fuzz

def find_query_despite_whitespace(document, query):

    normalized_query = re.sub(r'\s+', ' ', query).strip()

    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())

    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None


def rigorous_document_search(document: str, target: str):

    if target.endswith('.'):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    sentences = re.split(r'[.!?]\s*|\n', document)

    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None

    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index
