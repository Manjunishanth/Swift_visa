# chunking.py
from nltk.tokenize import sent_tokenize
import re

def _fallback_sent_tokenize(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def sentence_chunking(text, max_tokens=500):
    """
    Split text into chunks of at most `max_tokens` (words) each.
    Returns list of chunks. Each chunk is a string.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = _fallback_sent_tokenize(text)

    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        tokens = sentence.split()
        # If a single sentence exceeds max_tokens, split it by words
        if len(tokens) > max_tokens:
            # flush any current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                token_count = 0
            # split the long sentence into smaller windows
            for i in range(0, len(tokens), max_tokens):
                slice_tokens = tokens[i:i + max_tokens]
                chunks.append(" ".join(slice_tokens))
            continue

        if token_count + len(tokens) > max_tokens:
            # flush current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            token_count = 0

        current_chunk.append(sentence)
        token_count += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
