import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def extract_numbers(text: str) -> set:
    """Numbers in the text, normalised so '1,000' and '1000' compare equal."""
    return {m.replace(",", "") for m in _NUMBER.findall(text or "")}


def numbers_match(query_a: str, query_b: str) -> bool:
    """Deterministic guard for near-misses like 'under $300' vs 'under $500' or '2024' vs '2025'.

    Embeddings rate these as near-identical and Jev is weak at numeric comparison, so code decides.
    Spelled-out numbers ('two') are not detected; a mismatch there falls through to the judge.
    """
    return extract_numbers(query_a) == extract_numbers(query_b)


_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NEGATIONS = {"not", "no", "never", "none", "nor", "without", "except", "excluding", "exclude", "cannot"}


def count_negations(text: str) -> int:
    """Negation cues in the text; contractions like "don't" or "isn't" count as one each."""
    return sum(1 for w in _WORD.findall((text or "").lower().replace("\u2019", "'")) if w in _NEGATIONS or w.endswith("n't"))


def negations_match(query_a: str, query_b: str) -> bool:
    """Cheap guard for flips like 'cake with gluten' vs 'cake without gluten'.

    Word-list based, so it misses phrasings like 'gluten-free' and flags harmless ones.
    A mismatch should route to the judge, not force a miss.
    """
    return count_negations(query_a) == count_negations(query_b)
