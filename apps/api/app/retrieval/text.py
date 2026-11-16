"""Text normalization and Vietnamese-aware lexical tokenization.

Design notes
------------
* Unicode NFC is applied, but diacritics are **never** removed. ``mã`` and
  ``ma`` are different terms.
* The lexical analyzer is syllable based plus adjacent-syllable bigrams. This
  is deterministic and dependency-free; it is not full Vietnamese word
  segmentation (documented limitation).
* Alphanumeric identifiers (``ISO27001``, ``NĐ-30/2020``) are preserved as
  single normalized terms so exact terminology matching works.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

import tiktoken

_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

STOPWORDS = frozenset(
    {
        "và",
        "của",
        "là",
        "có",
        "cho",
        "trong",
        "với",
        "các",
        "những",
        "được",
        "một",
        "này",
        "đó",
        "theo",
        "tại",
        "từ",
        "để",
        "khi",
        "thì",
        "mà",
        "như",
        "hoặc",
        "nếu",
        "về",
        "ra",
        "vào",
        "lên",
        "xuống",
        "bị",
        "bởi",
        "do",
        "đã",
        "đang",
        "sẽ",
        "không",
        "cũng",
        "nhưng",
        "vì",
        "nên",
        "trên",
        "dưới",
        "giữa",
        "sau",
        "trước",
        "đến",
        "tới",
        "hơn",
        "rất",
        "quá",
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "are",
        "for",
        "with",
        "on",
        "at",
    }
)


def normalize_text(text: str) -> str:
    """Normalize without changing meaning or stripping Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    """Only for tolerant matching; never used to replace stored text."""
    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without).replace("đ", "d").replace("Đ", "D")


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # cl100k_base is a stable approximation for budgeting across providers.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split text by sentences, then words, never exceeding max_tokens.

    Used only for pathological single sentences; normal chunking happens on
    paragraph boundaries.
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = f"{current} {token}".strip()
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = token
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def tokenize(text: str) -> list[str]:
    """Return normalized lexical terms (unigrams, bigrams, identifiers)."""
    normalized = normalize_text(text).casefold()
    # Normalize the distinct Vietnamese letter đ but keep it (not an accent).
    words: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        raw = _WORD_RE.findall(sentence)
        words.extend(raw)
    terms: list[str] = []
    for index, word in enumerate(words):
        is_identifier = any(ch.isdigit() for ch in word) or len(word) >= 12
        if word not in STOPWORDS or is_identifier:
            terms.append(word[:120])
            if is_identifier and len(word) > 3:
                terms.append(f"id:{word[:116]}")
        if index + 1 < len(words):
            bigram = f"{word} {words[index + 1]}"
            if not (word in STOPWORDS and words[index + 1] in STOPWORDS):
                terms.append(bigram[:120])
    return terms


def term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def highlight_terms(text: str, terms: list[str]) -> list[str]:
    """Return terms present in text, useful for inspector explanations."""
    present = set(tokenize(text))
    return [term for term in terms if term in present]
