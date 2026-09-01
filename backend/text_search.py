from __future__ import annotations

import re
import unicodedata


MAX_TEXT_QUERY_CHARS = 256
MAX_TEXT_SEARCH_TERMS = 24
MAX_ANY_TEXT_SEARCH_TERMS = 8
MAX_TEXT_PREFIX_TERMS = 4
MIN_TEXT_PREFIX_CHARS = 3
TEXT_SEARCH_MODES = frozenset({"phrase", "all_terms", "any_terms"})
DEFAULT_INNODB_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "are",
        "as",
        "at",
        "be",
        "by",
        "com",
        "de",
        "en",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "la",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "und",
        "was",
        "what",
        "when",
        "where",
        "who",
        "will",
        "with",
        "www",
    }
)
_TEXT_SEARCH_TERM_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*(?:\*)?", re.UNICODE)


def _normalized_query(text_query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text_query).casefold().split())


def parse_text_search_terms(text_query: str) -> list[str]:
    normalized_query = _normalized_query(text_query)
    if not normalized_query:
        raise ValueError("text_query must contain searchable text.")
    if len(normalized_query) > MAX_TEXT_QUERY_CHARS:
        raise ValueError(
            f"text_query must be at most {MAX_TEXT_QUERY_CHARS} characters."
        )

    for index, char in enumerate(normalized_query):
        if char != "*":
            continue
        previous = normalized_query[index - 1] if index else ""
        following = (
            normalized_query[index + 1]
            if index + 1 < len(normalized_query)
            else ""
        )
        if not previous.isalnum() or (
            following and (following.isalnum() or following in "_'’*")
        ):
            raise ValueError(
                "Only a single trailing `*` on a word is supported for prefix matching."
            )

    terms = _TEXT_SEARCH_TERM_RE.findall(normalized_query)
    if not terms:
        raise ValueError("text_query must contain at least one letter or number.")
    if len(terms) > MAX_TEXT_SEARCH_TERMS:
        raise ValueError(
            f"text_query may contain at most {MAX_TEXT_SEARCH_TERMS} terms."
        )

    prefix_terms = [term for term in terms if term.endswith("*")]
    if len(prefix_terms) > MAX_TEXT_PREFIX_TERMS:
        raise ValueError(
            f"text_query may contain at most {MAX_TEXT_PREFIX_TERMS} prefix terms."
        )
    if any(len(term[:-1]) < MIN_TEXT_PREFIX_CHARS for term in prefix_terms):
        raise ValueError(
            f"Prefix terms must contain at least {MIN_TEXT_PREFIX_CHARS} characters before `*`."
        )
    return terms


def validate_text_query_syntax(text_query: str) -> None:
    _ = parse_text_search_terms(text_query)


def compile_boolean_text_query(text_query: str, match_mode: str) -> str:
    """Compile user text into bounded MariaDB BOOLEAN MODE candidate syntax."""
    if match_mode not in TEXT_SEARCH_MODES:
        raise ValueError(
            "text_match_mode must be one of: phrase, all_terms, any_terms."
        )
    terms = parse_text_search_terms(text_query)
    if match_mode == "any_terms" and len(terms) > MAX_ANY_TEXT_SEARCH_TERMS:
        raise ValueError(
            f"any_terms queries may contain at most {MAX_ANY_TEXT_SEARCH_TERMS} terms."
        )
    if match_mode == "phrase":
        candidate_terms = [
            term
            for term in terms
            if term.endswith("*")
            or (
                len(term) >= MIN_TEXT_PREFIX_CHARS
                and term not in DEFAULT_INNODB_STOPWORDS
            )
        ]
        if not candidate_terms:
            raise ValueError(
                "Phrase queries must contain at least one indexed word or word prefix."
            )
        # Native InnoDB phrase evaluation is pathologically slow for common legal
        # phrases and silently mishandles prefix terms. Use FULLTEXT only to narrow
        # candidates; build_section_text_match_expression applies the exact ordered
        # token post-filter.
        return " ".join(f"+{term}" for term in dict.fromkeys(candidate_terms))

    unindexed_terms = [
        term
        for term in terms
        if not term.endswith("*")
        and (
            len(term) < MIN_TEXT_PREFIX_CHARS
            or term in DEFAULT_INNODB_STOPWORDS
        )
    ]
    if unindexed_terms:
        raise ValueError(
            "all_terms and any_terms cannot independently match short or stop words; "
            "use phrase mode when those words must be matched in context."
        )

    # Duplicates do not change AND/OR semantics and only enlarge the FULLTEXT query.
    distinct_terms = list(dict.fromkeys(terms))
    if match_mode == "all_terms":
        return " ".join(f"+{term}" for term in distinct_terms)
    return " ".join(distinct_terms)


def compile_phrase_text_pattern(text_query: str) -> str:
    """Compile a phrase into a parameterized MariaDB REGEXP token-order filter."""
    terms = parse_text_search_terms(text_query)
    _ = compile_boolean_text_query(text_query, "phrase")
    pattern_terms: list[str] = []
    for term in terms:
        is_prefix = term.endswith("*")
        escaped_term = re.escape(term.removesuffix("*"))
        if is_prefix:
            escaped_term += "[[:alnum:]_]*"
        pattern_terms.append(escaped_term)
    phrase_body = "[^[:alnum:]_]+".join(pattern_terms)
    # MariaDB REGEXP searches for a matching substring. Its native word-boundary
    # classes avoid suffix matches (for example, `no` inside `piano`) without the
    # severe scan penalty caused by boundary alternations around the expression.
    return f"[[:<:]]{phrase_body}[[:>:]]"


def text_query_focus_terms(text_query: str, match_mode: str) -> list[str]:
    """Return snippet terms that mirror the sanitized FULLTEXT query."""
    if match_mode not in TEXT_SEARCH_MODES:
        raise ValueError(
            "text_match_mode must be one of: phrase, all_terms, any_terms."
        )
    stems = [term.removesuffix("*") for term in parse_text_search_terms(text_query)]
    if match_mode == "phrase":
        return [" ".join(stems)]
    return list(dict.fromkeys(stems))
