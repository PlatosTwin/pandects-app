"""Serve the signature block of a mixed last body page in sigPages.

A page that holds the end of the last clause and then the signatures is labeled
body (the house convention: body text plus a signature block is a body page).
The label stays. What changes is what the builder serves: on that page, the
testimonium ("IN WITNESS WHEREOF, the parties ...") and everything after it go
to sigPages, and the clause text stays in the body, so the last section ends
where the clause text ends.

The split is deliberately narrow. It looks only at the last body page, and only
when no signature page comes before it. On that page it splits:

- at the first line that opens with a testimonium phrase, after the page's last
  article or section heading; or, when the page has no testimonium,
- at the first line of the first signature block after the last heading: the
  party header lines ("PARENT:", "ACME CORP.") right above a "By:" line that is
  followed by a "Name:" or "Title:" field, or a block the extraction put on one
  line ("ACME CORP. By: /s/ Jane Doe Name: Jane Doe").

Nothing is split when:

- an article or section heading follows the split point (a signature block in
  the middle of the body, such as a joinder, is followed by more agreement);
- the testimonium is quoted (it opens with a quotation mark, as a form exhibit
  quoted in a clause does) or sits inside a sentence. A testimonium printed in
  capitals right after the sentence end of the clause's last line (the
  extraction glued the two lines) is split there, inside the line;
- the testimonium names another instrument ("this Joinder", "this Consent")
  rather than the agreement, which marks a form of exhibit printed in the body;
- agreement prose (a numbered clause, or a long sentence with an operative
  verb) sits between the split and the first signature field. Text after the
  first field (a consent signed under the blocks, a notary block) moves.

Page-number markers (``<page>`` lines) in the moved tail stay in the body: they
mark the page, not the signature block. Nothing is dropped: the body text and
the moved text together hold every line of the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "IN WITNESS WHEREOF" with the letter spacing some filings print, and the
# other testimonium openings seen on mixed pages.
_TESTIMONIUM_RE = re.compile(
    r"""
    (?:
      I\s*N\s+W\s*I\s*T\s*N\s*E\s*S\s*S\s+(?:W\s*H\s*E\s*R\s*E\s*O\s*F|T\s*H\s*E\s*R\s*E\s*O\s*F)\b
    | IN\s+TESTIMONY\s+WHEREOF\b
    | AS\s+WITNESS\b
    | The\s+part(?:y|ies)(?:\s+hereto)?(?:\s*,[^,\n]{1,80},)?\s+ha(?:ve|s)\s+(?:duly\s+)?(?:caused|executed|signed)\b
    | EXECUTED\s+(?:as\s+(?:of|a\s+deed)|by|on)\b
    | EXECUTED\s+in\b(?:\s+[\w,.'-]+){0,5}?\s+on\b
    | WITNESS,?\s+the\s+(?:signatures?|due\s+execution|execution|hands?)\s+of\b
    | (?:[A-Z][^\n.;:]{0,120}?\s)?ha(?:ve|s)\s+(?:duly\s+)?caused\s+this\s+(?:Agreement|Plan|Contract)\b[^\n]{0,160}?\b(?:executed|signed)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Instruments other than the agreement itself, named by a testimonium that
# belongs to a form printed in the body.
_FORM_INSTRUMENT_RE = re.compile(
    r"\bthis\s+(?:joinder|consent|letter|certificate|assignment|release|guarant(?:y|ee)|"
    r"waiver|instrument\s+of\s+accession|counterpart\s+signature\s+page|amendment|"
    r"notice|power\s+of\s+attorney|deed\s+of\s+adherence|accession)\b",
    re.IGNORECASE,
)
_AGREEMENT_RE = re.compile(r"\b(?:agreement|plan|contract|deed)\b", re.IGNORECASE)
_HEADING_CLOSE_RE = re.compile(r"</(?:section|article)>", re.IGNORECASE)
_HEADING_OPEN_RE = re.compile(r"<(?:section|article)>", re.IGNORECASE)
_PAGE_LINE_RE = re.compile(r"^\s*<page>[^<]*</page>\s*$")
_PAGE_TAG_RE = re.compile(r"<page>[^<]*</page>")
_QUOTE_CHARS = "\"'“”‘’«"

# Signature-block fields.
_FIELD_RE = re.compile(r"^\s*(?:By|Name|Title|Its|Per)\b\s*(?::|_|/s/|$|\s)", re.IGNORECASE)
_BY_RE = re.compile(r"^\s*By\b\s*(?::|_|/s/|\s|$)", re.IGNORECASE)
_NAME_TITLE_RE = re.compile(r"(?:^|\s)(?:Name|Title|Its)\s*:", re.IGNORECASE)
# A party header above a "By:" line: a role label ("PARENT:", "Buyer:"), a
# bracketed placeholder, a capacity line ("as Seller"), or a short line mostly
# in capitals (the entity name).
_ROLE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z ,&'/-]{0,60}:$")
_CAPACITY_LINE_RE = re.compile(r"^\(?(?:as|solely|in\s+its|for\s+and\s+on\s+behalf)\b", re.IGNORECASE)
_STOP_LINE_RE = re.compile(
    r"signature\s+pages?\s+follow|remainder\s+of\s+(?:this\s+)?page|intentionally\s+(?:left\s+)?blank",
    re.IGNORECASE,
)
# Agreement prose in a moved tail: a numbered clause, or a long sentence with
# an operative verb.
_CLAUSE_NUMBER_RE = re.compile(r"^\s*(?:Section\s+)?\d+(?:\.\d+)+\.?\s+[A-Z]|^\s*\((?:[a-z]|[ivx]+)\)\s+[A-Za-z]")
_OPERATIVE_RE = re.compile(r"\b(?:shall|will|agrees?|hereby)\b", re.IGNORECASE)
_PROSE_MIN_WORDS = 40


@dataclass(frozen=True)
class SignatureTailSplit:
    """The body page texts after the split, and the moved signature text.

    ``moved_index`` is the index of the body page the text came from, or None
    when nothing moved; ``trigger`` names what the split keyed on."""

    body_page_texts: list[str]
    moved_text: str | None = None
    moved_index: int | None = None
    trigger: str | None = None


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


_WITNESS_RE = re.compile(r"I\s*N\s+W\s*I\s*T\s*N\s*E\s*S\s*S\b|IN\s+TESTIMONY\b|AS\s+WITNESS\b", re.IGNORECASE)
_UPPER_TESTIMONIUM_RE = re.compile(r"I\s*N\s+W\s*I\s*T\s*N\s*E\s*S\s*S\s+W\s*H\s*E\s*R\s*E\s*O\s*F")
# A sentence end right before a testimonium glued onto the clause's last line.
_SENTENCE_END_RE = re.compile(r"[.;)\]]\s+$")


def _testimonium_candidates(text: str, region_start: int) -> list[int]:
    """Offsets where a testimonium may open, in order: any "IN WITNESS" phrase,
    and the other openings only at the start of a line."""
    starts = {m.start() for m in _WITNESS_RE.finditer(text, region_start)}
    # "^" also matches at region_start itself: text right after a heading tag.
    for m in re.finditer(r"(?m)^[ \t]*", text[region_start:]):
        pos = region_start + m.end()
        if pos < len(text) and _TESTIMONIUM_RE.match(text, pos):
            starts.add(pos)
    return sorted(starts)


def _testimonium_split(text: str, region_start: int) -> tuple[int, str] | None | bool:
    """(offset, trigger) of the first testimonium after region_start; None when
    there is none; False when there is one that must not be split at (a quoted
    phrase, one inside a sentence, or the testimonium of a form)."""
    for pos in _testimonium_candidates(text, region_start):
        if not _TESTIMONIUM_RE.match(text, pos):
            # "IN WITNESS" not followed by WHEREOF/THEREOF ("in witness of").
            continue
        start = _line_start(text, pos)
        prefix = text[max(start, region_start):pos]
        glued = bool(prefix.strip())
        if glued:
            if prefix.rstrip()[-1:] in _QUOTE_CHARS:
                return False
            # The clause's last line runs into the testimonium: split inside
            # the line only at a sentence end, before the printed capitals.
            if not (_SENTENCE_END_RE.search(prefix) and _UPPER_TESTIMONIUM_RE.match(text, pos)):
                return False
        line_end = text.find("\n", pos)
        line = text[pos: line_end if line_end >= 0 else len(text)]
        if _FORM_INSTRUMENT_RE.search(line) and not _AGREEMENT_RE.search(line):
            return False
        return pos, "testimonium_glued" if glued else "testimonium"
    return None


def _is_party_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 14 or _STOP_LINE_RE.search(stripped):
        return False
    if _ROLE_LINE_RE.match(stripped) or _CAPACITY_LINE_RE.match(stripped):
        return True
    if re.fullmatch(r"\[[^\]]{1,80}\]", stripped):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    return len(letters) >= 2 and sum(ch.isupper() for ch in letters) / len(letters) >= 0.6


_INLINE_BY_RE = re.compile(r"\bBy\s*:")
_ROLE_LABEL_LEAD_RE = re.compile(r"^\s*[A-Z][A-Z ]{1,30}:\s")
_INLINE_FIELD_RE = re.compile(r"\b(?:By|Name|Title|Its|Date)\s*:|/s/", re.IGNORECASE)


def _collapsed_block_line(line: str) -> bool:
    """A signature block extracted onto one line: a party name or role label,
    then "By:" and at least one more field ("ACME CORP. By: /s/ Jane Doe Name: ...")."""
    match = _INLINE_BY_RE.search(line)
    if match is None or len(_INLINE_FIELD_RE.findall(line)) < 2:
        return False
    words = line[: match.start()].split()
    if _ROLE_LABEL_LEAD_RE.match(line) and len(words) <= 40:
        # "BUYER: SELLER: ACME CORP., a Delaware corporation ... By:"
        return True
    if not 1 <= len(words) <= 16:
        return False
    capitalized = [w for w in words if w[:1].isupper() or not w[:1].isalpha()]
    return len(capitalized) / len(words) >= 0.75 and not re.search(r"[.;]\s+[a-z]", line[: match.start()])


def _block_split(text: str, region_start: int) -> tuple[int, str] | None:
    """The start of the first signature block after region_start, or None."""
    lines: list[tuple[int, str]] = []
    pos = region_start
    for raw in text[region_start:].split("\n"):
        lines.append((pos, raw))
        pos += len(raw) + 1
    content = [(p, line) for p, line in lines if line.strip() and not _PAGE_LINE_RE.match(line)]
    for k, (_, line) in enumerate(content):
        if _collapsed_block_line(line):
            # The party name and its fields extracted onto one line.
            start = k
            while start > 0 and k - start < 4 and _is_party_header(content[start - 1][1]):
                start -= 1
            block_pos = content[start][0]
            return block_pos + (len(content[start][1]) - len(content[start][1].lstrip())), "signature_block"
        if not _BY_RE.match(line):
            continue
        # A "Name:" or "Title:" field must follow (on the By line or the next two lines).
        window = " ".join(item[1] for item in content[k: k + 3])
        if not _NAME_TITLE_RE.search(window):
            continue
        start = k
        while start > 0 and k - start < 4 and _is_party_header(content[start - 1][1]):
            start -= 1
        if start == k:
            # No party header: a bare "By:" line opens the block only when it
            # follows a heading or a stop line, never right after clause text.
            if k > 0 and not _STOP_LINE_RE.search(content[k - 1][1]):
                return None
        block_pos = content[start][0]
        # Leading whitespace of the block's first line stays with it.
        return block_pos + (len(content[start][1]) - len(content[start][1].lstrip())), "signature_block"
    return None


def _tail_reads_as_prose(tail: str) -> bool:
    """True when agreement prose sits between the split and the first signature field.

    Text after the first field (a consent or joinder signed under the blocks, a
    notary block) is signature-page material and is not tested."""
    for line in tail.split("\n")[1:]:
        stripped = line.strip()
        if not stripped or _PAGE_LINE_RE.match(stripped):
            continue
        if _FIELD_RE.match(stripped) or _NAME_TITLE_RE.search(stripped) or "/s/" in stripped:
            return False
        if _CLAUSE_NUMBER_RE.match(stripped) and len(stripped.split()) >= 8:
            return True
        if (
            len(stripped.split()) >= _PROSE_MIN_WORDS
            and _OPERATIVE_RE.search(stripped)
            and not _TESTIMONIUM_RE.match(stripped)
        ):
            return True
    return False


def find_signature_tail(text: str) -> tuple[int, str] | None:
    """(split offset, trigger) for one page text, or None.

    The offset is where the moved text starts; it is never before the end of
    the page's last article or section heading."""
    if not text or not text.strip():
        return None
    region_start = 0
    for match in _HEADING_CLOSE_RE.finditer(text):
        region_start = match.end()
    found = _testimonium_split(text, region_start)
    if found is False:
        return None
    if found is None:
        found = _block_split(text, region_start)
    if not found:
        return None
    offset, trigger = found
    tail = text[offset:]
    if _HEADING_OPEN_RE.search(tail) or _tail_reads_as_prose(tail):
        return None
    return offset, trigger


def split_signature_tail(
    body_page_texts: list[str],
    body_page_orders: list[int],
    sig_page_orders: list[int],
) -> SignatureTailSplit:
    """Move the signature tail of the last body page out of the body."""
    texts = list(body_page_texts)
    if not texts or len(body_page_orders) != len(texts):
        # Without page orders the pages cannot be placed against the signature pages.
        return SignatureTailSplit(texts)
    last_order = body_page_orders[-1]
    if any(order < last_order for order in sig_page_orders):
        # A signature page before the last body page: the body runs on past
        # the signatures (an exhibit or a joinder labeled body).
        return SignatureTailSplit(texts)
    text = texts[-1]
    found = find_signature_tail(text)
    if found is None:
        return SignatureTailSplit(texts)
    offset, trigger = found
    head, tail = text[:offset], text[offset:]
    # Page-number markers stay with the body page, also one glued into a line.
    page_lines = _PAGE_TAG_RE.findall(tail)
    moved = "\n".join(line for line in _PAGE_TAG_RE.sub("", tail).split("\n") if line.strip() or not line).strip()
    moved = re.sub(r"\n{3,}", "\n\n", moved)
    if not moved:
        return SignatureTailSplit(texts)
    texts[-1] = "\n\n".join(part for part in (head.rstrip(), *page_lines) if part)
    return SignatureTailSplit(texts, moved, len(texts) - 1, trigger)
