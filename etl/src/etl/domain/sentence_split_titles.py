"""Move a section's closing tag out of the clause's own first sentence.

The tagger sometimes closes the heading tag of a section partway into the
clause's first sentence:

    <section>23.2 A party’</section>s exercise of any right ...
    <section>15.1 Subject to clauses</section> 15.2 and 15.3, no party may ...
    <section>Section 11.5. Calculation of LossesNotwithstanding</section> anything ...

It also moves a closing tag to the title's first sentence end, past an initial or
an abbreviation that is not one:

    <section>3.8 Opinion of Robert W.</section> Baird Co. Incorporated. The NV5 ...
    <section>4.10 Reasonable Restrictions. Seller</section> (i) has carefully read ...

The repair only moves the closing tag; the text is never changed. It keeps a
real heading whenever there is evidence of one (a period, a run-in heading that
ends before a sentence opener or the clause's subject, bold/underlined text or
a line break in the source HTML) and gives a number-only title only when the
words after the number positively start a sentence. A leftover long title is
far cheaper than a stripped heading, so anything ambiguous is left alone.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from etl.domain.xml_tag_repairs import (
    XMLTagRepairStats,
    _SENTENCE_OPENERS,
    _TITLE_SMALL_WORDS,
    _is_title_case,
    _strip_word,
)

REPAIR_NAME = "repair_sentence_split_section_titles"

SENTENCE_SPLIT_SECTION_TAG_RE = re.compile(
    r"<section>(?P<inner>[^<]*)</section>(?P<follow>[^<]*)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^(?P<number>\s*(?:(?:Section|SECTION|Clause|CLAUSE)\s+)?\d+(?:\.\d+)*\.?)(?P<gap>\s*)")
_HSPACE = r"[^\S\r\n]"
_DEFINING_VERBS = (
    r"means|mean|has the meaning|have the meaning|shall have the|shall mean|shall include|is defined in|are defined in|"
    r"is defined|are defined|has the respective|have the respective|includes|include|is as defined|refers to"
)
_DEFINING_VERB_RE = re.compile(rf"^{_HSPACE}*[:,]?{_HSPACE}*(?:{_DEFINING_VERBS})\b")
_TERM_VERBS = r"means|shall mean|has the meaning|shall have the meaning|is defined in|are defined in"
_UNQUOTED_TERM_RE = re.compile(rf"^(?P<term>[A-Z][^.;:\n<]{{0,78}}?)(?={_HSPACE}+(?:{_TERM_VERBS})\b)")
_RUN_ON_TERM_RE = re.compile(rf"^(?P<term>.*?\S){_HSPACE}+(?:{_TERM_VERBS})$")
# "Taxes Section 10." + "3 shall ...", "Payment of Royalties - To Seller’" + "s".
_TRAILING_REFERENCE_RE = re.compile(
    rf"{_HSPACE}+(?:Sections?|Clauses?|Articles?|Annex(?:es)?|Schedules?|Paragraphs?){_HSPACE}+\d+(?:\.\d+)*\.?$"
)
_DASH_RE = re.compile(rf"{_HSPACE}+[-–—]{_HSPACE}+")
_REFERENCE_START_RE = re.compile(r"^(?:Sections?|Clauses?|Articles?|Annex(?:es)?|Schedules?|Paragraphs?|Exhibits?)\b")
_HEADING_MAX_CHARS = 100
_HEADING_MAX_WORDS = 12
_EXTENDED_TITLE_MAX_CHARS = 150
_LAYOUT_MAX_WORDS = 20
_LAYOUT_BLOCK_MAX_WORDS = 12
# Words that make a phrase a clause sentence rather than a heading.
_CLAUSE_VERB_RE = re.compile(
    r"\b(?:shall|will|may|must|is|are|was|were|has|have|had|be|been|agrees?|undertakes?|covenants?|"
    r"warrants?|represents?|acknowledges?|confirms?|procures?|waives?|consents?|hereby|means|"
    r"sets?|apply|applies|provides?|contains?|includes?|appoints?|does|do)\b"
)
# A verb that, straight after a capitalized subject, makes the words a sentence.
_SUBJECT_VERB_RE = re.compile(
    r"(?:shall|will|may|must|is|are|has|have|agrees?|undertakes?|covenants?|warrants?|represents?|"
    r"acknowledges?|confirms?|procures?|waives?|appoints?|hereby|grants?|guarantees?)"
)
# The subject of a clause printed straight after a run-in heading
# ("Section 4.7 Financing Parent" + " has delivered ...").
_PARTY_WORDS = (
    r"(?:Parent|Purchasers?|Buyers?|Sellers?|Acquiror|Acquirer|Vendors?|Company|Target|Shareholders?|"
    r"Stockholders?|Guarantors?|Warrantors?|Covenantors?|Investors?|Subscribers?|Issuer|Licensor|Licensee)"
)
_PARTY_RE = re.compile(rf"{_PARTY_WORDS}")
# A title glued to the first word of its text: "Calculation of LossesNotwithstanding".
_GLUED_RE = re.compile(r"[a-z’')](?P<word>[A-Z][a-z]+)")
_GLUED_OPENERS = frozenset("the each nothing neither none promptly following".split())
# Words that positively start a clause sentence.
_SENTENCE_START_WORDS = _SENTENCE_OPENERS | frozenset(
    "none neither nothing to both either every until should pending once save under".split()
)
# A run-in heading ends right before one of these (once it holds a content word).
_RUN_STOP_WORDS = frozenset(
    """
    the any each in at no none neither except to for upon if subject on as all where when unless
    notwithstanding save following within after before during every nothing
    """.split()
)
# ... even when the word after it is capitalized ("Corporate Organization The Company").
_SUBJECT_STOP_WORDS = frozenset("the any each no none neither every all such this".split())
# A defined term never starts with one of these.
_NOT_A_TERM_START = frozenset("the this that it a an such each these those its any all".split())
_ADVERB_RE = re.compile(r"(?:\w+ly|hereby|each|jointly|severally|also)$")
_RUN_MAX_WORDS = 12
# A period that does not end a sentence: a single initial ("Robert W.", "Non-U.", "C.F.") or a
# common abbreviation ("St.", "Mr.", "Jr.", "Co.").
_ABBREVIATION_END_RE = re.compile(
    r"(?:(?<![A-Za-z])[A-Z]|\b(?:Mr|Mrs|Ms|Messrs|Dr|Hon|Prof|Rev|St|Mt|Ft|Co|Inc|Corp|Ltd|Bros|Jr|Sr|Nos?))\.$"
)
_SENTENCE_END_RE = re.compile(rf"\.(?={_HSPACE}+[A-Z\x22“(]|{_HSPACE}*(?:\n|$))")


def _norm(word: str) -> str:
    return _strip_word(word.replace("’", "'").replace("‘", "'")).lower()


# --- layout evidence from the source HTML -------------------------------------------------

_RAW_TOKEN_RE = re.compile(r"<[^>]*>|[^<]+")
_RAW_TAG_RE = re.compile(r"<\s*(/)?\s*([a-zA-Z][a-zA-Z0-9]*)")
_FORMAT_TAGS = frozenset("b strong u i em".split())
_BLOCK_TAGS = frozenset("p div td th tr li br table h1 h2 h3 h4 h5 h6 center blockquote pre".split())
_VOID_TAGS = frozenset("br img hr meta input link col wbr".split())
_STYLE_FORMAT_RE = re.compile(
    r"font-weight\s*:\s*(?:bold|[6-9]00)|text-decoration\s*:\s*underline|font-style\s*:\s*italic", re.IGNORECASE
)


@dataclass
class _Layout:
    """The alphanumeric characters of a page's source HTML, each with whether it is
    bold/underlined/italic and whether a block break (paragraph, cell, line break)
    comes right before it."""

    chars: str
    formatted: list[tuple[str, ...]]  # the formatting tags open around each character
    block_before: list[bool]

    @classmethod
    def from_raw(cls, raw: str) -> _Layout | None:
        if "<" not in raw:
            return None  # plain-text source: no layout evidence
        chars: list[str] = []
        formatted: list[tuple[str, ...]] = []
        block_before: list[bool] = []
        stack: list[tuple[str, bool]] = []
        pending_block = True
        for token in _RAW_TOKEN_RE.findall(raw):
            if token.startswith("<"):
                tag = _RAW_TAG_RE.match(token)
                if tag is None:
                    continue
                closing, name = tag.group(1), tag.group(2).lower()
                if name in _BLOCK_TAGS:
                    pending_block = True
                if name in _VOID_TAGS or token.endswith("/>"):
                    continue
                if closing:
                    for i in range(len(stack) - 1, -1, -1):
                        if stack[i][0] == name:
                            del stack[i:]
                            break
                else:
                    fmt = name in _FORMAT_TAGS or _STYLE_FORMAT_RE.search(token) is not None
                    stack.append((name, fmt))
                continue
            signature = tuple(name for name, fmt in stack if fmt)
            for ch in html.unescape(token):
                if ch.isalnum():
                    chars.append(ch.lower())
                    formatted.append(signature)
                    block_before.append(pending_block)
                    pending_block = False
        return cls("".join(chars), formatted, block_before)

    def heading_length(self, number: str, joined: str) -> tuple[int, str] | None:
        """Alphanumeric length of a heading that the source prints right after the
        section number, set off by formatting or by a block break, or None."""
        digits = "".join(ch for ch in number if ch.isalnum() and not ch.isalpha())
        body = "".join(ch.lower() for ch in joined if ch.isalnum())
        if not digits or len(body) < 4:
            return None
        key = digits + body[:30]
        start = self.chars.find(key)
        while start >= 0:
            p = start + len(digits)
            signature = self.formatted[p]
            kind = "format" if signature else "block"
            end = p + 1
            while end < len(self.chars) and not self.block_before[end] and (
                kind == "block" or self.formatted[end] == signature
            ):
                end += 1
            if end < len(self.chars) and end - p < len(body):
                return end - p, kind
            start = self.chars.find(key, start + 1)
        return None


def _cut_after_alnum(joined: str, count: int) -> int | None:
    """Offset in joined just after its count-th alphanumeric character (plus any
    punctuation glued to it, like the period in "Guarantee.")."""
    seen = 0
    for i, ch in enumerate(joined):
        if ch.isalnum():
            seen += 1
            if seen == count:
                end = i + 1
                while end < len(joined) and not joined[end].isspace() and not joined[end].isalnum():
                    end += 1
                return end
    return None


# --- run-in headings ------------------------------------------------------------------------


def _run_in_heading_end(joined: str, context: str | None = None) -> int | None:
    """End of a heading printed without a period before the clause sentence, or None.

    "Survival None of the ...", "Corporate Organization The Company is ...",
    "Financing Parent has ...", "Operations of Acquisition Sub Acquisition Sub ...",
    "Organization; Authority The undersigned ...".
    """
    line = joined.split("\n", 1)[0]
    words = list(re.finditer(r"\S+", line))
    if len(words) < 2:
        return None
    first = _strip_word(words[0].group())
    if not first or not first[0].isupper() or _REFERENCE_START_RE.match(first):
        return None
    second = _strip_word(words[1].group())
    if first.lower() in _SENTENCE_START_WORDS and not (first.lower() == "no" and second[:1].isupper()):
        return None
    norms = [_norm(w.group()) for w in words]
    content_before = 1
    for i in range(1, min(len(words), _RUN_MAX_WORDS + 1)):
        token = words[i].group()
        word = _strip_word(token)
        if not word:
            return None
        lower = word.lower()
        if words[i - 1].group().endswith((".", ":", ",")) and not words[i - 1].group().endswith("etc."):
            if words[i - 1].group().endswith(",") and word[:1].isupper() and lower in _RUN_STOP_WORDS:
                return _run_end(words, i)  # "No Undisclosed Liabilities, (Except for ..."
            return None  # a period or comma inside the run: not a plain run-in heading
        nxt = _strip_word(words[i + 1].group()) if i + 1 < len(words) else ""
        after = " ".join(_strip_word(w.group()).lower() for w in words[i + 1 : i + 3])
        # An adjacent repeat of the run's last 1-3 words: "Acquisition Sub Acquisition Sub".
        for k in (3, 2, 1):
            if i - k >= 0 and i + k <= len(words) and norms[i : i + k] == norms[i - k : i] and norms[i]:
                return _run_end(words, i)
        if word.isupper() and len(word) > 1 and lower in _RUN_STOP_WORDS and not first.isupper():
            return None  # "Disclosure IF THE DISCLOSURE SUMMARY ...": an all-caps clause
        if word[0].isupper():
            previous = _strip_word(words[i - 1].group())
            if lower in _RUN_STOP_WORDS and (
                nxt[:1].islower()
                or lower in _SUBJECT_STOP_WORDS
                or nxt == ""
                or (previous[:1].islower() and previous.lower() not in _TITLE_SMALL_WORDS)
            ):
                return _run_end(words, i)  # "Sellers’ obligations On Closing ..."
            subject_verb = _SUBJECT_VERB_RE.fullmatch(nxt.lower()) or (
                _ADVERB_RE.fullmatch(nxt.lower()) and re.match(_SUBJECT_VERB_RE, after.split(" ")[-1] or "")
            )
            # A list is the subject itself: "Sellers and Asset Sellers shall", "Holdings and
            # each of Alpha Holdings and Beta Holdings agrees".
            joins_list = any(norms[j] in ("and", "or") for j in range(1, i))
            party_list = joins_list and any(_PARTY_RE.fullmatch(_strip_word(w.group())) for w in words[:i])
            if subject_verb and _PARTY_RE.fullmatch(word) and not party_list:
                if words[i - 1].group().endswith(";"):
                    return None  # "Seller; Buyer shall": two parties, not a heading
                pair = f"{previous} {word}"
                if len(re.findall(rf"(?<!\w){re.escape(pair)}(?!\w)", context or joined)) >= 2:
                    return None  # "Target Shareholders" recurs: a party name, not heading + subject
                return _run_end(words, i, subject=True)  # "Financing Parent has"
            if subject_verb and norms[i] in norms[:i] and content_before >= 4 and not joins_list:
                return _run_end(words, i, subject=True)  # "... Memorandum Century shall"
            if _REFERENCE_START_RE.match(word) and nxt[:1].isdigit():
                return _run_end(words, i)  # "Separate undertakings Clauses 17.1.1 ..."
            if lower not in _TITLE_SMALL_WORDS:
                content_before += 1
            continue
        if _CLAUSE_VERB_RE.fullmatch(lower):
            return None
        if lower not in _TITLE_SMALL_WORDS:
            content_before += 1
    return None


def _run_end(words: list[re.Match[str]], i: int, subject: bool = False) -> int | None:
    run = [_strip_word(w.group()) for w in words[:i]]
    if not run or run[-1].lower() in _TITLE_SMALL_WORDS or len(run) > _RUN_MAX_WORDS:
        return None
    content = [w for w in run if w.lower() not in _TITLE_SMALL_WORDS and w.lower() not in _SENTENCE_START_WORDS]
    if not content or (subject and run[0].lower() == "no" and len(content) < 2):
        return None  # "No" + " Seller shall", "No Majority" + " Seller shall": the clause's subject
    lowercase = [w for w in run if w and w[0].islower() and w.lower() not in _TITLE_SMALL_WORDS]
    if (
        len(lowercase) > 2
        or any(_CLAUSE_VERB_RE.fullmatch(w.lower()) for w in run)
        or (run[-1][:1].islower() and run[-1].lower() in _SENTENCE_START_WORDS)
    ):
        return None  # "Termination under this" + " Article 12 shall ...": a sentence
    return words[i - 1].end()


# --- title boundaries -----------------------------------------------------------------------


def _first_sentence_end(text: str, start: int = 0) -> int | None:
    """Offset just after the first period in text that ends a sentence, skipping initials and
    abbreviations ("A.T. Massey Sub.", "Lindahl, Jr. State Industries Trust.")."""
    for match in _SENTENCE_END_RE.finditer(text, start):
        if _ABBREVIATION_END_RE.search(text[: match.end()]) is None:
            return match.end()
    return None


def _abbreviation_cut_end(rest: str, follow: str) -> int | None:
    """End of a heading whose closing tag stopped at an initial or abbreviation, or None.

    "Opinion of Robert W." + " Baird Co. Incorporated. The NV5 ...", "Capitalization of Altair
    U." + "S. There are ...". The heading runs to the next real sentence end on the line, and
    the words up to there must read as the rest of the heading, not as the clause: "Regulation
    D." + " The Company is aware ...", "Endo Pharmaceuticals Inc." + " Endo Guarantor shall ...".
    """
    if _ABBREVIATION_END_RE.search(rest.rstrip()) is None:
        return None
    line = follow.split("\n", 1)[0]
    glued_initial = re.match(rf"[A-Z]\.(?={_HSPACE}+(?P<next>\S+))", line)
    if glued_initial is not None and _norm(glued_initial.group("next")) in _SENTENCE_START_WORDS:
        return len(rest) + glued_initial.end()  # "Altair U." + "S. There are ..."
    end = _first_sentence_end(line)
    if end is None:
        return None
    fragment = line[:end]
    words = fragment.split()
    title_words = {_norm(word) for word in rest.split()}
    if (
        not words
        or len(words) > _HEADING_MAX_WORDS
        or not words[0][:1].isupper()
        or _norm(words[0]) in _SENTENCE_START_WORDS
        or _norm(words[0]) in title_words  # "AT&T." + " AT&T Broadband, LLC, a Delaware ..."
        or _CLAUSE_VERB_RE.search(fragment) is not None
        or len(rest) + end > _EXTENDED_TITLE_MAX_CHARS
        or not _is_title_case(rest + fragment)
    ):
        return None
    return len(rest) + end


def _past_sentence_end(rest: str, follow: str) -> int | None:
    """End of the heading when the title runs past it into the clause's first words, or None.

    "Reasonable Restrictions. Seller" + " (i) has carefully read ...", "Assistance to CPA18. W."
    + " P. Carey will ...", "Guarantee. RTL hereby guarantees". A second heading stays in the
    title: "Access to Information. Confidentiality.", "Entire Agreement. Changes in Writing.".
    """
    end = _first_sentence_end(rest)
    if end is None or not rest[end:].strip():
        return None
    heading, tail = rest[:end], rest[end:].strip()
    words = tail.split()
    if not _looks_like_heading(heading) or _REFERENCE_START_RE.match(heading) or _REFERENCE_START_RE.match(tail):
        return None
    if re.fullmatch(r"[A-Z]\.", tail):
        return end  # an initial: the clause's first words are "W. P. Carey will ..."
    if len(words) <= 3 and not tail.endswith(".") and re.match(r"\s*(?:\([a-z0-9]{1,4}\)\s*)?[a-z]", follow):
        return end  # the clause's subject: "Seller" + " (i) has ..."
    if _CLAUSE_VERB_RE.search(tail) is not None and not _is_title_case(tail) and not _strip_word(words[-1])[:1].isupper():
        return end  # "Guarantee. RTL hereby guarantees payment.", not "Notices. Time is of the Essence."
    return None


# --- the repair -----------------------------------------------------------------------------


def _split_heading_end(text: str, limit: int) -> int | None:
    """End of a heading that the tag cut short: a period before a new sentence, or the line end."""
    match = re.search(r"\.(?=[^\S\r\n]+[A-Z\x22“(])|\n", text[:limit])
    if match is None:
        return None
    return match.start() + 1 if match.group(0) == "." else match.start()


def _looks_like_heading(heading: str) -> bool:
    heading = heading.strip()
    return (
        bool(heading)
        and len(heading) <= _HEADING_MAX_CHARS
        and len(heading.split()) <= _HEADING_MAX_WORDS
        and _CLAUSE_VERB_RE.search(heading) is None
        and _norm(heading.split()[0]) not in _SENTENCE_START_WORDS
        and not heading.endswith(":")
    )


def _verb_follows(follow: str) -> bool:
    words = [_strip_word(w).lower() for w in follow.split()[:2]]
    if not words:
        return False
    if _SUBJECT_VERB_RE.fullmatch(words[0]):
        return True
    return len(words) == 2 and _ADVERB_RE.fullmatch(words[0]) is not None and bool(_SUBJECT_VERB_RE.fullmatch(words[1]))


def _sentence_signal(rest: str, follow: str, split_token: bool) -> bool:
    """Positive evidence that the words after the number start the clause's sentence."""
    words = rest.split()
    first = _norm(words[0])
    second = _strip_word(words[1]) if len(words) > 1 else _strip_word(follow.split()[0] if follow.split() else "")
    if first in _SENTENCE_START_WORDS and not (first == "no" and second[:1].isupper()):
        return True  # "A party’", "Subject to clauses", "The Buyer", "Each of the Warranties"
    if split_token:
        return True  # "A party’" + "s exercise": the tag splits a word
    if _REFERENCE_START_RE.match(words[0]) and len(words) > 1 and words[1][:1].isdigit():
        return True  # "Clause 11.12 and 11.13 apply" + " notwithstanding ..."
    if first != "" and _norm(words[-1]) in _TITLE_SMALL_WORDS and _CLAUSE_VERB_RE.search(" ".join(follow.split()[:8])):
        return True  # "Completion of" + " this agreement shall take place ..."
    if not words[0][:1].isupper():
        return False
    if len(words) <= 5 and _is_title_case(rest) and _verb_follows(follow):
        return True  # the whole of rest is the subject: "Kerr-McGee" + " shall ...", "Seller" + " shall ..."
    verb = next((i for i, w in enumerate(words) if _SUBJECT_VERB_RE.fullmatch(_strip_word(w))), None)
    return verb is not None and 1 <= verb <= 6 and all(w[:1].isupper() or w.lower() in _TITLE_SMALL_WORDS for w in words[:verb])


def _glued_cut(rest: str, follow: str) -> int | None:
    """Offset in rest of the first word of the text when the title is glued to it."""
    for match in _GLUED_RE.finditer(rest):
        word = match.group("word")
        before = rest[: match.start("word")]
        after = rest[match.end("word") :] + follow
        if not re.search(r"[A-Za-z]", before):
            continue
        if word.lower() in _SENTENCE_OPENERS | _GLUED_OPENERS:
            if re.match(r"\s*[a-z(]", after) or re.search(r"\s[a-z]", after[:40]):
                return match.start("word")
        elif _PARTY_RE.fullmatch(word) and re.search(r"[\s-]", before.strip()) and _verb_follows(after):
            return match.start("word")  # "Spin-OutParent" + " shall ...", not "NewCompany shall"
    return None


def _repair_sentence_split_title(
    inner: str, follow: str, layout: _Layout | None = None, context: str | None = None
) -> tuple[str, str] | None:
    """Return (new_inner, new_follow) for a tag that runs into its own sentence, or None.

    The joined text inner + follow is unchanged; only the closing tag moves.
    """
    number = _NUMBER_RE.match(inner)
    if number is None:
        return None
    rest = inner[number.end() :]
    if not rest.strip() or re.search(r"[\x22“”]", rest):
        return None
    joined = rest + follow

    def result(cut: int) -> tuple[str, str] | None:
        if cut > 0:
            head, tail = inner[: number.end()] + joined[:cut], joined[cut:]
        else:
            head, tail = inner[: number.end("number")], inner[number.end("number") :] + follow
        if cut > len(rest) and re.search(r"(?:\d+\.)+\d*[\s.]*$", joined[len(rest) : cut]):
            return None  # a table-of-contents line: "Allocation of Payment" + " 1.12"
        trimmed = re.sub(r"[^\S\r\n]*(?:[-–—]+|[:,;(])[^\S\r\n]*$", "", head)
        if trimmed != head and _NUMBER_RE.match(trimmed) is not None:
            head, tail = trimmed, head[len(trimmed) :] + tail  # "Gathering System -" keeps no dash
        if head == inner or head + tail != inner + follow:
            return None
        if re.search(r"[\x22“”]", head) or head.count("(") != head.count(")"):
            return None  # a title never takes in a quoted term or half a parenthetical
        return head, tail

    glued = _glued_cut(rest, follow)
    if glued is not None:
        return result(glued)  # "Calculation of LossesNotwithstanding" + " anything ..."

    boundary = _abbreviation_cut_end(rest, follow)
    if boundary is None:
        boundary = _past_sentence_end(rest, follow)
    if boundary is not None:
        return result(boundary)

    last, nxt = inner[-1:], follow[:1]
    digit_split = re.search(r"\d\.?$", rest) is not None and nxt.isdigit()
    split_token = digit_split or (last in "’'" and nxt.islower()) or (last.isalnum() and nxt.islower())
    continues = re.match(r"[^\S\r\n]+(?P<c>\S)", follow)
    sentence_runs_on = (
        (
            continues is not None
            and (
                continues.group("c").islower()
                or continues.group("c").isdigit()
                or re.match(r"[^\S\r\n]+\([a-z]{2,}\b", follow) is not None  # " (as named above ...)"
            )
            # A comma continues the sentence: "Survival None of the representations" + ", warranties".
            or re.match(r",[^\S\r\n]+[a-z]", follow) is not None
        )
        and _DEFINING_VERB_RE.match(follow) is None
        and re.match(r"[^\S\r\n]+\([a-z0-9]{1,4}\)", follow) is None
    )
    if not split_token and not sentence_runs_on:
        return None

    # A Title Case term inside the tag: "1.14 Parties are Sellers, TriQuint and Purchaser" + " collectively."
    inner_term = re.match(
        rf"^(?P<term>\S+(?:{_HSPACE}+\S+){{0,4}}?){_HSPACE}+(?:is|are){_HSPACE}+(?=[A-Z]|(?:the|a|an|all|any)\b)", rest
    )
    if inner_term is not None:
        sentence_end = re.search(r"[.;:](?=\s|$)", joined[inner_term.end() :])
        predicate = joined[inner_term.end() : inner_term.end() + sentence_end.end() if sentence_end else len(joined)]
        # A sentence, not a heading: a lowercase content word ("collectively"), but not
        # "Obligations are Several, Not Joint." or "Sellers are Responsible for Taxes, Fees and Costs".
        lowercase_word = any(
            w[:1].islower() and _norm(w) not in _TITLE_SMALL_WORDS for w in predicate.split()
        )
        if (
            lowercase_word
            and _is_title_case(inner_term.group("term"))
            and _norm(inner_term.group("term").split()[0]) not in _NOT_A_TERM_START
            and _CLAUSE_VERB_RE.search(inner_term.group("term")) is None
        ):
            return result(len(inner_term.group("term")))

    # Defined terms: "Proprietary IPR means" + " the IPR ...", "Effective Date" + " is the date ...".
    run_on_term = (
        _RUN_ON_TERM_RE.match(rest)
        or re.match(rf"^(?P<term>.*?\S){_HSPACE}+(?:is|are)$", rest)
        # "Claim Notice is defined in clause 10." + "4"
        or re.match(rf"^(?P<term>[^.;:]*?\S){_HSPACE}+(?:is|are){_HSPACE}+defined{_HSPACE}+in\b", rest)
    )
    follow_first = _norm(follow.split()[0]) if follow.split() else ""
    if (
        run_on_term is not None
        and follow_first in ("of", "not", "to", "in", "subject")
        and _norm(run_on_term.group("term").split()[0]) not in _NOT_A_TERM_START  # not "The Transfer is"
    ):
        return None  # "Business Day is" + " not a Saturday ...", "Knowledge is" + " to be construed ..."
    if run_on_term is not None and (
        _norm(run_on_term.group("term").split()[0]) in _NOT_A_TERM_START
        or _ADVERB_RE.fullmatch(follow_first)  # "Time is" + " expressly of the essence"
    ):
        run_on_term = None  # "This Agreement is" + " intended ...", "It is" + " agreed ..."
    if run_on_term is not None and len(run_on_term.group("term").split()) <= 5:
        return result(len(run_on_term.group("term"))) if _is_title_case(run_on_term.group("term")) else None
    if (
        re.match(r"[^\S\r\n]+(?:is|are)\b", follow)
        and len(rest.split()) <= 5
        and _is_title_case(rest)
        and not re.search(r"\b(?:the|a|an|The|A|An)\b", rest)
        and _norm(rest.split()[0]) not in _NOT_A_TERM_START
    ):
        return None
    if split_token and not digit_split and _is_title_case(rest) and last not in "’'":
        return None  # "Non-U" + "S. Plans": a heading split inside a word
    if rest.rstrip().endswith(".") and not digit_split and _CLAUSE_VERB_RE.search(rest) is None:
        return None  # "5.7 eSpeed SEC Documents." + " ...": a complete heading

    # The clause's own first subsection follows the heading: "The Escrow Agent" + " 8.2.1 ...".
    own_number = re.search(r"\d+(?:\.\d+)*", inner[: number.end()])
    if own_number is not None:
        subsection = re.search(
            rf"\s{re.escape(own_number.group(0))}\.\d+(?:\.\d+)*\.?(?=[^\S\r\n]*$|\s+[A-Z(])", joined
        )
        if subsection is not None and subsection.start() > 0:
            heading = joined[: subsection.start()].rstrip()
            words = heading.split()
            if (
                words
                and len(words) <= _HEADING_MAX_WORDS
                and _CLAUSE_VERB_RE.search(heading) is None
                and heading[-1:] not in ":,;"
                and _norm(words[-1]) not in _TITLE_SMALL_WORDS
                and heading.count("(") == heading.count(")")
                # "Subject to" + " clause 5.2.1, ...": a cross-reference, not a heading.
                and re.fullmatch(r"(?:sub-?)?(?:clauses?|sections?|paragraphs?|articles?)", _norm(words[-1])) is None
                and not re.match(r"\s*[,(]", heading[len(rest.rstrip()) :])  # "Liabilities" + ", (Except for ..."
                # A bilingual heading repeats the number: "Schiedsgutachterverfahren 6.6 Expert Proceedings".
                and re.search(rf"(?<![\d.]){re.escape(own_number.group(0))}(?![\d])", heading) is None
            ):
                return result(len(heading)) if len(heading) != len(rest.rstrip()) else None

    # Layout: the source prints a heading in bold/underline or on its own line.
    if layout is not None:
        found = layout.heading_length(inner[: number.end()], joined)
        if found is not None:
            length, kind = found
            cut = _cut_after_alnum(joined, length)
            if cut is None or cut >= len(joined.rstrip()):
                return None
            heading = joined[:cut]
            words = heading.split()
            after = joined[cut:].lstrip()
            if (
                not words
                or len(words) > (_LAYOUT_MAX_WORDS if kind == "format" else _LAYOUT_BLOCK_MAX_WORDS)
                or after[:1].islower()  # the sentence runs on: "Seller" + " shall ...", "Clause 4.1" + " and ..."
                or heading.rstrip()[-1:] in ",:;("
                or heading.count("(") != heading.count(")")
                or re.search(r"[\x22“”]", heading) is not None
                or _norm(words[-1]) in _TITLE_SMALL_WORDS
                or (
                    _REFERENCE_START_RE.match(words[0])
                    and len(words) > 1
                    and words[1][:1].isdigit()
                    # "Section 13.1-728.2 et seq. of the VSCA." printed bold is a heading.
                    and not (kind == "format" and re.search(r"\bof the\b", heading))
                )
                or _CLAUSE_VERB_RE.search(heading) is not None  # "<b>6.1 Completion shall take place ...</b>"
                or (
                    _norm(words[0]) in _SENTENCE_START_WORDS
                    and not (_norm(words[0]) == "no" and len(words) > 1 and words[1][:1].isupper())
                    and not _is_title_case(heading)  # "The Parties' Covenants", "The Escrow Agent"
                )
            ):
                if (
                    words
                    and len(words) <= 8
                    and _CLAUSE_VERB_RE.search(heading) is None
                    and not after[:1].islower()
                    # A lead-in or a run-in heading is not set off: "The Suspensive Conditions in:" + " 3.4.1",
                    # "<u>No Undisclosed Liabilities</u>, (Except ...".
                    and heading.rstrip()[-1:] not in ",:;("
                    and _norm(words[-1]) not in _TITLE_SMALL_WORDS
                    and (_is_title_case(heading) or _norm(words[0]) not in _SENTENCE_START_WORDS)
                ):
                    return None  # a short set-off heading we cannot place: never strip it
                found = None
            else:
                return result(cut)

    # The tag stopped short of a real heading or defined term: move the closing tag out.
    may_extend = _REFERENCE_START_RE.match(rest) is None
    term = _UNQUOTED_TERM_RE.match(joined)
    if may_extend and term is not None and len(term.group("term")) > len(rest) and _is_title_case(term.group("term")):
        return result(len(term.group("term")))  # "Buyer’" + "s Representative means ..."
    heading_end = _split_heading_end(joined, _EXTENDED_TITLE_MAX_CHARS)
    if (
        may_extend
        and heading_end is not None
        and heading_end > len(rest)
        and _CLAUSE_VERB_RE.search(joined[:heading_end]) is None
        and joined[:heading_end].rstrip()[-1:] not in ":,;"
        and _norm(rest.split()[0]) not in _SENTENCE_START_WORDS
        and _ADVERB_RE.fullmatch(_norm(rest.split()[0])) is None  # "Promptly following the Signing Date,"
        and (
            _is_title_case(joined[:heading_end])
            or (joined[heading_end : heading_end + 1] == "\n" and _looks_like_heading(joined[:heading_end]))
        )
    ):
        return result(heading_end)  # "Purchase Price and payment" + " thereof\n"

    # A heading that ends with a period inside the tag: "Guarantee. RTL hereby".
    period = re.search(r"\.(?=[^\S\r\n]+[A-Z\x22“(])", rest)
    if period is not None:
        heading = rest[: period.end()]
        if (
            len(heading.split()) <= _HEADING_MAX_WORDS
            and _CLAUSE_VERB_RE.search(heading) is None
            and _norm(heading.split()[0]) not in _SENTENCE_START_WORDS
            and _REFERENCE_START_RE.match(heading) is None
        ):
            return result(period.end())

    # A Title Case heading before a trailing cross-reference or a dash.
    for splitter in (_TRAILING_REFERENCE_RE, _DASH_RE):
        found_split = splitter.search(rest)
        if found_split is None or found_split.start() == 0:
            continue
        title = rest[: found_split.start()]
        title_words = title.split()
        if (
            _is_title_case(title)
            and _norm(title_words[0]) not in _SENTENCE_START_WORDS
            and title_words[-1].lower() not in _TITLE_SMALL_WORDS
        ):
            return result(found_split.start())

    run_end = _run_in_heading_end(joined, context)
    if run_end is not None:
        return result(run_end)  # "Survival None of ...", "Financing Parent has ..."

    if _sentence_signal(rest, follow, split_token):
        return result(0)  # an untitled clause: the title is the number alone
    return None


def repair_sentence_split_section_titles(
    tagged_text: str, raw_page_content: str | None = None
) -> tuple[str, XMLTagRepairStats]:
    """Move a section's closing tag out of the clause's own first sentence.

    Untitled clauses (common in English-law agreements) get a number-only title
    and keep their whole sentence as body text: "<section>23.2 A party’</section>s
    exercise ..." becomes "<section>23.2</section> A party’s exercise ...". When
    the tag cut a real heading short or ran past it ("7.5 Purchaser’" + "s Receipt
    of Approvals ...", "Survival None of ..."), the closing tag moves to the end
    of that heading. raw_page_content, when given, is the page's source HTML; a
    heading printed there in bold/underline or on its own line is kept.
    """
    stats = XMLTagRepairStats()
    layout_cache: list[_Layout | None] = []

    def layout() -> _Layout | None:
        if not layout_cache:
            layout_cache.append(
                _Layout.from_raw(raw_page_content) if isinstance(raw_page_content, str) and raw_page_content else None
            )
        return layout_cache[0]

    def repl(match: re.Match[str]) -> str:
        stats.attempts[REPAIR_NAME] += 1
        line_start = tagged_text.rfind("\n", 0, match.start()) + 1
        if tagged_text[line_start : match.start()].strip():
            # Mid-line: a cross-reference tagged as a heading ("... set forth in
            # <section>Section 3.11 of the Disclosure Schedules ..."), which the
            # later unwrap repairs handle.
            return match.group(0)
        if re.match(r"\s*(?:\S+\s+)?\d+(?:\.\d+)*(?:[,;]|\.?\s+of\s)", match.group("inner")):
            return match.group(0)  # "Section 13.2, Section 13.3 ...", "Section 3.11 of the ..."
        repaired = _repair_sentence_split_title(match.group("inner"), match.group("follow"), layout(), tagged_text)
        if repaired is None:
            return match.group(0)
        stats.applied[REPAIR_NAME] += 1
        head, tail = repaired
        return f"<section>{head}</section>{tail}"

    return SENTENCE_SPLIT_SECTION_TAG_RE.sub(repl, tagged_text), stats
