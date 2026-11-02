"""Remove underline dash runs that text extraction fused into .txt page text.

Old EDGAR .txt filings underline a caption or a phrase by printing a line of
dashes directly under it::

         1.9  Lost, Stolen or Destroyed Certificates. In the event that any
              --------------------------------------
    Certificates shall have been lost, ...

Page text extraction reflows the lines into paragraphs, so the dash line lands
inside the sentence ("In the event that any -------------------------------------- Certificates
shall have been lost"). This module removes those runs and nothing else.

The strip is deliberately narrow. A dash group in the page text is removed only
when the raw page text proves it is an underline:

* In the raw text, a line holds nothing but dash runs (each at least two
  dashes, one at least three), and every run sits directly under printed text
  on the line above: the run starts at the start of a word and ends on a
  printed character (either end may be one column off, as typed underlines
  often are), and most of the columns above it are printed. A blank to
  fill in (``Address: ------``, ``$----------``, a dash line under a ``By:``
  label) shares its line with text or sits under white space, so it never
  qualifies.
* In the page text, the group is the same runs, with the same lengths, as
  whole tokens on a line that also holds other text, and the word right before
  it is the last word of the raw line above the underline (where the reflow
  put it). Each raw underline removes at most one group.

Whole lines of dashes are left to the EDGAR rule-line strip (``edgar_sgml``),
which has its own guards for fill-in blanks and signature lines.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

_DASH_RUN_RE = re.compile(r"-+")
_TAG_RE = re.compile(r"<[^<>]*>")
_MIN_RUN = 2
_MIN_LONGEST_RUN = 3
# Share of the columns above a run that must hold a printed character.
_MIN_COVERED_SHARE = 0.6
# Characters a run may start under, besides the start of a word: an underlined
# "(a)" or a quoted term starts one column early.
_OPENERS = "(\"'“‘["


@dataclass
class UnderlineStripResult:
    text: str
    removed: list[str] = field(default_factory=list)


def _visible(token: str) -> str:
    return _TAG_RE.sub("", token)


def _is_underline_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or set(stripped) - {"-", " "}:
        return False
    runs = stripped.split()
    return all(len(run) >= _MIN_RUN for run in runs) and max(len(run) for run in runs) >= _MIN_LONGEST_RUN


def _printed(above: str, column: int) -> bool:
    return 0 <= column < len(above) and above[column] != " "


def _starts_word(above: str, column: int) -> bool:
    return column == 0 or above[column - 1] == " " or above[column] in _OPENERS or above[column - 1] in _OPENERS


def _run_is_under_text(above: str, start: int, end: int) -> bool:
    """The run underlines printed words. Typed underlines can sit one column off,
    so a run may start one column into its first word and end one column past
    its last printed character."""
    if not _printed(above, end - 1):
        if not _printed(above, end - 2):
            return False
        end -= 1
    if not _printed(above, start):
        if not (_printed(above, start + 1) and _starts_word(above, start + 1)):
            return False
        start += 1
    elif not _starts_word(above, start) and not (start >= 1 and _starts_word(above, start - 1)):
        return False
    if end - start < _MIN_RUN:
        return False
    covered = sum(1 for ch in above[start:end] if ch != " ")
    return covered >= _MIN_COVERED_SHARE * (end - start)


def raw_underlines(raw_text: str) -> Counter[tuple[str, tuple[int, ...]]]:
    """(last word of the underlined line, run lengths) for each underline in the raw text."""
    found: Counter[tuple[str, tuple[int, ...]]] = Counter()
    if not raw_text or "--" not in raw_text:
        return found
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(8).split("\n")
    for index in range(1, len(lines)):
        line = lines[index]
        if not _is_underline_line(line):
            continue
        above = lines[index - 1]
        if not above.strip() or _is_underline_line(above):
            continue
        runs = [(m.start(), m.end()) for m in _DASH_RUN_RE.finditer(line)]
        if not all(_run_is_under_text(above, start, end) for start, end in runs):
            continue
        anchor = _visible(above.split()[-1])
        if anchor:
            found[(anchor, tuple(end - start for start, end in runs))] += 1
    return found


def strip_inline_underlines(text: str, raw_text: str) -> UnderlineStripResult:
    """Remove the dash groups in ``text`` that ``raw_text`` shows are underlines."""
    result = UnderlineStripResult(text=text)
    if not text or "--" not in text:
        return result
    budget = raw_underlines(raw_text)
    if not budget:
        return result
    out_lines: list[str] = []
    changed = False
    for line in text.split("\n"):
        if "--" not in line:
            out_lines.append(line)
            continue
        tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", line)]
        is_dash = [_DASH_RUN_RE.fullmatch(tok) is not None and len(tok) >= _MIN_RUN for _, _, tok in tokens]
        if all(is_dash):
            out_lines.append(line)  # a whole dash line: edgar_sgml's rule-line strip decides
            continue
        cuts: list[tuple[int, int]] = []
        k = 0
        while k < len(tokens):
            if not is_dash[k]:
                k += 1
                continue
            j = k
            while j < len(tokens) and is_dash[j]:
                j += 1
            anchor = ""
            for b in range(k - 1, -1, -1):
                anchor = _visible(tokens[b][2])
                if anchor:
                    break
            key = (anchor, tuple(len(tokens[t][2]) for t in range(k, j)))
            if anchor and budget[key] > 0:
                budget[key] -= 1
                cuts.append((tokens[k][0], tokens[j - 1][1]))
                result.removed.append(line[tokens[k][0]:tokens[j - 1][1]])
            k = j
        if not cuts:
            out_lines.append(line)
            continue
        changed = True
        new = line
        for start, end in reversed(cuts):
            left, right = new[:start], new[end:]
            if left.endswith((" ", "\t")) and right.startswith((" ", "\t")):
                right = right.lstrip(" \t")
                left = left.rstrip(" \t") + " "
            elif not right.strip():
                left = left.rstrip(" \t")
            elif not left.strip():
                right = right.lstrip(" \t")
            new = left + right
        out_lines.append(new)
    if changed:
        result.text = "\n".join(out_lines)
    return result
