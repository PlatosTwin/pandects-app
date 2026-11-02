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
* Some filings type a whole underline line a few columns right of its words
  ("Board Actions." underlined from its third letter), so the line may be
  moved left by one common shift of up to six columns before the check.
* An underlined phrase that wraps leaves a short run at column 0 of the next
  underline ("-    ------------" under "A or Schedule A-1"). Such a run of one
  or two dashes counts when the underline before it reached the end of its
  line. Two-dash runs elsewhere count only in pairs, each exactly under a
  word ("TE" and "PP"), since one alone may be a fraction bar.
* A lone word that wrapped onto its own line between the underlined line and
  its underline ("Each of" / "GPU" / dashes) is skipped over, and the word
  is the anchor.
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
# Columns a whole underline line may sit to the right of the words it underlines.
_MAX_SHIFT = 6
_WIDE_GAP = "   "


@dataclass
class UnderlineStripResult:
    text: str
    removed: list[str] = field(default_factory=list)


def _visible(token: str) -> str:
    return _TAG_RE.sub("", token)


def _is_dash_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not (set(stripped) - {"-", " "})


def _is_underline_line(line: str) -> bool:
    if not _is_dash_line(line):
        return False
    runs = line.split()
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


def _runs_are_under_text(above: str, runs: list[tuple[int, int]], max_shift: int) -> bool:
    """Every run underlines printed words once the whole line is moved left by one
    common shift of 0..max_shift columns. Some filings type every underline a few
    columns right of its words (0415d686, a6a4e945: five columns)."""
    if all(_run_is_under_text(above, start, end) for start, end in runs):
        return True
    # A shifted line must underline running words: a wide gap under it means a
    # table rule under a row of figures, which stays.
    return any(
        all(
            _run_is_under_text(above, start - shift, end - shift)
            and _WIDE_GAP not in above[max(start - shift, 0) : end - shift].strip()
            for start, end in runs
        )
        for shift in range(1, max_shift + 1)
    )


def _continues_wrapped_underline(lines: list[str], index: int) -> bool:
    """The underline on the line before ``lines[index - 1]`` runs to the end of its
    text line, so the underlined phrase wraps and ``lines[index]`` may start with the
    rest of it: "... described in Schedule" / "--------" / "A or Schedule A-1 ..." /
    "-    ------------" (2b36bf00)."""
    if index < 3 or not _is_underline_line(lines[index - 2]):
        return False
    text_line = lines[index - 3].rstrip()
    last_run_end = len(lines[index - 2].rstrip())
    return bool(text_line) and last_run_end >= len(text_line) - 1


def _underline_runs(lines: list[str], index: int) -> list[tuple[int, int]] | None:
    """The dash runs of ``lines[index]`` when it underlines ``lines[index - 1]``."""
    line, above = lines[index], lines[index - 1]
    if not _is_dash_line(line) or not above.strip() or _is_dash_line(above):
        return None
    runs = [(m.start(), m.end()) for m in _DASH_RUN_RE.finditer(line)]
    lengths = [end - start for start, end in runs]
    # A wrapped underline ends at the start of the next line: a short run at
    # column 0 under the first printed character, when the line above was
    # underlined to its end.
    head: list[tuple[int, int]] = []
    if runs[0][0] == 0 and lengths[0] < _MIN_LONGEST_RUN and _printed(above, 0):
        if _continues_wrapped_underline(lines, index):
            head, runs, lengths = runs[:1], runs[1:], lengths[1:]
        elif lengths[0] < _MIN_RUN:
            return None
    if not runs:
        # A lone "-" could be anything; a lone "--" ends the wrapped underline.
        return head if head and head[0][1] >= _MIN_RUN else None
    if any(length < _MIN_RUN for length in lengths):
        return None
    if max(lengths) >= _MIN_LONGEST_RUN:
        max_shift = _MAX_SHIFT
    elif len(runs) >= 2 and all(_is_whole_word_underline(above, start, end) for start, end in runs):
        # Two-letter abbreviations underlined in place ("TE" and "PP"). One short
        # run alone could be a fraction bar, so it needs company and exact words.
        max_shift = 0
    else:
        return None
    if not _runs_are_under_text(above, runs, max_shift):
        return None
    return head + runs


def _is_whole_word_underline(above: str, start: int, end: int) -> bool:
    word = above[start:end]
    if " " in word or not word.isalnum():
        return False
    before = above[start - 1] if start > 0 else " "
    after = above[end] if end < len(above) else " "
    return (before == " " or before in _OPENERS) and not after.isalnum()


def raw_underlines(raw_text: str) -> Counter[tuple[str, tuple[int, ...]]]:
    """(last word of the underlined line, run lengths) for each underline in the raw text."""
    found: Counter[tuple[str, tuple[int, ...]]] = Counter()
    if not raw_text or "-" not in raw_text:
        return found
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(8).split("\n")
    for index in range(1, len(lines)):
        line = lines[index]
        if not _is_dash_line(line):
            continue
        runs = _underline_runs(lines, index)
        anchor_line = lines[index - 1]
        if runs is None:
            runs = _underline_under_stray_word(lines, index)
        if runs is None:
            continue
        anchor = _visible(anchor_line.split()[-1])
        if anchor:
            found[(anchor, tuple(end - start for start, end in runs))] += 1
    return found


def _underline_under_stray_word(lines: list[str], index: int) -> list[tuple[int, int]] | None:
    """A single word wrapped onto its own line between an underlined line and its
    underline (207ad8d9: "(a)  Compliance.  Each of" / "GPU" / the underline).
    The runs must underline the line two above, and the stray word must start at
    column 0, end before the first run, and not be a label such as "By:"."""
    if index < 2:
        return None
    stray = lines[index - 1]
    words = stray.split()
    if len(words) != 1 or stray[0] == " " or not words[0].isalnum():
        return None
    first_run = _DASH_RUN_RE.search(lines[index])
    if first_run is None or len(stray.rstrip()) >= first_run.start():
        return None
    window = lines[max(0, index - 4) : index - 1] + [lines[index]]
    return _underline_runs(window, len(window) - 1)


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
        # A lone "-" counts too: it is the tail of a wrapped underline
        # ("Cody Energy or - ------------ any"). The group must still match a raw
        # underline exactly, so a hyphen used as a dash in prose stays.
        is_dash = [_DASH_RUN_RE.fullmatch(tok) is not None for _, _, tok in tokens]
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
