export interface PhraseFrequency {
  phrase: string;
  clauseCount: number;
  totalOccurrences: number;
  wordCount: number;
}

export interface NgramOptions {
  minN: number;
  maxN: number;
  minClauseCount: number;
  topK: number;
}

const DEFAULT_OPTIONS: NgramOptions = {
  minN: 3,
  maxN: 6,
  minClauseCount: 2,
  topK: 40,
};

const STOP_BOUNDARY = new Set([
  "the",
  "a",
  "an",
  "of",
  "to",
  "in",
  "on",
  "at",
  "for",
  "and",
  "or",
  "but",
  "with",
  "by",
  "as",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "that",
  "this",
  "these",
  "those",
  "it",
  "its",
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

function shouldKeepPhrase(tokens: string[]): boolean {
  const first = tokens[0];
  const last = tokens[tokens.length - 1];
  if (!first || !last) return false;
  if (STOP_BOUNDARY.has(first) || STOP_BOUNDARY.has(last)) return false;
  return true;
}

export function extractNgrams(
  texts: string[],
  options: Partial<NgramOptions> = {},
): PhraseFrequency[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  type Stats = { clauseIds: Set<number>; totalOccurrences: number; wordCount: number };
  const stats = new Map<string, Stats>();

  texts.forEach((text, clauseIdx) => {
    const tokens = tokenize(text);
    const seenInClause = new Set<string>();
    for (let n = opts.minN; n <= opts.maxN; n++) {
      if (tokens.length < n) continue;
      for (let i = 0; i + n <= tokens.length; i++) {
        const gram = tokens.slice(i, i + n);
        if (!shouldKeepPhrase(gram)) continue;
        const phrase = gram.join(" ");
        let entry = stats.get(phrase);
        if (!entry) {
          entry = { clauseIds: new Set(), totalOccurrences: 0, wordCount: n };
          stats.set(phrase, entry);
        }
        entry.totalOccurrences += 1;
        if (!seenInClause.has(phrase)) {
          entry.clauseIds.add(clauseIdx);
          seenInClause.add(phrase);
        }
      }
    }
  });

  const frequencies: PhraseFrequency[] = [];
  stats.forEach((entry, phrase) => {
    if (entry.clauseIds.size >= opts.minClauseCount) {
      frequencies.push({
        phrase,
        clauseCount: entry.clauseIds.size,
        totalOccurrences: entry.totalOccurrences,
        wordCount: entry.wordCount,
      });
    }
  });

  frequencies.sort((a, b) => {
    if (b.clauseCount !== a.clauseCount) return b.clauseCount - a.clauseCount;
    if (b.wordCount !== a.wordCount) return b.wordCount - a.wordCount;
    return b.totalOccurrences - a.totalOccurrences;
  });

  const dedupedByLonger: PhraseFrequency[] = [];
  for (const candidate of frequencies) {
    const containedInExisting = dedupedByLonger.some(
      (kept) =>
        kept.wordCount > candidate.wordCount &&
        kept.clauseCount === candidate.clauseCount &&
        kept.phrase.includes(candidate.phrase),
    );
    if (!containedInExisting) dedupedByLonger.push(candidate);
  }

  return dedupedByLonger.slice(0, opts.topK);
}

export function highlightPhrase(text: string, phrase: string): Array<{ text: string; match: boolean }> {
  if (!phrase) return [{ text, match: false }];
  const pattern = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  const regex = new RegExp(pattern, "gi");
  const parts: Array<{ text: string; match: boolean }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ text: text.slice(lastIndex, match.index), match: false });
    }
    parts.push({ text: match[0], match: true });
    lastIndex = match.index + match[0].length;
    if (match[0].length === 0) regex.lastIndex += 1;
  }
  if (lastIndex < text.length) {
    parts.push({ text: text.slice(lastIndex), match: false });
  }
  return parts;
}
