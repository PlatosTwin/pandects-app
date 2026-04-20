import { describe, it, expect } from "vitest";
import { extractNgrams, highlightPhrase, tokenize } from "./ngram-analysis";

describe("tokenize", () => {
  it("lowercases and splits on whitespace, drops punctuation", () => {
    expect(tokenize("The Parties' Agreement.")).toEqual(["the", "parties", "agreement"]);
  });
});

describe("extractNgrams", () => {
  it("returns phrases that appear in at least minClauseCount clauses", () => {
    const texts = [
      "The parties hereby agree to indemnify the seller for tax liabilities.",
      "The parties hereby agree to indemnify the buyer for tax matters.",
      "Nothing in common here.",
    ];
    const result = extractNgrams(texts, { minN: 3, maxN: 5, minClauseCount: 2, topK: 10 });
    expect(result.length).toBeGreaterThan(0);
    const phrases = result.map((r) => r.phrase);
    expect(phrases.some((p) => p.includes("parties hereby agree"))).toBe(true);
  });

  it("deduplicates shorter substrings with identical clauseCount", () => {
    const texts = [
      "material adverse effect on the company",
      "material adverse effect on the company",
    ];
    const result = extractNgrams(texts, { minN: 3, maxN: 6, minClauseCount: 2, topK: 10 });
    const phrases = result.map((r) => r.phrase);
    expect(phrases).toContain("material adverse effect on the company");
    expect(phrases).not.toContain("material adverse effect");
  });
});

describe("highlightPhrase", () => {
  it("splits text into matching and non-matching parts", () => {
    const parts = highlightPhrase("This is a tax indemnity clause", "tax indemnity");
    expect(parts).toHaveLength(3);
    expect(parts[1]).toEqual({ text: "tax indemnity", match: true });
  });

  it("returns whole string unchanged when phrase is empty", () => {
    const parts = highlightPhrase("hello", "");
    expect(parts).toEqual([{ text: "hello", match: false }]);
  });
});
