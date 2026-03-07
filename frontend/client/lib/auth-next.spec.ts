import { describe, expect, it } from "vitest";
import { buildAccountPathWithNext, safeNextPath } from "./auth-next";

describe("safeNextPath", () => {
  it("defaults to /account when value is missing", () => {
    expect(safeNextPath(null)).toBe("/account");
    expect(safeNextPath(undefined)).toBe("/account");
  });

  it("rejects non-relative and protocol-relative values", () => {
    expect(safeNextPath("https://example.com")).toBe("/account");
    expect(safeNextPath("//evil.example/path")).toBe("/account");
  });

  it("accepts safe relative paths", () => {
    expect(safeNextPath("/sections?query=test")).toBe("/sections?query=test");
  });
});

describe("buildAccountPathWithNext", () => {
  it("omits next when destination is /account", () => {
    expect(buildAccountPathWithNext("/account")).toBe("/account");
  });

  it("encodes next path for account route", () => {
    expect(buildAccountPathWithNext("/sections?query=test")).toBe(
      "/account?next=%2Fsections%3Fquery%3Dtest",
    );
  });
});
