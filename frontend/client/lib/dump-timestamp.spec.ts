import { describe, expect, it } from "vitest";
import type { DumpInfo } from "@shared/insights";
import { organizeDumps } from "@/hooks/use-insights";
import { formatDumpTimestamp, parseDumpTimestamp } from "./dump-timestamp";

describe("dump timestamps", () => {
  it("parses dump versions as UTC", () => {
    expect(parseDumpTimestamp("public_2026-09-09_16-08-00")).toBe(Date.UTC(2026, 8, 9, 16, 8, 0));
    expect(parseDumpTimestamp("public_2026-09-09")).toBe(Date.UTC(2026, 8, 9));
    expect(parseDumpTimestamp("latest")).toBeNull();
    expect(formatDumpTimestamp("public_2026-09-09_16-08-00")).toBe("September 9, 2026");
    expect(formatDumpTimestamp("latest")).toBe("Latest");
  });

  it("orders latest first, then newest first, unknown formats last", () => {
    const dump = (timestamp: string) => ({ timestamp }) as DumpInfo;
    const { dumps } = organizeDumps([
      dump("public_2026-08-01_10-00-00"),
      dump("odd"),
      dump("latest"),
      dump("public_2026-09-09_16-08-00"),
      dump("public_2026-09-09_09-00-00"),
    ]);
    expect(dumps.map((d) => d.timestamp)).toEqual([
      "latest",
      "public_2026-09-09_16-08-00",
      "public_2026-09-09_09-00-00",
      "public_2026-08-01_10-00-00",
      "odd",
    ]);
  });
});
