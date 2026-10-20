import { afterEach, describe, expect, it, vi } from "vitest";

const authFetch = vi.fn();
vi.mock("@/lib/auth-fetch", () => ({ authFetch: (url: string) => authFetch(url) }));
vi.mock("@/hooks/use-toast", () => ({ toast: vi.fn(() => ({ dismiss: vi.fn() })) }));

const { CSV_EXPORT_MAX_PAGES, exportAllPagesForCsv, fetchAllPagesForCsv } = await import(
  "./csv-export"
);

function page(results: number[], totalCount: number, hasNext: boolean) {
  return {
    ok: true,
    json: async () => ({ results, total_count: totalCount, has_next: hasNext }),
  };
}

describe("fetchAllPagesForCsv", () => {
  afterEach(() => authFetch.mockReset());

  it("pages through the submitted query until the last page", async () => {
    authFetch
      .mockResolvedValueOnce(page([1, 2], 3, true))
      .mockResolvedValueOnce(page([3], 3, false));

    const out = await fetchAllPagesForCsv<number>("v1/sections", "year=2024&page=7&page_size=25");

    expect(out).toEqual({
      rows: [1, 2, 3],
      totalCount: 3,
      totalCountIsApproximate: false,
      truncated: false,
    });
    const urls = authFetch.mock.calls.map(([url]) => new URL(url, "http://x"));
    expect(urls.map((u) => u.searchParams.get("page"))).toEqual(["1", "2"]);
    expect(urls.every((u) => u.searchParams.get("page_size") === "100")).toBe(true);
    expect(urls.every((u) => u.searchParams.get("year") === "2024")).toBe(true);
  });

  it("stops at the page cap and reports a partial export", async () => {
    authFetch.mockImplementation(async () => page([1], 999, true));

    const out = await fetchAllPagesForCsv<number>("v1/sections", "");

    expect(authFetch).toHaveBeenCalledTimes(CSV_EXPORT_MAX_PAGES);
    expect(out.truncated).toBe(true);
    expect(out.totalCount).toBe(999);
  });

  it("throws instead of silently exporting the current page", async () => {
    authFetch.mockResolvedValueOnce({ ok: false, status: 429, statusText: "Too Many" });

    await expect(fetchAllPagesForCsv("v1/sections", "")).rejects.toThrow("HTTP 429");
  });

  it("ignores a second export while one is running", async () => {
    let release: (value: unknown) => void = () => {};
    authFetch.mockImplementationOnce(
      () => new Promise((resolve) => (release = resolve)),
    );

    const first = exportAllPagesForCsv<number>("v1/sections", "");
    const second = await exportAllPagesForCsv<number>("v1/sections", "");
    release(page([1], 1, false));

    expect(second).toBeNull();
    expect((await first)?.rows).toEqual([1]);
    expect(authFetch).toHaveBeenCalledTimes(1);
  });
});
