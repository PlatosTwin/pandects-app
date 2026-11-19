import { beforeEach, describe, expect, it, vi } from "vitest";

const apiFetchJson = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/client", () => ({ apiFetchJson }));

import { FAVORITES_EXISTS_BATCH_SIZE, favoritesExists } from "@/lib/favorites-api";

// gunicorn's default limit for a request line.
const REQUEST_LINE_LIMIT = 4094;

describe("favoritesExists", () => {
  beforeEach(() => {
    apiFetchJson.mockReset();
  });

  it("makes no request for an empty list", async () => {
    expect(await favoritesExists("section", [])).toEqual({});
    expect(apiFetchJson).not.toHaveBeenCalled();
  });

  it("sends a long agreement's sections in batches and merges the answers", async () => {
    const uuids = Array.from(
      { length: FAVORITES_EXISTS_BATCH_SIZE * 2 + 5 },
      (_, i) => `17dbbed7-d85b-5266-b967-${String(i).padStart(12, "0")}`,
    );
    apiFetchJson.mockImplementation(async (url: string) => {
      const sent = new URL(url, "https://api.test").searchParams.get("item_uuids")!.split(",");
      return { favorites: { [sent[0]]: `fav-${sent[0].slice(-3)}` } };
    });

    const found = await favoritesExists("section", uuids);

    expect(apiFetchJson).toHaveBeenCalledTimes(3);
    const sent = apiFetchJson.mock.calls.map(([url]) => String(url));
    for (const url of sent) {
      expect(new URL(url, "https://api.test").pathname.length + new URL(url, "https://api.test").search.length)
        .toBeLessThan(REQUEST_LINE_LIMIT - 100);
    }
    const all = sent.flatMap((url) =>
      new URL(url, "https://api.test").searchParams.get("item_uuids")!.split(","),
    );
    expect(all).toEqual(uuids);
    expect(Object.keys(found)).toEqual([
      uuids[0],
      uuids[FAVORITES_EXISTS_BATCH_SIZE],
      uuids[FAVORITES_EXISTS_BATCH_SIZE * 2],
    ]);
  });
});
