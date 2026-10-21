import { describe, expect, it } from "vitest";

import { contextLabels, favoriteHref } from "@/components/favorites/helpers";
import { parseAgreementOrigin } from "@/lib/agreement-links";
import type { Favorite } from "@/lib/favorites-api";

const base = { id: "f-1", agreement_uuid: "agr-1" } as Favorite;

describe("favoriteHref", () => {
  it("links a section favorite to its section, with Favorites as the origin", () => {
    const href = favoriteHref({ ...base, item_type: "section", item_uuid: "sec-1" });
    const url = new URL(href ?? "", "http://x");
    expect(url.pathname).toBe("/agreements/agr-1");
    expect(url.searchParams.get("focusSectionUuid")).toBe("sec-1");
    expect(parseAgreementOrigin(url.searchParams.get("from"))?.label).toBe("Back to Favorites");
  });

  it("links an agreement favorite without a focus section", () => {
    const href = favoriteHref({ ...base, item_type: "agreement", item_uuid: "agr-1" });
    expect(href).toBe("/agreements/agr-1?from=%2Ffavorites");
  });

  it("returns null without an agreement", () => {
    expect(
      favoriteHref({ ...base, agreement_uuid: null, item_type: "section", item_uuid: "s" } as Favorite),
    ).toBeNull();
  });
});

describe("contextLabels", () => {
  it("returns the string labels saved in the favorite's context", () => {
    const fav = {
      ...base,
      context: { standard_id_labels: { "802": "Definitions", "803": 7, "804": " " } },
    } as Favorite;
    expect(contextLabels(fav, "standard_id_labels")).toEqual({ "802": "Definitions" });
  });

  it("returns an empty map when the context has no label map", () => {
    expect(contextLabels({ ...base, context: null } as Favorite, "standard_id_labels")).toEqual({});
    expect(
      contextLabels({ ...base, context: { standard_id_labels: ["802"] } } as Favorite, "standard_id_labels"),
    ).toEqual({});
  });
});
