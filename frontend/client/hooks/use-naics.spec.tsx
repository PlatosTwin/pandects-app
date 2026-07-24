// @vitest-environment jsdom
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { NaicsResponse } from "@/lib/naics";
import {
  buildNaicsLabelIndex,
  formatNaicsIndustry,
  lookupNaicsLabel,
} from "@/lib/naics";

vi.mock("@/lib/auth-fetch", () => ({
  authFetch: vi.fn(),
}));
vi.mock("@/lib/analytics", () => ({
  trackEvent: vi.fn(),
}));

import { authFetch } from "@/lib/auth-fetch";
import { useNaics } from "@/hooks/use-naics";

const mockAuthFetch = vi.mocked(authFetch);

const NAICS_RESPONSE: NaicsResponse = {
  sectors: [
    {
      sector_code: "62",
      sector_desc: "Health Care and Social Assistance",
      sector_group: "Services",
      super_sector: "Services-Providing",
      sub_sectors: [
        {
          sub_sector_code: "621",
          sub_sector_desc: "Ambulatory Health Care Services",
        },
      ],
    },
    {
      sector_code: "31",
      sector_desc: "Manufacturing",
      sector_group: "Goods",
      super_sector: "Goods-Producing",
      sub_sectors: [],
    },
  ],
};

function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => body,
  } as Response;
}

function errorResponse(status: number, statusText: string): Response {
  return {
    ok: false,
    status,
    statusText,
    json: async () => ({}),
  } as Response;
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { refetchOnWindowFocus: false, retry: false },
    },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(
      QueryClientProvider,
      { client: queryClient },
      children,
    );
  };
}

beforeEach(() => {
  mockAuthFetch.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useNaics", () => {
  it("fetches /v1/naics once and exposes sector and subsector labels", async () => {
    mockAuthFetch.mockResolvedValue(okResponse(NAICS_RESPONSE));
    const { result } = renderHook(() => useNaics(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(mockAuthFetch).toHaveBeenCalledTimes(1);
    expect(String(mockAuthFetch.mock.calls[0]?.[0])).toContain("v1/naics");
    expect(result.current.error).toBeNull();
    expect(result.current.labelByCode["62"]).toBe(
      "Health Care and Social Assistance",
    );
    expect(result.current.labelByCode["621"]).toBe(
      "Ambulatory Health Care Services",
    );
  });

  it("exposes an empty lookup and the error when the fetch fails", async () => {
    mockAuthFetch.mockResolvedValue(errorResponse(500, "Internal Server Error"));
    const { result } = renderHook(() => useNaics(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.error).not.toBeNull());

    expect(result.current.labelByCode).toEqual({});
    // A numeric code must never surface raw when the lookup is unavailable.
    expect(formatNaicsIndustry(result.current.labelByCode, "62")).toBeNull();
  });
});

describe("NAICS lookup helpers", () => {
  const index = buildNaicsLabelIndex(NAICS_RESPONSE);

  it("resolves exact sector and subsector codes", () => {
    expect(lookupNaicsLabel(index, "62")).toBe(
      "Health Care and Social Assistance",
    );
    expect(lookupNaicsLabel(index, "621")).toBe(
      "Ambulatory Health Care Services",
    );
  });

  it("falls back from a longer code to its 2-digit sector label", () => {
    expect(lookupNaicsLabel(index, "6221")).toBe(
      "Health Care and Social Assistance",
    );
    expect(lookupNaicsLabel(index, "99")).toBeNull();
  });

  it("formats resolved codes as 'Label (code)'", () => {
    expect(formatNaicsIndustry(index, "62")).toBe(
      "Health Care and Social Assistance (62)",
    );
    expect(formatNaicsIndustry(index, " 621 ")).toBe(
      "Ambulatory Health Care Services (621)",
    );
  });

  it("returns null for unresolvable numeric codes and empty values", () => {
    expect(formatNaicsIndustry(index, "99")).toBeNull();
    expect(formatNaicsIndustry(index, "")).toBeNull();
    expect(formatNaicsIndustry(index, null)).toBeNull();
    expect(formatNaicsIndustry(index, undefined)).toBeNull();
  });

  it("passes non-numeric values through unchanged", () => {
    expect(formatNaicsIndustry(index, "Pharmaceuticals")).toBe(
      "Pharmaceuticals",
    );
  });
});
