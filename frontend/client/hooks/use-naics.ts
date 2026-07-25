import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import {
  buildNaicsLabelIndex,
  type NaicsLabelByCode,
  type NaicsResponse,
} from "@/lib/naics";

interface UseNaicsReturn {
  labelByCode: NaicsLabelByCode;
  isLoading: boolean;
  error: string | null;
}

const EMPTY_INDEX: NaicsLabelByCode = {};

async function fetchNaics(): Promise<NaicsResponse> {
  const response = await authFetch(apiUrl("v1/naics"));
  if (!response.ok) {
    trackEvent("api_error", {
      endpoint: "api/naics",
      status: response.status,
      status_text: response.statusText,
    });
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return (await response.json()) as NaicsResponse;
}

export function useNaics(): UseNaicsReturn {
  const query = useQuery({
    queryKey: keys.naics.all,
    queryFn: fetchNaics,
    enabled: !IS_SERVER_RENDER,
    // Static reference data: fetch once per session, keep it cached all day.
    staleTime: Infinity,
    gcTime: 24 * 60 * 60 * 1000,
  });

  const labelByCode = useMemo(
    () => (query.data ? buildNaicsLabelIndex(query.data) : EMPTY_INDEX),
    [query.data],
  );

  return {
    labelByCode,
    isLoading: query.isLoading,
    error: query.error ? (query.error as Error).message : null,
  };
}
