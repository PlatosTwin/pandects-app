import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { trackEvent } from "@/lib/analytics";
import { apiGet, errorMessage, isApiError } from "@/lib/api/client";
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

async function fetchNaics(signal?: AbortSignal): Promise<NaicsResponse> {
  try {
    return await apiGet<NaicsResponse>("v1/naics", undefined, { signal });
  } catch (error) {
    if (isApiError(error) && !error.isNetworkError) {
      trackEvent("api_error", {
        endpoint: "v1/naics",
        status: error.status,
        status_text: error.statusText,
      });
    }
    throw error;
  }
}

export function useNaics(): UseNaicsReturn {
  const query = useQuery({
    queryKey: keys.naics.all,
    queryFn: ({ signal }) => fetchNaics(signal),
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
    error: query.error ? errorMessage(query.error, "Unable to load NAICS labels.") : null,
  };
}
