import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { readSessionCache, writeSessionCache } from "@/lib/session-cache";

export type AgreementSummary = {
  agreements: number;
  sections: number;
  pages: number;
  latest_filing_date: string | null;
};

const AGREEMENT_SUMMARY_CACHE_KEY = "agreement-index-summary:v2";
const AGREEMENT_SUMMARY_CACHE_TTL_MS = 5 * 60 * 1000;

export function useAgreementSummary() {
  const [summary, setSummary] = useState<AgreementSummary | null>(() =>
    readSessionCache<AgreementSummary>(
      AGREEMENT_SUMMARY_CACHE_KEY,
      AGREEMENT_SUMMARY_CACHE_TTL_MS,
    ),
  );
  const [isLoading, setIsLoading] = useState(summary === null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (summary !== null) return;

    let cancelled = false;
    let timeoutId: number | null = null;
    let idleId: number | null = null;

    const fetchSummary = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const res = await authFetch(apiUrl("v1/agreements-summary"));
        if (!res.ok) {
          throw new Error(`Summary request failed (${res.status})`);
        }
        const data = (await res.json()) as AgreementSummary;
        if (!cancelled) {
          setSummary(data);
          writeSessionCache(AGREEMENT_SUMMARY_CACHE_KEY, data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load agreement summary.",
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    const scheduleFetch = () => {
      if (!cancelled) {
        void fetchSummary();
      }
    };

    const browserWindow = typeof window !== "undefined" ? window : null;

    if (browserWindow && "requestIdleCallback" in browserWindow) {
      idleId = browserWindow.requestIdleCallback(scheduleFetch, {
        timeout: 1500,
      });
    } else {
      timeoutId = window.setTimeout(scheduleFetch, 800);
    }

    return () => {
      cancelled = true;
      if (
        idleId !== null &&
        browserWindow &&
        "cancelIdleCallback" in browserWindow
      ) {
        browserWindow.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [summary]);

  return {
    summary,
    isLoading,
    error,
  };
}
