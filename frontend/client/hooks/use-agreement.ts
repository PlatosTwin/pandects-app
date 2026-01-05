import { useState, useCallback } from "react";
import { Agreement } from "@shared/agreement";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";

export function useAgreement() {
  const [agreement, setAgreement] = useState<Agreement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAgreement = useCallback(
    async (agreementUuid: string, focusSectionUuid?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (focusSectionUuid) {
        params.set("focusSectionUuid", focusSectionUuid);
        params.set("neighborSections", "1");
      }
      const suffix = params.toString() ? `?${params.toString()}` : "";
      const response = await authFetch(
        apiUrl(`v1/agreements/${agreementUuid}${suffix}`),
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: Agreement = await response.json();
      setAgreement(data);
    } catch (err) {
      if (import.meta.env.DEV) {
        console.error("Failed to fetch agreement:", err);
      }
      setError(err instanceof Error ? err.message : "Failed to load agreement");
    } finally {
      setIsLoading(false);
    }
    },
    [],
  );

  const clearAgreement = useCallback(() => {
    setAgreement(null);
    setError(null);
  }, []);

  return {
    agreement,
    isLoading,
    error,
    fetchAgreement,
    clearAgreement,
  };
}
