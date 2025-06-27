import { useState, useCallback } from "react";
<<<<<<< HEAD
import { Agreement } from "@shared/agreement";
=======
import { Agreement, AgreementResponse } from "@shared/agreement";
>>>>>>> b67e1464ae3dac5ea6912901280ebad2df92dbdd

export function useAgreement() {
  const [agreement, setAgreement] = useState<Agreement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAgreement = useCallback(async (agreementUuid: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:5000/api/agreements/${agreementUuid}`,
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

<<<<<<< HEAD
      const data: Agreement = await response.json();
=======
      const data: AgreementResponse = await response.json();
      console.log(data.agreement)
>>>>>>> b67e1464ae3dac5ea6912901280ebad2df92dbdd
      setAgreement(data);
    } catch (err) {
      console.error("Failed to fetch agreement:", err);
      setError(err instanceof Error ? err.message : "Failed to load agreement");
    } finally {
      setIsLoading(false);
    }
  }, []);

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
