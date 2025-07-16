import { useState, useEffect, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { apiUrl } from "@/lib/api-config";

export interface LLMOutputState {
  pageUuid: string;
  promptId: string;
  isLoading: boolean;
  isSaved: boolean;
  showSettings: boolean;
  fontSize: number;
  llmOutput: string;
  lastSaved: string;
  wordCount: number;
  charCount: number;
  showCopySuccess: boolean;
  copyError: boolean;
  showSaveConfirmation: boolean;
  showErrorModal: boolean;
  errorMessage: string;
  showInfoModal: boolean;
  infoMessage: string;
}

export function useLLMOutput() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [state, setState] = useState<LLMOutputState>({
    pageUuid: "",
    promptId: "",
    isLoading: false,
    isSaved: true,
    showSettings: false,
    fontSize: 16,
    llmOutput: "",
    lastSaved: "",
    wordCount: 0,
    charCount: 0,
    showCopySuccess: false,
    copyError: false,
    showSaveConfirmation: false,
    showErrorModal: false,
    errorMessage: "",
    showInfoModal: false,
    infoMessage: "",
  });

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem("llm-output-settings");
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setState((prev) => ({
          ...prev,
          fontSize: parsed.fontSize ?? 16,
          pageUuid: parsed.pageUuid ?? "",
          llmOutput: parsed.llmOutput ?? "",
        }));
      } catch (error) {
        console.error("Failed to load settings:", error);
      }
    }
  }, []);

  // Handle PRG pattern - check for success state in URL parameters
  useEffect(() => {
    const saved = searchParams.get("saved");
    const pageUuid = searchParams.get("pageUuid");
    const timestamp = searchParams.get("timestamp");

    if (saved === "true" && pageUuid && timestamp) {
      // Show success state from redirect
      setState((prev) => ({
        ...prev,
        pageUuid: pageUuid,
        isSaved: true,
        lastSaved: new Date(parseInt(timestamp)).toLocaleString(),
      }));

      // Clean up URL by removing the success parameters (optional)
      // This creates a clean URL for bookmarking and sharing
      const newParams = new URLSearchParams(searchParams);
      newParams.delete("saved");
      newParams.delete("pageUuid");
      newParams.delete("timestamp");

      // Replace current URL without the success parameters
      navigate(
        `/editor${newParams.toString() ? "?" + newParams.toString() : ""}`,
        {
          replace: true,
        },
      );
    }
  }, [searchParams, navigate]);

  // Save settings to localStorage whenever they change
  useEffect(() => {
    const settingsToSave = {
      fontSize: state.fontSize,
      pageUuid: state.pageUuid,
      llmOutput: state.llmOutput,
    };
    localStorage.setItem("llm-output-settings", JSON.stringify(settingsToSave));
  }, [state.fontSize, state.pageUuid, state.llmOutput]);

  const updateState = useCallback((updates: Partial<LLMOutputState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const calculateStats = useCallback(() => {
    const words = state.llmOutput.trim()
      ? state.llmOutput.trim().split(/\s+/).length
      : 0;
    const chars = state.llmOutput.length;

    updateState({
      wordCount: words,
      charCount: chars,
    });
  }, [state.llmOutput, updateState]);

  // Calculate stats whenever content changes
  useEffect(() => {
    calculateStats();
  }, [calculateStats]);

  const loadPage = useCallback(async () => {
    if (!state.pageUuid.trim()) return;

    updateState({ isLoading: true, lastSaved: "" });

    try {
      const res = await fetch(apiUrl(`api/llm/${state.pageUuid}`));

      // Check if the response is ok (status 200-299)
      if (!res.ok) {
        if (res.status === 404) {
          // UUID not found in database - this is informational, not an error
          updateState({
            showInfoModal: true,
            infoMessage: "No page with that UUID exists in mna.llm_output.",
          });
          return;
        }
        // Other HTTP errors
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const responseData = await res.json();

      // Check if the response data is empty or null
      if (!responseData || Object.keys(responseData).length === 0) {
        updateState({
          showInfoModal: true,
          infoMessage: "No page with that UUID exists in mna.llm_output.",
        });
        return;
      }

      const { promptId, llmOutput, llmOutputCorrected } = responseData;
      const output = llmOutputCorrected ?? llmOutput;

      updateState({
        promptId: promptId,
        llmOutput: output,
        isSaved: true,
      });
    } catch (error) {
      console.error("Failed to load page:", error);
      // Check if it's a network error
      if (error instanceof TypeError && error.message.includes("fetch")) {
        updateState({
          showErrorModal: true,
          errorMessage:
            "Network error: unable to reach the back end database. Check your connection and try again.",
        });
      } else {
        updateState({
          showErrorModal: true,
          errorMessage:
            "Network error: unable to reach the back end database. Check your connection and try again.",
        });
      }
    } finally {
      updateState({ isLoading: false });
    }
  }, [state.pageUuid, updateState]);

  const requestSave = useCallback(() => {
    if (!state.llmOutput.trim()) return;

    updateState({ showSaveConfirmation: true });
  }, [state.llmOutput, updateState]);

  const confirmSave = useCallback(async () => {
    updateState({ showSaveConfirmation: false });

    try {
      const res = await fetch(
        apiUrl(`api/llm/${state.pageUuid}/${state.promptId}`),
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            llmOutputCorrected: state.llmOutput,
          }),
        },
      );

      // Check if the response is ok (status 200-299)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      // POST/Redirect/GET pattern:
      // After successful save, redirect to the same page with success parameters
      // This prevents resubmission on browser refresh
      const timestamp = Date.now();
      const successUrl = `/editor?saved=true&pageUuid=${encodeURIComponent(state.pageUuid)}&timestamp=${timestamp}`;

      navigate(successUrl);
    } catch (error) {
      console.error("Failed to save page:", error);
      // Check if it's a network error
      if (error instanceof TypeError && error.message.includes("fetch")) {
        updateState({
          showErrorModal: true,
          errorMessage:
            "Network error: unable to reach the back end database. Check your connection and try again.",
        });
      } else {
        updateState({
          showErrorModal: true,
          errorMessage:
            "Network error: unable to reach the back end database. Check your connection and try again.",
        });
      }
    }
  }, [state.llmOutput, state.pageUuid, state.promptId, updateState, navigate]);

  const cancelSave = useCallback(() => {
    updateState({ showSaveConfirmation: false });
  }, [updateState]);

  const copyToClipboard = useCallback(async () => {
    if (!state.llmOutput.trim()) return;

    // Helper function for fallback copy method
    const fallbackCopy = () => {
      const textArea = document.createElement("textarea");
      textArea.value = state.llmOutput;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);
        return successful;
      } catch (err) {
        document.body.removeChild(textArea);
        throw err;
      }
    };

    try {
      // Try modern Clipboard API first, but catch any errors including permissions
      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(state.llmOutput);
        } catch (clipboardError) {
          // If modern API fails for any reason, use fallback
          const success = fallbackCopy();
          if (!success) {
            throw new Error("Both clipboard methods failed");
          }
        }
      } else {
        // Use fallback method directly
        const success = fallbackCopy();
        if (!success) {
          throw new Error("Fallback copy method failed");
        }
      }

      updateState({ showCopySuccess: true, copyError: false });

      // Hide success message after 3 seconds
      setTimeout(() => {
        updateState({ showCopySuccess: false });
      }, 3000);
    } catch (error) {
      console.error("Failed to copy to clipboard:", error);
      // Show error message to user
      updateState({ copyError: true, showCopySuccess: false });
      setTimeout(() => {
        updateState({ copyError: false });
      }, 3000);
    }
  }, [state.llmOutput, updateState]);

  const clearContent = useCallback(() => {
    updateState({
      llmOutput: "",
      isSaved: true,
      wordCount: 0,
      charCount: 0,
    });
  }, [updateState]);

  const increaseFontSize = useCallback(() => {
    updateState({ fontSize: Math.min(state.fontSize + 2, 24) });
  }, [state.fontSize, updateState]);

  const decreaseFontSize = useCallback(() => {
    updateState({ fontSize: Math.max(state.fontSize - 2, 12) });
  }, [state.fontSize, updateState]);

  const toggleSettings = useCallback(() => {
    updateState({ showSettings: !state.showSettings });
  }, [state.showSettings, updateState]);

  const updatePageUuid = useCallback(
    (value: string) => {
      updateState({ pageUuid: value });
    },
    [updateState],
  );

  const updateLLMOutput = useCallback(
    (value: string) => {
      updateState({
        llmOutput: value,
        isSaved: false,
      });
    },
    [updateState],
  );

  const closeErrorModal = useCallback(() => {
    updateState({
      showErrorModal: false,
      errorMessage: "",
    });
  }, [updateState]);

  const closeInfoModal = useCallback(() => {
    updateState({
      showInfoModal: false,
      infoMessage: "",
    });
  }, [updateState]);

  return {
    state,
    actions: {
      loadPage,
      requestSave,
      confirmSave,
      cancelSave,
      copyToClipboard,
      clearContent,
      increaseFontSize,
      decreaseFontSize,
      toggleSettings,
      updatePageUuid,
      updateLLMOutput,
      closeErrorModal,
      closeInfoModal,
    },
  };
}
