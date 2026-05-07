import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LegalAcceptancePrompt } from "@/components/auth/LegalAcceptancePrompt";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { useAuth } from "@/hooks/use-auth";
import {
  completeZitadelWebsiteAuth,
  finalizeZitadelWebsiteAuth,
  type McpTokenResult,
} from "@/lib/auth-api";
import { setSessionToken } from "@/lib/auth-session";
import { authSessionTransport } from "@/lib/auth-transport";
import { navigateToNextPath, safeNextPath } from "@/lib/auth-next";
import { navigateToMcpTokenResult } from "@/lib/mcp-token-result";
import {
  AUTH_WAKEUP_MESSAGE,
  isAuthWakeupError,
  prewarmAuthBackend,
  shouldHandleAuthWakeupMessage,
  withAuthWakeRetry,
} from "@/lib/auth-wake";

type CallbackState =
  | { kind: "working" }
  | { kind: "error"; message: string }
  | { kind: "legal"; email: string; nextPath: string };

export default function AuthZitadelCallback() {
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [state, setState] = useState<CallbackState>({ kind: "working" });
  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);
  const [submittingLegal, setSubmittingLegal] = useState(false);
  const [wakePending, setWakePending] = useState(false);

  const search = useMemo(() => window.location.search, []);

  useEffect(() => {
    let cancelled = false;

    const finishSession = async (payload: {
      next_path: string;
      session_token?: string;
    }) => {
      if (authSessionTransport() === "bearer") {
        if (!payload.session_token) {
          throw new Error("Missing session token.");
        }
        setSessionToken(payload.session_token);
      }
      await refresh();
      if (!cancelled) {
        navigateToNextPath(navigate, payload.next_path, { replace: true });
      }
    };

    const finishMcpToken = async (payload: McpTokenResult) => {
      if (!cancelled) {
        navigateToMcpTokenResult(navigate, payload);
      }
    };

    const handleCallback = async () => {
      try {
        await prewarmAuthBackend();
        const params = new URLSearchParams(search);
        const oauthError = params.get("error");
        const oauthDescription = params.get("error_description");
        if (oauthError) {
          throw new Error(oauthDescription ?? oauthError);
        }
        const code = params.get("code");
        const callbackState = params.get("state");
        const intentId =
          params.get("intentId") ??
          params.get("intentID") ??
          params.get("intent_id") ??
          params.get("id");
        const intentToken = params.get("token") ?? params.get("intent_token");
        const userId =
          params.get("userId") ?? params.get("userID") ?? params.get("user_id");
        const hasOAuthCode = Boolean(code && callbackState);
        const hasIntent = Boolean(intentId && intentToken);
        if (!hasOAuthCode && !hasIntent) {
          throw new Error("Missing auth provider authorization response.");
        }

        window.history.replaceState(
          null,
          document.title,
          window.location.pathname,
        );

        const result = await withAuthWakeRetry(async () => {
          try {
            return hasIntent
              ? await completeZitadelWebsiteAuth({
                  intent_id: intentId!,
                  intent_token: intentToken!,
                  ...(userId ? { user_id: userId } : {}),
                })
              : await completeZitadelWebsiteAuth({
                  code: code!,
                  state: callbackState!,
                });
          } catch (error) {
            if (isAuthWakeupError(error)) {
              setWakePending(true);
            }
            throw error;
          }
        });
        if (cancelled) return;
        setWakePending(false);
        if (result.status === "mcp_token") {
          await finishMcpToken(result);
          return;
        }
        if (result.status === "legal_required") {
          setState({
            kind: "legal",
            email: result.user.email,
            nextPath: safeNextPath(result.next_path),
          });
          return;
        }
        await finishSession(result);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        setWakePending(false);
        setState({ kind: "error", message });
      }
    };

    void handleCallback();
    return () => {
      cancelled = true;
    };
  }, [navigate, refresh, search]);

  const submitLegal = async () => {
    if (state.kind !== "legal" || !legalCheckedAtMs) return;
    setSubmittingLegal(true);
    try {
      const result = await withAuthWakeRetry(async () => {
        try {
          return await finalizeZitadelWebsiteAuth({
            checked_at_ms: legalCheckedAtMs,
            docs: ["tos", "privacy", "license"],
          });
        } catch (error) {
          if (isAuthWakeupError(error)) {
            setWakePending(true);
          }
          throw error;
        }
      });
      if (result.status !== "authenticated") {
        throw new Error("Could not finish sign-in.");
      }
      setWakePending(false);
      if (authSessionTransport() === "bearer") {
        if (!result.session_token) {
          throw new Error("Missing session token.");
        }
        setSessionToken(result.session_token);
      }
      await refresh();
      navigateToNextPath(navigate, result.next_path, { replace: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setWakePending(false);
      setState({ kind: "error", message });
    } finally {
      setSubmittingLegal(false);
    }
  };

  return (
    <PageShell
      title="Signing in…"
      subtitle="Completing your account sign-in."
      size="md"
    >
      {state.kind === "working" ? (
        <Card className="p-6">
          <div className="flex items-center gap-3 text-sm text-muted-foreground" role="status" aria-live="polite">
            <LoadingSpinner size="md" aria-label="Signing in" />
            {wakePending ? AUTH_WAKEUP_MESSAGE : "Finishing sign-in…"}
          </div>
        </Card>
      ) : state.kind === "error" ? (
        <Card className="p-6">
          <div className="grid gap-4" role="alert">
            <div>
              <div className="text-sm font-medium">Could not sign in</div>
              <div className="mt-1 text-sm text-muted-foreground">
                {shouldHandleAuthWakeupMessage(state.message) ? AUTH_WAKEUP_MESSAGE : state.message}
              </div>
            </div>
            <Button onClick={() => navigate("/login", { replace: true })}>
              Back to sign in
            </Button>
          </div>
        </Card>
      ) : (
        <LegalAcceptancePrompt
          email={state.email}
          checked={legalAccepted}
          disabled={submittingLegal}
          onCheckedChange={(checked) => {
            setLegalAccepted(checked);
            setLegalCheckedAtMs(checked ? Date.now() : null);
          }}
          onSubmit={() => void submitLegal()}
          submitLabel={submittingLegal ? "Finishing sign-in…" : "Continue"}
        />
      )}
    </PageShell>
  );
}
