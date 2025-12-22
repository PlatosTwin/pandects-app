import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { setSessionToken } from "@/lib/auth-session";

function safeNextPath(value: string | null): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

export default function AuthGoogleCallback() {
  const navigate = useNavigate();
  const { refresh } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);

  const params = useMemo(() => {
    return new URLSearchParams(window.location.hash.replace(/^#/, ""));
  }, []);

  useEffect(() => {
    const sessionToken = params.get("sessionToken");
    const next = safeNextPath(params.get("next"));
    const callbackError = params.get("error");

    window.history.replaceState(
      null,
      document.title,
      window.location.pathname + window.location.search,
    );

    if (callbackError) {
      setError(callbackError);
      setBusy(false);
      return;
    }

    if (!sessionToken) {
      setError("missing_session_token");
      setBusy(false);
      return;
    }

    setSessionToken(sessionToken);
    void refresh().finally(() => {
      navigate(next, { replace: true });
    });
  }, [navigate, params, refresh]);

  return (
    <PageShell title="Signing in…" subtitle="Completing Google sign-in." size="md">
      <Card className="p-6">
        {error ? (
          <div className="grid gap-4" role="alert">
            <div>
              <div className="text-sm font-medium">Google sign-in failed</div>
              <div className="mt-1 text-sm text-muted-foreground">
                Error: {error}
              </div>
            </div>
            <Button onClick={() => navigate("/account", { replace: true })}>
              Back to account
            </Button>
          </div>
        ) : busy ? (
          <div className="text-sm text-muted-foreground" role="status" aria-live="polite">
            Finishing sign-in…
          </div>
        ) : (
          <Button onClick={() => navigate("/account", { replace: true })}>
            Back to account
          </Button>
        )}
      </Card>
    </PageShell>
  );
}
