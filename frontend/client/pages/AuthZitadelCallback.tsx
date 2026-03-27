import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { useAuth } from "@/hooks/use-auth";
import { linkExternalSubject } from "@/lib/auth-api";
import { clearZitadelLinkState, finishZitadelLinkFlow } from "@/lib/zitadel-link";

function safeReturnPath(value: string): string {
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

function withQueryParam(pathname: string, key: string, value: string): string {
  const [basePath, existingQuery] = pathname.split("?", 2);
  const params = new URLSearchParams(existingQuery ?? "");
  params.set(key, value);
  const query = params.toString();
  return query ? `${basePath}?${query}` : basePath;
}

export default function AuthZitadelCallback() {
  const navigate = useNavigate();
  const { status } = useAuth();
  const [error, setError] = useState<string | null>(null);

  const search = useMemo(() => window.location.search, []);

  useEffect(() => {
    if (status === "loading") return;

    let cancelled = false;
    const handleCallback = async () => {
      if (status !== "authenticated") {
        clearZitadelLinkState();
        setError("Sign in to Pandects before linking ZITADEL.");
        return;
      }

      try {
        const { accessToken, returnTo } = await finishZitadelLinkFlow(search);
        await linkExternalSubject({
          provider: "zitadel",
          access_token: accessToken,
        });
        if (cancelled) return;
        navigate(withQueryParam(safeReturnPath(returnTo), "mcpLinked", "1"), {
          replace: true,
        });
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        navigate(withQueryParam("/account", "mcpLinkError", message), {
          replace: true,
        });
      }
    };

    void handleCallback();
    return () => {
      cancelled = true;
    };
  }, [navigate, search, status]);

  return (
    <PageShell
      title="Linking ZITADEL…"
      subtitle="Completing your MCP access link."
      size="md"
    >
      <Card className="p-6">
        {error ? (
          <div className="grid gap-4" role="alert">
            <div>
              <div className="text-sm font-medium">ZITADEL linking failed</div>
              <div className="mt-1 text-sm text-muted-foreground">{error}</div>
            </div>
            <Button onClick={() => navigate("/account", { replace: true })}>
              Back to account
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-sm text-muted-foreground" role="status" aria-live="polite">
            <LoadingSpinner size="md" aria-label="Linking ZITADEL access" />
            Finishing ZITADEL link…
          </div>
        )}
      </Card>
    </PageShell>
  );
}
