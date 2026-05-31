import { useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { apiUrl } from "@/lib/api-config";
import { grantOAuthConsent } from "@/lib/auth-api";

// Human-readable labels for the scope identifiers the MCP server publishes.
// Keep in sync with mcp_supported_scopes() on the backend.
const SCOPE_DESCRIPTIONS: Record<string, string> = {
  "sections:search": "Search sections in the Pandects database.",
  "agreements:search": "Search agreements in the Pandects database.",
  "agreements:read": "Read agreement metadata.",
  "agreements:read_fulltext": "Read full agreement text.",
};

function describeScope(scope: string): string {
  return SCOPE_DESCRIPTIONS[scope] ?? scope;
}

function isHttpOrHttps(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function buildClientCallbackUrl(redirectUri: string, params: Record<string, string>): string {
  const url = new URL(redirectUri);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  return url.toString();
}

export default function OAuthConsent() {
  const location = useLocation();
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);

  const clientId = params.get("client_id") ?? "";
  const clientName = params.get("client_name")?.trim() || null;
  const redirectUri = params.get("redirect_uri") ?? "";
  const responseType = params.get("response_type") ?? "";
  const scope = params.get("scope") ?? "";
  const state = params.get("state");
  const codeChallenge = params.get("code_challenge") ?? "";
  const codeChallengeMethod = params.get("code_challenge_method") ?? "";

  const scopes = useMemo(() => scope.split(" ").filter(Boolean), [scope]);

  const paramsValid =
    !!clientId &&
    !!redirectUri &&
    isHttpOrHttps(redirectUri) &&
    !!responseType &&
    scopes.length > 0 &&
    !!codeChallenge &&
    !!codeChallengeMethod;

  const redirectHost = useMemo(() => {
    try {
      return new URL(redirectUri).host;
    } catch {
      return redirectUri;
    }
  }, [redirectUri]);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAllow = async () => {
    if (!paramsValid) return;
    setSubmitting(true);
    setError(null);
    try {
      await grantOAuthConsent({ client_id: clientId, scope });
      // Re-navigate to /v1/auth/oauth/authorize. The new grant means the
      // gate passes and the backend 302s the browser to the client's
      // redirect_uri with the auth code.
      const authorizeUrl = new URL(apiUrl("v1/auth/oauth/authorize"));
      authorizeUrl.searchParams.set("client_id", clientId);
      authorizeUrl.searchParams.set("redirect_uri", redirectUri);
      authorizeUrl.searchParams.set("response_type", responseType);
      authorizeUrl.searchParams.set("scope", scope);
      authorizeUrl.searchParams.set("code_challenge", codeChallenge);
      authorizeUrl.searchParams.set("code_challenge_method", codeChallengeMethod);
      if (state) authorizeUrl.searchParams.set("state", state);
      window.location.href = authorizeUrl.toString();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  const handleDeny = () => {
    if (!paramsValid) return;
    // RFC 6749 §4.1.2.1: signal denial to the relying party by redirecting
    // back to its callback with error=access_denied and the original state.
    const errorParams: Record<string, string> = { error: "access_denied" };
    if (state) errorParams.state = state;
    window.location.href = buildClientCallbackUrl(redirectUri, errorParams);
  };

  return (
    <PageShell
      title="Authorize access"
      subtitle={
        clientName
          ? `${clientName} wants to access your Pandects account.`
          : "An external application wants to access your Pandects account."
      }
      size="md"
      className="max-w-3xl"
    >
      <Card className="mx-auto w-full max-w-2xl border-border bg-card/95 p-6 shadow-sm sm:p-8">
        <div className="grid gap-6">
          {!paramsValid ? (
            <Alert variant="destructive">
              <AlertTitle>Invalid authorization request</AlertTitle>
              <AlertDescription>
                The authorization link is missing required information. Return to the application that sent you here and start the sign-in flow again.
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <p className="text-sm text-muted-foreground">
                After allowing, you will be redirected to{" "}
                <span className="font-medium text-foreground">{redirectHost}</span>.
              </p>
              <div>
                <div className="text-sm font-medium">Requested permissions</div>
                <ul className="mt-2 space-y-1 pl-5 text-sm text-muted-foreground list-disc">
                  {scopes.map((s) => (
                    <li key={s}>
                      <span className="text-foreground">{describeScope(s)}</span>{" "}
                      <span className="text-muted-foreground">({s})</span>
                    </li>
                  ))}
                </ul>
              </div>
              {error ? (
                <Alert variant="destructive">
                  <AlertTitle>Could not record consent</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              ) : null}
              <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
                <Button variant="outline" onClick={handleDeny} disabled={submitting}>
                  Deny
                </Button>
                <Button onClick={handleAllow} disabled={submitting}>
                  {submitting ? "Allowing…" : "Allow"}
                </Button>
              </div>
            </>
          )}
        </div>
      </Card>
    </PageShell>
  );
}
