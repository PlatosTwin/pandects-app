import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { apiUrl } from "@/lib/api-config";
import { fetchOAuthConsentClient, grantOAuthConsent } from "@/lib/auth-api";
import {
  classifyOAuthClient,
  type OAuthClientIdentity,
} from "@/lib/oauth-client-identity";

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

export default function OAuthConsent() {
  const location = useLocation();
  const params = useMemo(
    () => new URLSearchParams(location.search),
    [location.search],
  );

  const clientId = params.get("client_id") ?? "";
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

  // Anyone can link to this page with any query string, so who the client is
  // comes from its registration on the server, not from the URL.
  const [client, setClient] = useState<
    | { status: "loading" }
    | { status: "error" }
    | { status: "ready"; name: string | null; identity: OAuthClientIdentity }
  >({ status: "loading" });

  useEffect(() => {
    if (!paramsValid) return;
    let cancelled = false;
    fetchOAuthConsentClient({ client_id: clientId, redirect_uri: redirectUri })
      .then((registration) => {
        if (cancelled) return;
        setClient({
          status: "ready",
          name: registration.client_name ?? null,
          identity: classifyOAuthClient(
            registration.client_name ?? null,
            redirectUri,
            registration.redirect_uris,
          ),
        });
      })
      .catch(() => {
        if (!cancelled) setClient({ status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [paramsValid, clientId, redirectUri]);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Both buttons go back through /v1/auth/oauth/authorize, which checks the
  // redirect URI against the client's registration before sending the browser
  // anywhere; this page's own query string is never trusted as a destination.
  const authorizeHref = (deny: boolean): string => {
    const authorizeUrl = new URL(apiUrl("v1/auth/oauth/authorize"));
    authorizeUrl.searchParams.set("client_id", clientId);
    authorizeUrl.searchParams.set("redirect_uri", redirectUri);
    authorizeUrl.searchParams.set("response_type", responseType);
    authorizeUrl.searchParams.set("scope", scope);
    authorizeUrl.searchParams.set("code_challenge", codeChallenge);
    authorizeUrl.searchParams.set("code_challenge_method", codeChallengeMethod);
    if (state) authorizeUrl.searchParams.set("state", state);
    if (deny) authorizeUrl.searchParams.set("deny", "1");
    return authorizeUrl.toString();
  };

  const handleAllow = async () => {
    if (!paramsValid) return;
    setSubmitting(true);
    setError(null);
    try {
      await grantOAuthConsent({
        client_id: clientId,
        redirect_uri: redirectUri,
        scope,
      });
      // The new grant means the gate passes and the backend 302s the browser
      // to the client's redirect_uri with the auth code.
      window.location.href = authorizeHref(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  const handleDeny = () => {
    if (!paramsValid) return;
    // RFC 6749 §4.1.2.1: the backend redirects to the registered callback with
    // error=access_denied and the original state.
    window.location.href = authorizeHref(true);
  };

  return (
    <PageShell
      title="Authorize access"
      description={
        client.status === "ready" && client.identity.kind === "known"
          ? `${client.identity.knownLabel} wants to access your Pandects account.`
          : "An external application wants to access your Pandects account."
      }
      layout="form"
    >
      <Card className="w-full p-6 shadow-sm sm:p-8">
        <div className="grid gap-6">
          {!paramsValid || client.status === "error" ? (
            <Alert variant="destructive">
              <AlertTitle>Invalid authorization request</AlertTitle>
              <AlertDescription>
                The authorization link is missing required information. Return
                to the application that sent you here and start the sign-in flow
                again.
              </AlertDescription>
            </Alert>
          ) : client.status === "loading" ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : (
            <ConsentDetails
              clientName={client.name}
              identity={client.identity}
              scopes={scopes}
              error={error}
              submitting={submitting}
              onAllow={handleAllow}
              onDeny={handleDeny}
            />
          )}
        </div>
      </Card>
    </PageShell>
  );
}

function ConsentDetails({
  clientName,
  identity,
  scopes,
  error,
  submitting,
  onAllow,
  onDeny,
}: {
  clientName: string | null;
  identity: OAuthClientIdentity;
  scopes: string[];
  error: string | null;
  submitting: boolean;
  onAllow: () => void;
  onDeny: () => void;
}) {
  return (
    <>
      <dl className="grid gap-3 text-sm">
        <div>
          <dt className="text-muted-foreground">
            After you allow, you will be sent to
          </dt>
          <dd className="mt-0.5 break-all font-mono text-base font-medium text-foreground">
            {identity.host}
            {identity.kind === "local" ? (
              <span className="ml-2 font-sans text-sm font-normal text-muted-foreground">
                (an app on this computer)
              </span>
            ) : null}
          </dd>
        </div>
        {identity.otherHosts.length > 0 ? (
          <div>
            <dt className="text-muted-foreground">
              Allowing also lets this application sign in through
            </dt>
            <dd className="mt-0.5 break-all font-mono text-foreground">
              {identity.otherHosts.join(", ")}
            </dd>
          </div>
        ) : null}
        {clientName ? (
          <div>
            <dt className="text-muted-foreground">
              Name the application gave itself
            </dt>
            <dd className="mt-0.5 break-words text-foreground">{clientName}</dd>
          </div>
        ) : null}
      </dl>
      {identity.kind === "impersonation" ? (
        <Alert variant="destructive">
          <AlertTitle>This may not be {identity.knownLabel}</AlertTitle>
          <AlertDescription>
            It calls itself {identity.knownLabel}, but {identity.knownLabel}{" "}
            doesn&apos;t use {identity.foreignHosts.join(", ")}. Deny unless you
            are sure.
          </AlertDescription>
        </Alert>
      ) : identity.kind === "unknown" ? (
        <Alert variant="warning">
          <AlertTitle>Unrecognized application</AlertTitle>
          <AlertDescription>
            Pandects doesn&apos;t recognize this application. Allow only if you
            just started connecting it yourself.
          </AlertDescription>
        </Alert>
      ) : null}
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
        <Button variant="outline" onClick={onDeny} disabled={submitting}>
          Deny
        </Button>
        <Button onClick={onAllow} disabled={submitting}>
          {submitting ? "Allowing…" : "Allow"}
        </Button>
      </div>
    </>
  );
}
