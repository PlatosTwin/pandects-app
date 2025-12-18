import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { toast } from "@/components/ui/use-toast";
import {
  createApiKey,
  deleteAccount,
  fetchUsage,
  listApiKeys,
  loginWithGoogleCredential,
  revokeApiKey,
} from "@/lib/auth-api";
import type { ApiKeySummary, UsageByDay } from "@/lib/auth-types";
import { loadGoogleIdentityServices } from "@/lib/google-identity";
import { setSessionToken } from "@/lib/auth-session";
import { apiUrl } from "@/lib/api-config";
import { authSessionTransport } from "@/lib/auth-transport";
import { Check, Copy } from "lucide-react";
import { trackEvent } from "@/lib/analytics";

function formatDate(value: string | null) {
  if (!value) return "—";
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? value : dt.toLocaleString();
}

export default function Account() {
  const { status, user, login, register, logout, refresh } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [googleStatus, setGoogleStatus] = useState<
    "loading" | "ready" | "unavailable"
  >("loading");
  const googleButtonRef = useRef<HTMLDivElement | null>(null);

  const [apiKeys, setApiKeys] = useState<ApiKeySummary[]>([]);
  const [usageByDay, setUsageByDay] = useState<UsageByDay[]>([]);
  const [usageTotal, setUsageTotal] = useState(0);

  const [newKeyName, setNewKeyName] = useState("");
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [copiedNewKey, setCopiedNewKey] = useState(false);

  const [legalAccepted, setLegalAccepted] = useState(false);
  const [legalCheckedAtMs, setLegalCheckedAtMs] = useState<number | null>(null);
  const [googlePendingCredential, setGooglePendingCredential] = useState<string | null>(null);
  const [googleNeedsLegal, setGoogleNeedsLegal] = useState(false);

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");

  const hasAnyKey = apiKeys.some((k) => !k.revokedAt);

  const loadAccountData = async () => {
    const [keys, usage] = await Promise.all([listApiKeys(), fetchUsage()]);
    setApiKeys(keys.keys);
    setUsageByDay(usage.byDay);
    setUsageTotal(usage.total);
  };

  useEffect(() => {
    if (!user) return;
    void loadAccountData().catch((err) => {
      toast({ title: "Failed to load account", description: String(err) });
    });
  }, [user]);

  useEffect(() => {
    if (!user) return;

    const refreshAccountData = () => {
      void loadAccountData().catch((err) => {
        toast({ title: "Failed to load account", description: String(err) });
      });
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") refreshAccountData();
    };

    window.addEventListener("focus", refreshAccountData);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("focus", refreshAccountData);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [user]);

  useEffect(() => {
    setCopiedNewKey(false);
  }, [revealedKey]);

  const redactedReminder = useMemo(() => {
    if (status !== "authenticated") return null;
    if (hasAnyKey) return null;
    return "Create an API key to use the API programmatically.";
  }, [status, hasAnyKey]);

  useEffect(() => {
    if (user) return;
    setGoogleStatus("loading");
    setGooglePendingCredential(null);
    setGoogleNeedsLegal(false);

    const resolveClientId = async (): Promise<string | null> => {
      const fromEnv = import.meta.env.VITE_GOOGLE_OAUTH_CLIENT_ID;
      if (typeof fromEnv === "string" && fromEnv.trim().length > 0) {
        return fromEnv.trim();
      }
      try {
        const res = await fetch(apiUrl("api/auth/google/client-id"));
        if (!res.ok) return null;
        const data = (await res.json()) as { clientId?: unknown };
        return typeof data.clientId === "string" && data.clientId.trim().length > 0
          ? data.clientId.trim()
          : null;
      } catch {
        return null;
      }
    };

    void resolveClientId()
      .then(async (clientId) => {
        if (!clientId) {
          setGoogleStatus("unavailable");
          return;
        }

        await loadGoogleIdentityServices();
        if (!googleButtonRef.current) {
          setGoogleStatus("unavailable");
          return;
        }
        if (!window.google?.accounts?.id) {
          setGoogleStatus("unavailable");
          return;
        }

        window.google.accounts.id.initialize({
          client_id: clientId,
          callback: async ({ credential }) => {
            setBusy(true);
            try {
              const res = await loginWithGoogleCredential(credential);
              if (authSessionTransport() === "bearer") {
                if (!res.sessionToken) throw new Error("Missing session token.");
                setSessionToken(res.sessionToken);
              }
              await refresh();
              toast({ title: "Signed in" });
            } catch (err) {
              const msg = String(err);
              if (msg.includes("legal_required")) {
                setGooglePendingCredential(credential);
                setGoogleNeedsLegal(true);
                toast({
                  title: "Agree to continue",
                  description: "Accept the Terms, Privacy Policy, and License to create your account.",
                });
              } else {
                toast({ title: "Google sign-in failed", description: msg });
              }
            } finally {
              setBusy(false);
            }
          },
          cancel_on_tap_outside: true,
        });

        googleButtonRef.current.replaceChildren();
        const width = Math.round(
          Math.min(360, Math.max(240, googleButtonRef.current.getBoundingClientRect().width)),
        );
        window.google.accounts.id.renderButton(googleButtonRef.current, {
          theme: "outline",
          size: "large",
          text: "continue_with",
          shape: "rectangular",
          width,
        });

        setGoogleStatus("ready");
      })
      .catch(() => setGoogleStatus("unavailable"));
  }, [refresh, user]);

  return (
    <PageShell
      title="Account"
      subtitle="Sign in to unlock full access, manage API keys, and view API usage."
      size="md"
      actions={
        user ? (
          <Button variant="outline" onClick={logout}>
            Sign out
          </Button>
        ) : null
      }
    >
      {status === "loading" ? (
        <Card className="p-6">Loading…</Card>
      ) : !user ? (
        <Card className="p-6">
          <div className="grid gap-6">
            <div className="flex justify-center pt-1">
              <div
                ref={googleButtonRef}
                className="min-h-[44px] w-full max-w-[360px]"
              />
            </div>
            {googleNeedsLegal ? (
              <div className="grid gap-3">
                <div className="flex items-start gap-3 rounded-lg border border-border/70 bg-muted/20 p-4 text-sm">
                  <Checkbox
                    id="legal-google"
                    checked={legalAccepted}
                    onCheckedChange={(next) => {
                      const isChecked = next === true;
                      setLegalAccepted(isChecked);
                      if (isChecked) {
                        const ts = Date.now();
                        setLegalCheckedAtMs(ts);
                        trackEvent("legal_consent_checked", {
                          context: "google_signup",
                          checked_at_ms: ts,
                        });
                      } else {
                        setLegalCheckedAtMs(null);
                      }
                    }}
                  />
                  <label htmlFor="legal-google" className="leading-relaxed">
                    I have read and agree to the{" "}
                    <Link
                      to="/terms"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "tos", context: "google_signup" })
                      }
                    >
                      Terms of Service
                    </Link>
                    ,{" "}
                    <Link
                      to="/privacy-policy"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "privacy", context: "google_signup" })
                      }
                    >
                      Privacy Policy
                    </Link>
                    , and{" "}
                    <Link
                      to="/license"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "license", context: "google_signup" })
                      }
                    >
                      License
                    </Link>
                    .
                  </label>
                </div>
                <Button
                  disabled={busy || !legalAccepted || !legalCheckedAtMs || !googlePendingCredential}
                  onClick={async () => {
                    if (!googlePendingCredential || !legalCheckedAtMs) return;
                    setBusy(true);
                    try {
                      trackEvent("legal_consent_submitted", {
                        context: "google_signup",
                        checked_at_ms: legalCheckedAtMs,
                        submitted_at_ms: Date.now(),
                      });
                      const res = await loginWithGoogleCredential(googlePendingCredential, {
                        checkedAtMs: legalCheckedAtMs,
                        docs: ["tos", "privacy", "license"],
                      });
                      if (authSessionTransport() === "bearer") {
                        if (!res.sessionToken) throw new Error("Missing session token.");
                        setSessionToken(res.sessionToken);
                      }
                      await refresh();
                      toast({ title: "Signed in" });
                      setGoogleNeedsLegal(false);
                      setGooglePendingCredential(null);
                    } catch (err) {
                      toast({ title: "Google sign-in failed", description: String(err) });
                    } finally {
                      setBusy(false);
                    }
                  }}
                  className="w-64 justify-self-center"
                >
                  Continue
                </Button>
              </div>
            ) : null}
            {googleStatus === "ready" ? (
              <div className="flex items-center gap-3 py-1">
                <div className="h-px flex-1 bg-border" />
                <div className="text-xs text-muted-foreground">or</div>
                <div className="h-px flex-1 bg-border" />
              </div>
            ) : googleStatus === "unavailable" ? (
              <div className="text-center text-xs text-muted-foreground">
                Google sign-in is not configured.
              </div>
            ) : (
              <div className="text-center text-xs text-muted-foreground">
                Loading Google sign-in…
              </div>
            )}
          </div>

          <Tabs defaultValue="signin" className="mt-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="signin">Sign in</TabsTrigger>
              <TabsTrigger value="register">Create account</TabsTrigger>
            </TabsList>

            <TabsContent value="signin">
              <form
                className="mt-4 grid gap-4"
                onSubmit={async (e) => {
                  e.preventDefault();
                  setBusy(true);
                  try {
                    await login(email, password);
                    toast({ title: "Signed in" });
                  } catch (err) {
                    toast({ title: "Sign-in failed", description: String(err) });
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <div className="grid gap-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <Button
                  type="submit"
                  disabled={busy}
                  className="w-64 justify-self-center"
                >
                  Sign in
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="register">
              <form
                className="mt-4 grid gap-4"
                onSubmit={async (e) => {
                  e.preventDefault();
                  setBusy(true);
                  try {
                    if (!legalCheckedAtMs) {
                      throw new Error("Please accept the Terms, Privacy Policy, and License.");
                    }
                    trackEvent("legal_consent_submitted", {
                      context: "email_register",
                      checked_at_ms: legalCheckedAtMs,
                      submitted_at_ms: Date.now(),
                    });
                    await register(email, password, {
                      checkedAtMs: legalCheckedAtMs,
                      docs: ["tos", "privacy", "license"],
                    });
                    toast({ title: "Account created" });
                    navigate("/");
                  } catch (err) {
                    toast({
                      title: "Registration failed",
                      description: String(err),
                    });
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <div className="grid gap-2">
                  <Label htmlFor="email2">Email</Label>
                  <Input
                    id="email2"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="password2">Password</Label>
                  <Input
                    id="password2"
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <div className="mt-2 flex items-start gap-3 rounded-lg border border-border/70 bg-muted/20 p-4 text-sm">
                  <Checkbox
                    id="legal-register"
                    checked={legalAccepted}
                    onCheckedChange={(next) => {
                      const isChecked = next === true;
                      setLegalAccepted(isChecked);
                      if (isChecked) {
                        const ts = Date.now();
                        setLegalCheckedAtMs(ts);
                        trackEvent("legal_consent_checked", {
                          context: "email_register",
                          checked_at_ms: ts,
                        });
                      } else {
                        setLegalCheckedAtMs(null);
                      }
                    }}
                  />
                  <label htmlFor="legal-register" className="leading-relaxed">
                    I have read and agree to the{" "}
                    <Link
                      to="/terms"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "tos", context: "email_register" })
                      }
                    >
                      Terms of Service
                    </Link>
                    ,{" "}
                    <Link
                      to="/privacy-policy"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "privacy", context: "email_register" })
                      }
                    >
                      Privacy Policy
                    </Link>
                    , and{" "}
                    <Link
                      to="/license"
                      className="text-primary hover:underline"
                      onClick={() =>
                        trackEvent("legal_link_click", { doc: "license", context: "email_register" })
                      }
                    >
                      License
                    </Link>
                    .
                  </label>
                </div>
                <Button
                  type="submit"
                  disabled={busy || !legalAccepted}
                  className="w-64 justify-self-center"
                >
                  Create account
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </Card>
      ) : (
        <div className="grid gap-6">
          {redactedReminder ? (
            <Alert>
              <AlertTitle>API access</AlertTitle>
              <AlertDescription>{redactedReminder}</AlertDescription>
            </Alert>
          ) : null}

          <Card className="p-6">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">API keys</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Use `X-API-Key` for API access. Keep keys secret — you can only
                  view a newly created key once.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => {
                    setBusy(true);
                    loadAccountData()
                      .catch((err) => {
                        toast({ title: "Failed to refresh", description: String(err) });
                      })
                      .finally(() => setBusy(false));
                  }}
                  disabled={busy}
                >
                  Refresh
                </Button>
                <Input
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="Key name (optional)"
                  className="w-48"
                />
                <Button
                  onClick={async () => {
                    setBusy(true);
                    try {
                      const created = await createApiKey(newKeyName || undefined);
                      setRevealedKey(created.apiKeyPlaintext);
                      setNewKeyName("");
                      await loadAccountData();
                    } catch (err) {
                      toast({
                        title: "Failed to create API key",
                        description: String(err),
                      });
                    } finally {
                      setBusy(false);
                    }
                  }}
                  disabled={busy}
                >
                  New key
                </Button>
              </div>
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-muted-foreground">
                  <tr className="border-b">
                    <th className="py-2 text-left font-medium">Name</th>
                    <th className="py-2 text-left font-medium">Prefix</th>
                    <th className="py-2 text-left font-medium">Created</th>
                    <th className="py-2 text-left font-medium">Last used</th>
                    <th className="py-2 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {apiKeys.length === 0 ? (
                    <tr>
                      <td className="py-3 text-muted-foreground" colSpan={5}>
                        No API keys yet.
                      </td>
                    </tr>
                  ) : (
                    apiKeys.map((k) => (
                      <tr key={k.id} className="border-b last:border-b-0">
                        <td className="py-3">{k.name ?? "—"}</td>
                        <td className="py-3 font-mono">{k.prefix}</td>
                        <td className="py-3">{formatDate(k.createdAt)}</td>
                        <td className="py-3">{formatDate(k.lastUsedAt)}</td>
                        <td className="py-3 text-right">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={async () => {
                              setBusy(true);
                              try {
                                await revokeApiKey(k.id);
                                await loadAccountData();
                              } catch (err) {
                                toast({
                                  title: "Failed to revoke key",
                                  description: String(err),
                                });
                              } finally {
                                setBusy(false);
                              }
                            }}
                            disabled={busy || !!k.revokedAt}
                          >
                            {k.revokedAt ? "Revoked" : "Revoke"}
                          </Button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-semibold">Usage (last 30 days)</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Total: {usageTotal.toLocaleString()} API requests.
            </p>

            <div className="mt-4 grid gap-2">
              {usageByDay.length === 0 ? (
                <div className="text-sm text-muted-foreground">No usage yet.</div>
              ) : (
                usageByDay.map((row) => (
                  <div
                    key={row.day}
                    className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                  >
                    <span className="font-mono text-xs">{row.day}</span>
                    <span className="text-sm">{row.count.toLocaleString()}</span>
                  </div>
                ))
              )}
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-semibold text-destructive">
              Delete account
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              This permanently deletes your account and revokes your API keys.
              Usage data is retained.
            </p>

            <div className="mt-4">
              <Button
                variant="destructive"
                onClick={() => {
                  setDeleteConfirmText("");
                  setDeleteDialogOpen(true);
                }}
                disabled={busy}
              >
                Delete account
              </Button>
            </div>
          </Card>
        </div>
      )}

      <AlertDialog
        open={!!revealedKey}
        onOpenChange={(open) => {
          if (!open) setRevealedKey(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Your new API key</AlertDialogTitle>
            <AlertDialogDescription>
              Copy this now — you won’t be able to view it again.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="relative rounded-md border bg-muted px-3 py-2 pr-12 font-mono text-sm break-all">
            <button
              type="button"
              onClick={() => {
                if (!revealedKey) return;
                void navigator.clipboard.writeText(revealedKey);
                setCopiedNewKey(true);
                toast({ title: "Copied to clipboard" });
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded bg-background p-1.5 shadow-sm border border-border hover:bg-accent"
              title="Copy to clipboard"
              aria-label="Copy API key"
            >
              {copiedNewKey ? (
                <Check className="h-3 w-3 text-green-600" />
              ) : (
                <Copy className="h-3 w-3 text-muted-foreground" />
              )}
            </button>
            {revealedKey}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Close</AlertDialogCancel>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete your account?</AlertDialogTitle>
            <AlertDialogDescription>
              Type <span className="font-mono">Delete</span> to confirm. This
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>

          <div className="grid gap-2">
            <Label htmlFor="delete-confirm">Confirmation</Label>
            <Input
              id="delete-confirm"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder="Type Delete"
              autoComplete="off"
            />
          </div>

          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <Button
              type="button"
              variant="destructive"
              disabled={busy || deleteConfirmText !== "Delete"}
              onClick={async () => {
                setBusy(true);
                try {
                  await deleteAccount({ confirm: deleteConfirmText });
                  toast({ title: "Account deleted" });
                  setDeleteDialogOpen(false);
                  logout();
                  navigate("/");
                } catch (err) {
                  toast({ title: "Delete failed", description: String(err) });
                } finally {
                  setBusy(false);
                }
              }}
            >
              Delete account
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageShell>
  );
}
